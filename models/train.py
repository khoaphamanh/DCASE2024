import torch
from torch import nn
from beats.beats_custom import BEATsCustom
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import numpy as np
from loss import AdaCosLoss
from torchinfo import summary
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import neptune
from torch.optim.lr_scheduler import LambdaLR

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing


# class Anomaly Detechtion
class AnomalyDetection(DataPreprocessing):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

        # set the seed
        torch.manual_seed(seed)

        # information this class
        self.path_models_directory = os.path.dirname(os.path.abspath(__file__))

        # pretrained models BEATs
        self.path_pretrained_models_directory = os.path.join(
            self.path_models_directory, "pretrained_models"
        )
        if os.path.exists(self.path_pretrained_models_directory):
            import download_models

        self.path_beat_iter3_state_dict = os.path.join(
            self.path_pretrained_models_directory, "BEATs_iter3.pt"
        )

        # configuration of the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count()

    def load_model(self, input_size=12, emb_size=None):
        # function to load model beats
        model = BEATsCustom(
            path_state_dict=self.path_beat_iter3_state_dict,
            input_size=input_size,
            emb_size=emb_size,
        )
        return model

    def load_dataset_tensor(self, k_smote):
        """
        load data smote and train, test data as Tensor
        """
        # load data smote and convert to tensor
        train_data_smote, train_label_smote = self.smote(k_smote=k_smote)

        # convert to tensor
        train_data_smote = torch.tensor(train_data_smote)
        train_label_smote = torch.tensor(train_label_smote)

        dataset_smote = TensorDataset(train_data_smote, train_label_smote)

        # load raw data attribute
        (
            train_dataset_attribute,
            train_label_attribute,
            test_dataset_attribute,
            test_label_attribute,
        ) = self.load_data_attribute()

        # convert to tensor
        train_dataset_attribute = torch.tensor(train_dataset_attribute)
        train_label_attribute = torch.tensor(train_label_attribute)

        train_dataset_attribute = TensorDataset(
            train_dataset_attribute, train_label_attribute
        )

        test_dataset_attribute = torch.tensor(test_dataset_attribute)
        test_label_attribute = torch.tensor(test_label_attribute)

        test_dataset_attribute = TensorDataset(
            test_dataset_attribute, test_label_attribute
        )

        return dataset_smote, train_dataset_attribute, test_dataset_attribute

    def data_loader(
        self, dataset, batch_size, num_instances=None, uniform_sampling=False
    ):
        """
        convert tensor data to dataloader
        """
        # check if uniform_sampling
        if uniform_sampling and isinstance(num_instances, int):

            # split to get the label
            _, y_train_smote = dataset.tensors

            class_instances_count = torch.tensor(
                [(y_train_smote == l).sum() for l in torch.unique(y_train_smote)]
            )
            weight = 1.0 / class_instances_count
            instance_weight = torch.tensor([weight[l] for l in y_train_smote])

            # batch uniform sampling
            sampler = WeightedRandomSampler(
                weights=instance_weight,
                num_samples=num_instances,
            )

            dataloader = DataLoader(
                dataset=dataset, sampler=sampler, batch_size=batch_size
            )

        else:
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

        return dataloader

    def anomaly_detection(
        self,
        k_smote,
        batch_size,
        num_instances,
        learning_rate,
        step_warmup,
        step_accumulation,
        emb_size=None,
    ):
        """
        main function to find the result
        """

        # load data
        dataset_smote, train_dataset_attribute, test_dataset_attribute = (
            self.load_dataset_tensor(k_smote=k_smote)
        )

        # dataloader
        dataloader_smote = self.data_loader(
            dataset=dataset_smote,
            batch_size=batch_size,
            num_instances=num_instances,
            uniform_sampling=True,
        )
        dataloader_train_attribute = self.data_loader(
            dataset=train_dataset_attribute, batch_size=batch_size
        )
        dataloader_test_attribute = self.data_loader(
            dataset=test_dataset_attribute, batch_size=batch_size
        )

        # load model
        input_size = dataloader_train_attribute.dataset.tensors[0].shape[1] // self.fs
        model = self.load_model(input_size=input_size, emb_size=emb_size)

        # model to device
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

        # loss
        if emb_size == None:
            emb_size = model.embedding_asp
        loss = AdaCosLoss(num_classes=self.num_classes_attribute(), emb_size=emb_size)

        # optimizer
        parameters = list(model.parameters()) + list(loss.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

        # scheduler
        def lr_lambda(step):
            """
            function to reset learning rate after warmup_steps, lr increase from very small (step 1) to max_lr (warmup_step).
            lr very small (warmup_step + 1) to lr_max (warmup_step*2)
            default step is 1
            """
            if step % 10 == 0:
                return 1
            else:
                return (step % step_warmup) / step_warmup

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # training attribute classification
        self.training_loop(
            dataloader_smote=dataloader_smote,
            model=model,
            step_accumulation=step_accumulation,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def training_loop(
        self,
        dataloader_smote: DataLoader,
        model: nn.Module,
        step_accumulation: int,
        loss: AdaCosLoss,
        optimizer: torch.optim.AdamW,
        scheduler: LambdaLR,
    ):
        """
        training loop for smote data
        """

        # loss train
        loss_smote_total = 0

        for iter, (X_smote, y_smote) in dataloader_smote:

            # model in traning model
            model.train()
            loss.train()

            # data to device
            X_smote = X_smote.to(self.device)
            y_smote = y_smote.to(self.device)

            # forward pass
            embedding_smote = model(X_smote)

            # calculate the loss
            loss_smote = loss(embedding_smote, y_smote)
            loss_smote_total = loss_smote_total + loss_smote.item()

            # update the loss
            loss_smote.backward()

            # update scheduler
            scheduler.step()

            # gradient accumulated
            if (iter + 1) % step_accumulation == 0:

                # update model weights and zero grad
                optimizer.step()
                optimizer.zero_grad()

            # report the loss and evaluation mode every 1000 iteration
            if (iter + 1) % 1000 == 0:
                loss_smote_total = loss_smote_total / 1000
                print("loss_smote_total {}".format(loss_smote_total))
                loss_smote_total = 0

            print("iter", iter)
            current_lr = optimizer.param_groups[0]["lr"]
            print("current_lr:", current_lr)


# run this script
if __name__ == "__main__":

    # create the seed
    seed = 1998
    develop_name = "develop"

    ad = AnomalyDetection(data_name=develop_name, seed=seed)

    # path_beat_iter3_state_dict = ad.path_beat_iter3_state_dict
    # print("path_beat_iter3_state_dict:", path_beat_iter3_state_dict)

    # model = BEATsCustom(path_state_dict=path_beat_iter3_state_dict)

    # a = torch.randn(2, 496, 768)
    # asp = AttentiveStatisticsPooling(input_size=10)

    # out = asp(a)
    # print("out shape:", out.shape)

    # a = torch.randn(8, 10 * 16000)

    # # # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # # # a = a.to(device)

    # model = BEATsCustom(
    #     path_state_dict=path_beat_iter3_state_dict, input_size=10, emb_size=None
    # )
    # # # model = model.to(device)

    # out = model(a)
    # print("out shape:", out.shape)
    # out = out.to(device)
    # true = torch.rand_like(out)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # criterion = nn.MSELoss()  # Mean Squared Error loss (regression)
    # loss = criterion(out, true)
    # loss.backward()
    # optimizer.step()
    # print("out shape:", out.shape)
    # summary(model)

    # Check the dtype of the weights for each parameter
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, dtype: {param.dtype}")

    # train_data, train_label, test_data, test_label = ad.load_raw_data()
    # print("train_data:", train_data.shape)
    # print("train_data", train_data.dtype)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("train_data", test_data.dtype)
    # print("test_label:", test_label.shape)

    # train_data, train_label, test_data, test_label = ad.load_data_attribute()

    # print("train_data:", train_data.shape)
    # print("train_data", train_data.dtype)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("train_data", test_data.dtype)
    # print("test_label:", test_label.shape)

    # label_train_unique, count = np.unique(train_label[:, 1], return_counts=True)
    # print("label_train_unique:", label_train_unique)
    # print("count:", count)

    # label_test_unique, count = np.unique(test_label[:, 1], return_counts=True)
    # print("label_test_unique:", label_test_unique)
    # print("count:", count)

    # label_test_unique, count = np.unique(test_label[:, 1], return_counts=True)
    # print("label_test_unique:", label_test_unique)
    # print("count:", count)

    # print("train_data:", train_data.shape)
    # print("train_data", train_data.dtype)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("train_data", test_data.dtype)
    # print("test_label:", test_label.shape)

    # train_data_smote, train_label_smote = ad.smote()
    # print("train_data_smote shape:", train_data_smote.shape)
    # print("train_label_smote shape:", train_label_smote.shape)

    # label_train_unique, count = np.unique(train_label_smote, return_counts=True)
    # print("label_train_unique:", label_train_unique)
    # print("count:", count)

    # print(ad.fs)
    # print(ad.data_name)
    # print(ad.seed)

    # dataset_smote, train_dataset_attribute, test_dataset_attribute = (
    #     ad.load_dataset_tensor()
    # )

    # dataloader_smote = ad.data_loader(
    #     dataset=dataset_smote, batch_size=8, num_instances=10000, uniform_sampling=True
    # )
    # analys = []

    # # Print the sampled batches
    # for data, labels in dataloader_smote:

    #     # print("Labels:", labels)
    #     analys.append(labels.tolist())

    # analys = np.array(analys).ravel()

    # # Calculate the unique elements and their counts
    # unique_elements, counts = np.unique(analys, return_counts=True)

    # # Calculate the probability of each unique element
    # probabilities = counts / len(analys)
    # print("probabilities:", probabilities)

    """
    test model anomaly detection
    """
    k_smote = 5
    batch_size = 8
    num_instances = 320000
    learning_rate = 0.0001
    step_warmup = 120
    step_accumulation = 32
    emb_size = None

    ad.anomaly_detection(
        k_smote=k_smote,
        batch_size=batch_size,
        num_instances=num_instances,
        learning_rate=learning_rate,
        step_warmup=step_warmup,
        step_accumulation=step_accumulation,
        emb_size=emb_size,
    )
