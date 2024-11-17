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
import seaborn as sns
import matplotlib.pyplot as plt

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
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.gpu_name = None
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
        project,
        api_token,
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
        # init neptune
        run = neptune.init_run(project=project, api_token=api_token)

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
        num_params = sum(p.numel() for p in model.parameters())

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
            if step % step_warmup == 0:
                return 1
            else:
                return (step % step_warmup) / step_warmup

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        # save the hyperparameters and configuration
        hyperparameters = {}
        hyperparameters["k_smote"] = k_smote
        hyperparameters["batch_size"] = batch_size
        hyperparameters["num_instances"] = num_instances
        hyperparameters["num_iterations"] = num_instances // (
            batch_size * step_accumulation
        )
        hyperparameters["learning_rate"] = learning_rate
        hyperparameters["step_warmup"] = step_warmup
        hyperparameters["step_accumulation"] = step_accumulation
        hyperparameters["emb_size"] = emb_size
        run["hyperparameters"] = hyperparameters

        configuration = {}
        configuration["seed"] = self.seed
        configuration["num_params"] = num_params
        configuration["n_gpus"] = self.n_gpus
        configuration["device"] = self.device
        configuration["gpu_name"] = self.gpu_name
        run["configuration"] = configuration

        # training attribute classification
        self.training_loop(
            run=run,
            dataloader_smote=dataloader_smote,
            model=model,
            step_accumulation=step_accumulation,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def training_loop(
        self,
        run: neptune.init_run,
        dataloader_smote: DataLoader,
        dataloader_train_attribute: DataLoader,
        dataloader_test_attribute: DataLoader,
        model: nn.Module,
        step_accumulation: int,
        loss: AdaCosLoss,
        optimizer: torch.optim.AdamW,
        scheduler: LambdaLR,
    ):
        """
        training loop for smote data
        """

        # step report and evaluation
        step_eval = step_accumulation * 10

        # loss train
        loss_smote_total = 0

        for iter, (X_smote, y_smote) in enumerate(dataloader_smote):

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

            # gradient accumulated and report loss
            if (iter + 1) % step_accumulation == 0:

                # update model weights and zero grad
                optimizer.step()
                optimizer.zero_grad()

                # report loss
                loss_smote_total = loss_smote_total / step_accumulation
                print("loss_smote_total {}".format(loss_smote_total))
                run["loss_smote_total"].append(loss_smote_total, step=iter)
                loss_smote_total = 0

            # update scheduler
            scheduler.step()

            # report the loss and evaluation mode every 1000 iteration
            if (iter + 1) % step_eval == 0:

                # evaluation mode for train data attribute
                self.evaluation_mode(
                    run=run,
                    iter=iter,
                    model=model,
                    loss=loss,
                    dataloader_attribute=dataloader_train_attribute,
                    type_labels=["train_source_normal", "train_target_normal"],
                )

                # evaluation mode for test data attribute
                self.evaluation_mode(
                    run=run,
                    iter=iter,
                    model=model,
                    loss=loss,
                    dataloader_attribute=dataloader_train_attribute,
                    type_labels=[
                        "test_source_normal",
                        "test_target_normal",
                        "test_source_anomaly",
                        "test_target_anomaly",
                    ],
                )

            print("iter", iter)
            current_lr = optimizer.param_groups[0]["lr"]
            print("current_lr:", current_lr)
            run["current_lr"].append(current_lr, step=iter)

    def evaluation_mode(
        self,
        run: neptune.init_run,
        iter: int,
        model: nn.Module,
        loss: AdaCosLoss,
        dataloader_attribute: DataLoader,
        type_labels=["train_source_normal", "train_target_normal"],
    ):
        """
        evaluation mode in training loop
        """
        # model in evaluation and no grad mode
        model.eval()
        loss.eval()

        # saved array
        len_train = dataloader_attribute.dataset.tensors[0].shape[0]
        y_pred_train_label_array = np.empty(shape=(len(len_train),))
        y_train_array = np.empty(shape=(len(len_train), 3))

        with torch.no_grad():
            for iter, (X_train, y_train) in enumerate(dataloader_attribute):

                # data to device
                X_train = X_train.to(self.device)

                # forward pass
                embedding_train = model(X_train)

                # pred the label
                y_pred_train_label = loss.pred_labels(embedding=embedding_train)

                # save to array
                y_train_array[iter * batch_size : iter * batch_size + batch_size] = (
                    y_train.cpu().numpy()
                )
                y_pred_train_label_array[
                    iter * batch_size : iter * batch_size + batch_size
                ] = y_pred_train_label.cpu().numpy()

            # calculate accuracy
            type_data = type_labels[0].split("_")[0]
            accuracy_type_labels = self.accuracy_calculation(
                y_train_array=y_train_array,
                y_pred_train_label_array=y_pred_train_label_array,
                type_labels=type_labels,
            )

            # calculate confusion matrix
            cm = confusion_matrix(
                y_true=y_train_array[:, 1], y_pred=y_pred_train_label_array
            )
            cm_img = self.plot_confusion_matrix(cm=cm, type_data=type_data)

            # save metrics in run
            for typ_l, acc in zip(type_labels, accuracy_type_labels):
                run["{}/accuracy_{}".format(type_data, typ_l)].append(acc, step=iter)

            run["{}/confusion_matrix".format(type_data)].append(acc, step=iter)
            plt.close()

    def accuracy_calculation(
        self,
        y_train_array: np.array,
        y_pred_train_label_array: np.array,
        type_labels=["train_source_normal", "train_target_normal"],
    ):
        """
        get the accuracy given y_train_array shape (index, attribute, condition)
        and y_pred_train_label_array shape (pred_attribute)
        """
        # get the indices
        indices = self.get_indices(
            y_true_array=y_train_array,
            type_labels=type_labels,
        )
        y_train_array = y_train_array[:, 1]

        # calculate accuracy
        accuracy = []
        for idx in indices:
            y_pred = y_pred_train_label_array[idx]
            y_true = y_train_array[idx]
            acc = accuracy_score(y_pred=y_pred, y_true=y_true)
            accuracy.append(acc)

        return accuracy

    def get_indices(
        self,
        y_true_array: np.array,
        type_labels=["train_source_normal", "train_target_normal"],
    ):
        """
        get indices of the domain or condition or type data given the
        y_train_array shape (index, attribute, condition) and y_pred_train_label_array shape (pred_attribute)
        """

        # y_true_array only consider the first column
        y_true_array = y_true_array[:, 0]

        # get the id from type_labels
        ts_ids = [self.indices_timeseries_analysis[typ] for typ in type_labels]

        # get the index of each id
        indices = []
        for id in ts_ids:
            idx = np.where(np.isin(y_true_array, id))[0]
            indices.append(idx)

        return indices

    def plot_confusion_matrix(self, cm, type_data="train"):
        """
        plot the confusion matrix
        """
        fig = plt.figure(figsize=(35, 16))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        titel = "Confusion Matrix {}".format(type_data)
        plt.title(titel, fontsize=18)
        plt.xlabel("Predicted Labels", fontsize=15)
        plt.ylabel("True Labels", fontsize=15)

        return fig


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
    project = "DCASE2024/wav-test"
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
    k_smote = 5
    batch_size = 8
    num_instances = 320000
    learning_rate = 0.0001
    step_warmup = 120
    step_accumulation = 32
    emb_size = None

    ad.anomaly_detection(
        project=project,
        api_token=api_token,
        k_smote=k_smote,
        batch_size=batch_size,
        num_instances=num_instances,
        learning_rate=learning_rate,
        step_warmup=step_warmup,
        step_accumulation=step_accumulation,
        emb_size=emb_size,
    )
