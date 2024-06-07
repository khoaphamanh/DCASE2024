from transformers import Wav2Vec2ForCTC
import numpy as np
import torch
import sys
import os
from torch import nn
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, TensorDataset
import neptune
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from neptune.utils import stringify_unsupported
from sklearn.neighbors import NearestNeighbors

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing
import utils
from metrics import AdaCosLoss, ArcFaceLoss


# fine-tune model wav2vec
class Wav2VecXLR300MCustom(nn.Module):
    def __init__(
        self,
        model_name: str,
        emb_size: int,
        output_size: int,
        classifier_head=True,
        window_size=None,
    ):
        super().__init__()
        if window_size == None:
            window_size = 16000

        flatten_dim = int((((window_size / 16000) * 50) - 1)) * 32

        self.pre_trained_wav2vec = Wav2Vec2ForCTC.from_pretrained(model_name)
        if classifier_head:
            self.out_layer = nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=flatten_dim, out_features=emb_size),
                nn.ReLU(),
                nn.Linear(in_features=emb_size, out_features=output_size),
            )
        else:
            self.out_layer = nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=flatten_dim, out_features=emb_size),
            )

    def forward(self, x):
        x = self.pre_trained_wav2vec(x).logits
        x = self.out_layer(x)
        return x


# main class anomaly detection
class AnomalyDetection:
    def __init__(
        self,
        data_preprocessing: DataPreprocessing,
        seed,
    ):

        # data preprocessing
        self.data_preprocessing = data_preprocessing
        self.domain_dict = data_preprocessing.domain_to_number()
        self.data_name = data_preprocessing.data_name
        self.unique_labels_dict = data_preprocessing.load_unique_labels_dict()
        self.unique_labels_train = self.unique_labels_dict["train"]
        self.unique_labels_test = self.unique_labels_dict["test"]
        self.num_classes_train = len(self.unique_labels_train)

        # time series information
        self.fs = data_preprocessing.fs

        # model configuration parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count()
        self.seed = seed
        self.train_size = 0.8
        self.num_workers = 0

    def load_data(self, window_size=None, hop_size=None):
        """
        Load train data with labels are attributes given window size and hop size. Defaulf is 16000 for hop size and 32000 for window size
        """
        # load data
        X_train, y_train, X_test, y_test = self.data_preprocessing.load_data(
            window_size=window_size, hop_size=hop_size, train=True, test=True
        )

        return X_train, y_train, X_test, y_test

    def standardize(self, X_train, X_val):
        """
        Use standardize to scale the data
        """
        # load scaler
        scaler = StandardScaler()

        # reshape train data
        n_samples_train = X_train.shape[0]
        n_samples_val = X_val.shape[0]

        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)

        # fit the scaler
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        X_train = X_train.reshape(n_samples_train, -1)
        X_val = X_val.reshape(n_samples_val, -1)

        return X_train, X_val

    def data_loader(self, batch_size: int, window_size=None, hop_size=None):
        """
        Turn data to pytorch,split it to train val and turn it to dataloader
        """
        # load data
        X_train, y_train, X_test, y_test = self.load_data(
            window_size=window_size, hop_size=hop_size
        )

        X_train, X_test = self.standardize(X_train=X_train, X_val=X_test)

        len_train = len(X_train)
        len_test = len(X_test)

        # compute the class weights
        self.class_weights = torch.tensor(
            class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
        ).float()

        # dataloader
        train_data = TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train).long()
        )
        val_data = TensorDataset(
            torch.tensor(X_test).float(), torch.tensor(y_test).long()
        )

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers
        )

        return train_loader, val_loader, len_train, len_test

    def plot_confusion_matrix(self, cm, name="train"):
        """
        plot the confusion matrix
        """
        fig = plt.figure(figsize=(35, 16))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        titel = "Confusion Matrix {}".format(name)
        plt.title(titel, fontsize=18)
        plt.xlabel("Predicted Labels", fontsize=15)
        plt.ylabel("True Labels", fontsize=15)

        return fig

    def get_source_target_indices(self, y_true):
        """
        get the source and target indices given y_true
        """
        # get the indices of the source and target in y_true
        source_labels = self.domain_dict["train_source"]
        target_labels = self.domain_dict["train_target"]
        source_indices = np.where(np.isin(y_true, source_labels))[0]
        target_indices = np.where(np.isin(y_true, target_labels))[0]

        return source_indices, target_indices

    def get_normal_anomaly_indices(self, y_true):
        """
        get the normal and anomaly indices given y_true
        """
        # get the indices of the source and target in y_true
        normal_labels = self.domain_dict["test_normal"]
        anomaly_labels = self.domain_dict["test_anomaly"]
        normal_indices = np.where(np.isin(y_true, normal_labels))[0]
        anomaly_indices = np.where(np.isin(y_true, anomaly_labels))[0]

        return normal_indices, anomaly_indices

    def embedding_source_target(self, embedding_train_array, y_true):
        """
        split embedding source and target given embedding array and y_true array
        """
        # get the indices of the source and target in y_true
        source_indices, target_indices = self.get_source_target_indices(y_true=y_true)

        # split the embedding
        embedding_source = embedding_train_array[source_indices]
        embedding_target = embedding_train_array[target_indices]

        return embedding_source, embedding_target

    def accuracy_source_target(self, y_pred_label, y_true):
        """
        calculate the accuracy source and target in train data given y_true and y_pred_label
        """
        # get the indices of the source and target in y_true
        source_indices, target_indices = self.get_source_target_indices(y_true=y_true)

        # get the y_true and y_pred of source and targets
        y_pred_source = y_pred_label[source_indices]
        y_pred_target = y_pred_label[target_indices]

        y_true_source = y_true[source_indices]
        y_true_target = y_true[target_indices]

        accuracy_train_source_this_batch = accuracy_score(
            y_pred=y_pred_source, y_true=y_true_source
        )
        accuracy_train_target_this_batch = accuracy_score(
            y_pred=y_pred_target, y_true=y_true_target
        )

        return accuracy_train_source_this_batch, accuracy_train_target_this_batch

    def find_domain_knn(
        self,
        knn_source,
        knn_target,
        embedding_test_array,
        embedding_train_array,
        y_test_cm,
        percentile,
    ):
        """
        find the domain of a given embedding test array
        """
        # split y_test to index of original time series and label
        indices_timeseries = y_test_cm[:, 0]
        labels = y_test_cm[:, 1]

        # find the distance to source and target
        distance_source, _ = knn_source.kneighbors(embedding_test_array)
        distance_target, _ = knn_target.kneighbors(embedding_test_array)
        distance_concat = np.stack((distance_source, distance_target))

        # calculate argmax
        argmin_distance_all_window = np.argmin(distance_concat, axis=0)

        # get the each time series index from window index. {index_timeseries:index_windows_of_this_timeseries}
        unique_indices_ts = np.unique(indices_timeseries)
        indices_ts_to_window_dict = {
            element: np.where(indices_timeseries == element)[0]
            for element in unique_indices_ts
        }

        # get the domain for each time series: source 0, target 1
        domain = {}
        mahalobis_distance = {}
        for idx_ts, idx_w in indices_ts_to_window_dict.items():

            # argmin of each window in a timeseries
            argmin_distance_window_of_ts = argmin_distance_all_window[idx_w]
            domain_votes, count_votes = np.unique(
                argmin_distance_window_of_ts, return_counts=True
            )

            # get the index for each votes {0:index of 0, 1:index of 1}
            domain_votes_dict = {
                element: np.where(argmin_distance_window_of_ts == element)[0]
                for element in domain_votes
            }

            # if source domain has more votes
            if count_votes[0] > count_votes[1]:
                domain = domain_votes[0]

            # if target domain has more votes
            elif count_votes[1] > count_votes[0]:
                domain = domain_votes[1]

            # if target domain and source domain have same votes
            elif count_votes[1] == count_votes[0]:
                1

            source_votes_index = domain_votes_dict[domain]

    def train_test_loop(
        self,
        project: str,
        api_token: str,
        model_name: str,
        batch_size: int,
        emb_size: int,
        lr: float,
        wd: int,
        epochs: int,
        k: int,
        percentile: float,
        loss_name="adacos",
        optimizer_name="AdamW",
        classifier_head=True,
        scale=None,
        margin=None,
        window_size=None,
        hop_size=None,
        distance=None,
    ):
        """
        Train test loop
        """
        # init run
        run = neptune.init_run(project=project, api_token=api_token)

        # load dataloader
        train_loader, test_loader, len_train, len_test = self.data_loader(
            batch_size=batch_size, window_size=window_size, hop_size=hop_size
        )

        # save parameter in neptune
        if window_size == None:
            window_size = self.fs * 2
        if hop_size == None:
            hop_size = self.fs

        hyperparameters = {
            "batch_size": batch_size,
            "emb_size": emb_size,
            "lr": lr,
            "window_size": window_size,
            "hop_size": hop_size,
            "weight_decay": wd,
            "loss_name": loss_name,
            "optimizer_name": optimizer_name,
            "model_nane": model_name,
        }
        if loss_name in ["arcface", "adacos"]:
            hyperparameters["classifier_head"] = classifier_head
            if loss_name == "arcface":
                hyperparameters["scale"] = scale
                hyperparameters["margin"] = margin

        configurations = {
            "seed": self.seed,
            "device": self.device,
            "n_gpus": self.n_gpus,
            "num_workers": self.num_workers,
            "data_name": self.data_name,
        }

        run["configurations"] = stringify_unsupported(configurations)
        run["hyperparameters"] = hyperparameters

        # init model neural network
        model = Wav2VecXLR300MCustom(
            model_name=model_name,
            emb_size=emb_size,
            output_size=self.num_classes_train,
            classifier_head=classifier_head,
            window_size=window_size,
        )

        # init model knn
        if distance == None:
            distance = "cosine"
        knn_source = NearestNeighbors(n_neighbors=k, metric=distance)
        knn_target = NearestNeighbors(n_neighbors=k, metric=distance)

        # if multiple gpus
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

        # loss
        if loss_name == "arcface":
            loss = ArcFaceLoss(
                num_classes=self.num_classes_train,
                emb_size=emb_size,
                class_weights=self.class_weights,
                scale=scale,
                margin=margin,
            )

        elif loss_name == "adacos":
            loss = AdaCosLoss(
                num_classes=self.num_classes_train,
                emb_size=emb_size,
                class_weights=self.class_weights,
            )

        # optimizer
        parameters = list(model.parameters()) + list(loss.parameters())

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(params=parameters, lr=lr, weight_decay=wd)

        elif optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(params=parameters, lr=lr, weight_decay=wd)

        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(params=parameters, lr=lr, weight_decay=wd)

        # training loop:
        for ep in range(epochs):

            # metrics init
            loss_train = 0
            accuracy_train = 0
            accuracy_train_source = 0
            accuracy_train_target = 0
            f1_train = 0

            accuracy_test = 0

            # confustion matrix
            y_train_cm = np.empty(shape=(len_train,))
            y_pred_train_cm = np.empty(shape=(len_train,))

            y_test_cm = np.empty(shape=(2, len_test))
            y_pred_test_cm = np.empty(shape=(len_test,))

            # embedding array init
            embedding_train_array = np.empty_like(shape=(len_train, emb_size))
            embedding_test_array = np.empty_like(shape=(len_test, emb_size))

            # training mode
            model.train()
            for batch_train, (X_train, y_train) in enumerate(train_loader):

                # to device
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass
                embedding_train = model(X_train)
                embedding_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = embedding_train.cpu().numpy()

                y_pred_train_logit = loss.logits(
                    embedding=embedding_train, y_true=y_train
                )
                y_pred_train_label = y_pred_train_logit.argmax(dim=1)

                # calculate the loss, accuracy, f1 score and confusion matrix
                # loss
                loss_train_this_batch = loss(embedding_train, y_train)
                loss_train = loss_train + loss_train_this_batch.item()

                # accuracy train
                accuracy_train_this_batch = accuracy_score(
                    y_pred=y_pred_train_label.cpu().numpy(),
                    y_true=y_train.cpu().numpy(),
                )
                accuracy_train = accuracy_train + accuracy_train_this_batch

                accuracy_train_source_this_batch, accuracy_train_target_this_batch = (
                    self.accuracy_source_target(
                        y_pred_label=y_pred_train_label.cpu().numpy(),
                        y_true=y_train.cpu().numpy(),
                    )
                )
                accuracy_train_source = (
                    accuracy_train_source + accuracy_train_source_this_batch
                )
                accuracy_train_target = (
                    accuracy_train_target + accuracy_train_target_this_batch
                )

                # f1 train
                f1_train_this_batch = f1_score(
                    y_pred=y_pred_train_label.cpu().numpy(),
                    y_true=y_train.cpu().numpy(),
                    average="weighted",
                )
                f1_train = f1_train + f1_train_this_batch

                # confusion matrix train
                y_train_cm[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_train.cpu().numpy()
                y_pred_train_cm[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_pred_train_label.cpu().numpy()

                # gradient decent, backpropagation and update parameters
                optimizer.zero_grad()
                loss_train_this_batch.backward()
                optimizer.step()

            # evaluation mode
            model.eval()
            with torch.inference_mode():
                for batch_test, (X_test, y_test) in enumerate(test_loader):

                    # to device
                    X_test = X_test.to(self.device)

                    # forward pass
                    embedding_test = model(X_test)
                    embedding_test_array[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = embedding_test.cpu().numpy()

                    # all y test
                    y_test_cm[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = y_test.cpu().numpy()

            # fit embedding to knn models
            embedding_train_source, embedding_train_target = (
                self.embedding_source_target(
                    embedding_train_array=embedding_train_array, y_true=y_train_cm
                )
            )
            knn_source.fit(embedding_train_source)
            knn_target.fit(embedding_train_target)

            # print out the metrics
            loss_train = loss_train / len(train_loader)
            accuracy_train = accuracy_train / len(train_loader)
            accuracy_train_target = accuracy_train_target / len(train_loader)
            accuracy_train_target = accuracy_train_target / len(train_loader)
            f1_train = f1_train / len(train_loader)

            accuracy_test = accuracy_test / len(test_loader)
            f1_test = f1_test / len(test_loader)

            cm_train = confusion_matrix(y_true=y_train_cm, y_pred=y_pred_train_cm)
            cm_val = confusion_matrix(y_true=y_test_cm, y_pred=y_pred_test_cm)

            print("epoch {}".format(ep))
            print(
                "loss train = {:.4f}, accuracy train = {:.4f}, f1 train = {:.4f}".format(
                    loss_train, accuracy_train, f1_train
                )
            )
            print(
                "accuracy train source = {:.4f}, accuracy train target = {:.4f}".format(
                    accuracy_train_source, accuracy_train_target
                )
            )
            print(
                "accuracy test = {:.4f},  f1 test = {:.4f}".format(
                    accuracy_test, f1_test
                )
            )

            # log the metrics in neptune
            metrics = {
                "loss_train": loss_train,
                "accuracy_train": accuracy_train,
                "accuracy_train_source": accuracy_train_source,
                "accuracy_train_target": accuracy_train_target,
                "f1_train": f1_train,
                "accuracy_test": accuracy_test,
                "f1_test": f1_test,
            }

            run["metrics"].append(metrics, step=ep)

            cm_train_fig = self.plot_confusion_matrix(cm=cm_train, name="train")
            plt.close()
            cm_val_fig = self.plot_confusion_matrix(cm=cm_val, name="val")
            plt.close()
            run["metrics/confusion_matrix_train"].append(cm_train_fig, step=ep)
            run["metrics/confusion_matrix_val"].append(cm_val_fig, step=ep)

        # running time
        run["runing_time"] = run["sys/running_time"]

        # end log neptune
        run.stop()


if __name__ == "__main__":

    # set the seed
    seed = utils.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("\n")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # hyperparameters
    lr = utils.lr_np
    emb_size = utils.emb_size_np
    batch_size = utils.batch_size_np
    wd = utils.wd_np
    epochs = utils.epochs_np
    optimizer_name = utils.optimizer_name_np
    model_name = utils.model_name_np
    scale = utils.scale_np
    margin = utils.margin_np
    loss_name = utils.loss_name_np
    classifier_head = utils.classifier_head_np
    window_size = utils.window_size_np
    hop_size = utils.hop_size_np

    # general hyperparameters
    project = utils.project
    api_token = utils.api_token
    data_name = utils.data_name_dev

    # data preprocessing
    data_preprocessing = DataPreprocessing(data_name=data_name)
    anomaly_detection = AnomalyDetection(
        data_preprocessing=data_preprocessing, seed=seed
    )

    # train model
    anomaly_detection.train_test_loop(
        model_name=model_name,
        project=project,
        api_token=api_token,
        batch_size=batch_size,
        emb_size=emb_size,
        lr=lr,
        wd=wd,
        epochs=epochs,
        scale=scale,
        margin=margin,
        classifier_head=classifier_head,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        window_size=window_size,
        hop_size=hop_size,
    )
    # train_data, train_label = anomaly_detection.load_train_data()
    # print("train_data shape:", train_data.shape)

    # unique = anomaly_detection.num_classes_train
    # print("unique:", unique)
    # processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")

    # test = train_data[0:5]
    # print("test shape:", test.shape)
    # test = torch.tensor(test)
    # # test = processor(test, return_tensors="pt", sampling_rate=fs).input_values
    # print("test shape:", test.shape)
    # # print("test:", test)
    # fs = 16000
    # model = Wav2VecXLR300MCustom(fs=fs, emb_size=emb_size, output_size=67)
    # with torch.inference_mode():
    #     out = model(test)
    #     print("out shape:", out.shape)
    #     print("out shape:", out)
