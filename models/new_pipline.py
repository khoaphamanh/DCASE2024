from transformers import Wav2Vec2ForCTC
import numpy as np
import torch.nn.functional as F
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
from sklearn.metrics import roc_curve, auc
from torchaudio.transforms import SpeedPerturbation

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing_new import DataPreprocessing
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

        self.classifier_head = classifier_head
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
        if self.classifier_head == False:
            x = F.normalize(x)
        return x


# main class anomaly detection
class AnomalyDetection:
    def __init__(
        self,
        data_name,
        seed,
    ):

        # data preprocessing
        self.data_preprocessing = DataPreprocessing(data_name)
        self.data_name = self.data_preprocessing.data_name
        self.num_classes_train = self.data_preprocessing.num_classes_train
        self.timeseries_analysis = self.data_preprocessing.timeseries_analysis
        self.labels_num_to_string = self.data_preprocessing.labels_num_to_string

        # time series information
        self.fs = self.data_preprocessing.fs

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

    def standardize(self, X_train, X_test):
        """
        Use standardize to scale the data
        """
        # load scaler
        scaler = StandardScaler()

        # reshape train data
        n_samples_train = X_train.shape[0]
        n_samples_val = X_test.shape[0]

        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

        # fit the scaler
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = X_train.reshape(n_samples_train, -1)
        X_test = X_test.reshape(n_samples_val, -1)

        return X_train, X_test

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

    def speed_perturb(self, X, speed_factors):
        """
        use augmentation as purturb the speed of the input given the speed factors and then down/upsampling the timeseries using cutting, padding
        """
        # load the speed purturb as the augmentation
        augmentation = SpeedPerturbation(self.fs, speed_factors)

        # use the augmentation for inputs
        len_input = X.shape[-1]
        X_augmented = augmentation(X)[0]
        len_output = X_augmented.shape[-1]

        # downsampling: cutting
        if len_output > len_input:
            start_idx = torch.randint(0, len_output - len_input, size=(1,)).item()
            X_augmented = X_augmented[:, start_idx : start_idx + len_input]

        # upsampling: padding
        else:
            X_augmented = F.pad(
                X_augmented, (0, len_input - len_output), mode="constant", value=0
            )

        return X_augmented

    def get_indices(self, y_true_array=None, y_pred_array=None, type_labels=None):
        """
        get the source, target, normal, anomaly indices from y_array (could be y_true or y_pred)
        """
        # indices from y_true:
        if y_true_array != None:
            # labels for domain: source, target, normal, anomaly
            ts_domain = [self.timeseries_analysis[type] for type in type_labels]

            # get the indices of each labels
            y_true_array = y_true_array[:, 0]
            indices = []
            for ts in ts_domain:
                idx = np.where(np.isin(y_true_array, ts))[0]
                indices.append(idx)

            return indices

        elif y_pred_array != None:
            # domain labels:
            labels = [self.labels_num_to_string[type] for type in type_labels]
            indices = []
            for lb in labels:
                idx = np.where(y_pred_array == lb)[0]
                indices.append(idx)

            return indices

    def loss_source_target(self, embedding_array, y_true_array, loss, type_labels):
        """
        calculate the loss source, target given y_array
        """
        # get the indices
        indices = self.get_indices(y_true_array=y_true_array, type_labels=type_labels)

        # calculate the losses
        y_true_array = y_true_array[:, 1]
        losses = []
        for idx in indices:
            embedding_domain = embedding_array[idx]
            y_true_domain = y_true_array[idx]
            loss_domain = loss.calculate_loss(embedding_domain, y_true_domain)
            losses.append(loss_domain)

        return losses

    def accuracy_source_target(self, y_pred_array, y_true_array, type_labels):
        """
        calculate the accuracy source, target given y_true and y_pred_label
        """
        # get the indices
        indices = self.get_indices(y_true_array=y_true_array, type_labels=type_labels)

        # calculate accuracy
        y_true_array = y_true_array[:, 1]
        accuracy = []
        for idx in indices:
            y_pred_domain = y_pred_array[idx]
            y_true_domain = y_true_array[idx]
            acc = accuracy_score(y_pred=y_pred_domain, y_true=y_true_domain)
            accuracy.append(acc)

        return accuracy

    def embedding_source_target(self, embedding_train_array, y_pred_array):
        """
        split embedding source and target given embedding array and y_pred_array
        """
        # get the indices of predicted source and target
        source_indices, target_indices = self.get_indices(
            y_pred_array=y_pred_array, type_labels=("source", "target")
        )

        # split the embedding
        embedding_source = embedding_train_array[source_indices]
        embedding_target = embedding_train_array[target_indices]

        return embedding_source, embedding_target

    def domain_anomaly_score_decision(
        self,
        k,
        distance,
        percentile,
        embedding_train_array,
        embedding_test_array,
        y_pred_train_array,
        y_test_array,
    ):

        # init model knn
        if distance == None:
            distance = "cosine"
        knn_source = NearestNeighbors(n_neighbors=k, metric=distance)
        knn_target = NearestNeighbors(n_neighbors=k, metric=distance)

        # get the threshold for the anomaly decision
        embedding_train_source, embedding_train_target = self.embedding_source_target(
            embedding_train_array=embedding_train_array, y_pred_array=y_pred_train_array
        )

        # fit knn models
        knn_source.fit(embedding_train_source)
        knn_target.fit(embedding_train_target)

        # get the threshold for the anomaly decision
        distance_train_source, _ = knn_source.kneighbors(embedding_train_source)
        distance_train_source = np.mean(distance_train_source, axis=1)
        distance_train_target, _ = knn_target.kneighbors(embedding_train_target)
        distance_train_target = np.mean(distance_train_target, axis=1)

        threshold_source = np.percentile(distance_train_source, percentile)
        threshold_target = np.percentile(distance_train_target, percentile)

        # find the distance to source and target
        distance_test_source, _ = knn_source.kneighbors(embedding_test_array)
        distance_test_source = np.mean(distance_test_source, axis=1)

        distance_test_target, _ = knn_target.kneighbors(embedding_test_array)
        distance_test_target = np.mean(distance_test_target, axis=1)

        distance_test_concat = np.stack((distance_test_source, distance_test_target))

        # split y_test to timeseries numerical order (windows)
        windows = y_test_array[:, 0]

        # get the each time series index from window index. timeseries =  [ts7000, ts7001,...,ts8399]; ts_to_window_dict ={time series:windows_of_this_timeseries}
        timeseries = np.unique(windows)
        ts_to_window_dict = {
            element: np.where(windows == element)[0] for element in timeseries
        }

        # get the domain, anomaly score, anomaly decision for each time series (time series in range (7000,8399)): source 0, target 1, normal 0, anomaly 1. {time series: domain}, {time series: min(max distance) to min domain}, {time series: decision}
        domain_dict = {}
        anomaly_score = {}
        anomaly_decision = {}

        for ts, ws in ts_to_window_dict.items():

            # distances of each window in a single timeseries
            distance_concat_windows_this_ts = distance_test_concat[:, ws]

            # max distance of all windows from a sigle timeseries to train source and target data
            distance_concat_windows_this_ts = distance_concat_windows_this_ts.max(
                axis=1
            )

            # get the domain of this ts and anomaly score
            domain = np.argmin(distance_concat_windows_this_ts)
            distance = np.min(distance_concat_windows_this_ts)
            domain_dict[ts] = domain

            # calculate mahanalobis distance as anomly score and anomaly decision
            if domain == 0:
                threshold = threshold_source

            elif domain == 1:
                threshold = threshold_target

            anomaly_decision[ts] = 1 if distance > threshold else 0
            anomaly_score[ts] = distance

        return domain_dict, anomaly_score, anomaly_decision

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
        speed_purturb=False,
        speed_factors=None,
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

        # init model pretrained neural network and fine tunned it
        model = Wav2VecXLR300MCustom(
            model_name=model_name,
            emb_size=emb_size,
            output_size=self.num_classes_train,
            classifier_head=classifier_head,
            window_size=window_size,
        )
        num_params = sum(p.numel() for p in model.parameters())

        # if multiple gpus
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

        # save parameter in neptune
        if window_size == None:
            window_size = self.fs * 2
        if hop_size == None:
            hop_size = self.fs
        if distance == None:
            distance = "cosine"

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
            "percentile": percentile,
            "distance": distance,
            "k": k,
            "speed_purturb": speed_purturb,
            "num_params": num_params,
        }

        if speed_purturb == True:
            hyperparameters["perturb_factors"] = stringify_unsupported(speed_factors)

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

        # if multiple gpus
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

        # init loss
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

            # init metrics
            loss_train = 0

            # y array
            y_train_array = np.empty(shape=(len_train, 2))
            y_pred_train_array = np.empty(shape=(len_train,))

            y_test_array = np.empty(shape=(len_test, 3))
            y_pred_test_array = np.empty(shape=(len_test,))

            # embedding array
            embedding_train_array = np.empty((len_train, emb_size))
            embedding_test_array = np.empty((len_test, emb_size))

            # training mode
            model.train()
            loss.train()

            # training batch loop
            for batch_train, (X_train, y_train) in enumerate(train_loader):

                print("batch_train", batch_train)

                # speed perturb
                if speed_purturb:
                    X_train = self.speed_perturb(X_train, speed_factors=speed_factors)

                # to device
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass
                embedding_train = model(X_train)
                y_pred_train_label = loss.pred_labels(embedding=embedding_train)

                # save to array
                embedding_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = (embedding_train.cpu().detach().numpy())

                y_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_train.cpu().numpy()

                y_pred_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_pred_train_label.cpu().numpy()

                # calculate the loss
                loss_train_this_batch = loss(embedding_train, y_train)
                loss_train = loss_train + loss_train_this_batch.item()

                # gradient decent, backpropagation and update parameters
                optimizer.zero_grad()
                loss_train_this_batch.backward()
                optimizer.step()

            # evaluation mode
            model.eval()
            loss.eval()
            with torch.no_grad():
                for batch_test, (X_test, y_test) in enumerate(test_loader):
                    print("batch_test", batch_test)

                    # to device
                    X_test = X_test.to(self.device)

                    # forward pass
                    embedding_test = model(X_test)
                    y_pred_test_label = loss.pred_labels(embedding=embedding_test)

                    # save to array
                    embedding_test_array[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = embedding_test.cpu().numpy()

                    y_test_array[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = y_test.cpu().numpy()

                    y_pred_test_array[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = y_pred_test_label.cpu().numpy()

            # type labels
            type_labels_train = ("train_source", "train_target")
            type_labels_test = (
                "test_source_normal",
                "test_source_anomaly",
                "test_target_normal",
                "test_target_anomaly",
            )

            # loss train, train source, train target
            loss_train = loss_train / len(train_loader)
            loss_train_source, loss_train_target = self.loss_source_target(
                embedding_array=embedding_train_array,
                y_true_array=y_train_array,
                loss=loss,
                type_labels=type_labels_train,
            )

            # loss_test_source_normal, loss_test_source_anomaly, loss_test_target_normal, loss_test_target_anomaly
            (
                loss_test_source_normal,
                loss_test_source_anomaly,
                loss_test_target_normal,
                loss_test_target_anomaly,
            ) = self.loss_source_target(
                embedding_array=embedding_test_array,
                y_true_array=y_test_array,
                type_labels=type_labels_test,
            )

            # accuracy train, train soure add train target for classification
            accuracy_train = accuracy_score(
                y_pred=y_pred_train_array, y_true=y_train_array
            )
            accuracy_train_source, accuracy_train_target = self.accuracy_source_target(
                y_pred=y_pred_train_array,
                y_true_array=y_train_array,
                type_labels=type_labels_train,
            )

            # accuracy_test_source_normal, accuracy_test_source_anomaly, accuracy_test_target_normal, accuracy_test_target_anomaly
            (
                accuracy_test_source_normal,
                accuracy_test_source_anomaly,
                accuracy_test_target_normal,
                accuracy_test_target_anomaly,
            ) = self.accuracy_source_target(
                y_pred=y_pred_test_array,
                y_true=y_test_array,
                type_labels=type_labels_test,
            )

            # domain, anomaly score and anomaly detection
            domain_dict, anomaly_score, anomaly_decision = (
                self.domain_anomaly_score_decision(
                    k=k,
                    distance=distance,
                    percentile=percentile,
                    embedding_train_array=embedding_train_array,
                    embedding_test_array=embedding_test_array,
                    y_pred_train_array=y_pred_train_array,
                    y_test_array=y_test_array,
                )
            )
