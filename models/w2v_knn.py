from transformers import Wav2Vec2ForCTC
import numpy as np
import torch
import torch.nn.functional as F
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
        data_preprocessing: DataPreprocessing,
        seed,
    ):

        # data preprocessing
        self.data_preprocessing = data_preprocessing
        self.data_name = data_preprocessing.data_name
        self.unique_labels_dict = data_preprocessing.load_unique_labels_dict()
        self.unique_labels_train = self.unique_labels_dict["train"]
        self.unique_labels_test = self.unique_labels_dict["test"]
        self.num_classes_train = len(self.unique_labels_train)
        self.labels_analysis_dict = data_preprocessing.labels_analysis_dict()

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

    def get_source_target_indices(self, y_true, test_classification=False):
        """
        get the source and target indices given y_true, elim for the case that label without anomaly normal.
        """
        # check test classification
        if test_classification == False:
            source = "train_source"
            target = "train_target"

            # get the indices of train source and train target
            domain = [source, target]

        # this case use for check test loss/accuracy in classification task
        elif test_classification == True:
            y_true = y_true[:, 1]
            source_normal = "test_source_normal"
            source_anomaly = "test_source_anomaly"
            target_normal = "test_target_normal"
            target_anomaly = "test_target_anomaly"

            # get the indices of test source target normal anomaly
            domain = [source_normal, source_anomaly, target_normal, target_anomaly]

        indices = []
        for d in domain:
            domain_labels = self.labels_analysis_dict[d]
            domain_indices = np.where(np.isin(y_true, domain_labels))[0]
            indices.append(domain_indices)

        # return source indices, target indices for test classification False
        # return source_normal_indices, source_anomaly_indices, target_normal_indices, target_anomaly_indices for test classification True
        return indices

    def accuracy_source_target(self, y_pred, y_true, test_classification=False):
        """
        calculate the accuracy source and target in train data given y_true and y_pred_label
        """

        # indices: source_indices, target_indices if classification False
        # indices:source_normal_indices, source_anomaly_indices, target_normal_indices, target_anomaly_indices for test classification True
        indices = self.get_source_target_indices(
            y_true=y_true,
            test_classification=test_classification,
        )

        # this case
        if test_classification == True:
            y_true = y_true[:, 2]

        # get the accuracy train source and target
        accuracy = []
        for idx in indices:
            y_pred_domain = y_pred[idx]
            y_true_domain = y_true[idx]
            acc = accuracy_score(y_pred=y_pred_domain, y_true=y_true_domain)
            accuracy.append(acc)

        # return accuracy_train_source, accuracy_train_target if test classification False
        # return accuracy_test_source_normal, accuracy_test_source_anomaly, accuracy_test_target_normal, accuracy_test_target_anomaly if test classification True
        return accuracy

    def loss_source_target(
        self,
        embedding_array,
        y_true,
        loss,
        test_classification=False,
    ):
        """
        calculate the loss source and target
        """
        # indices: source_indices, target_indices if classification False
        # indices:source_normal_indices, source_anomaly_indices, target_normal_indices, target_anomaly_indices for test classification True
        indices = self.get_source_target_indices(
            y_true=y_true, test_classification=test_classification
        )

        if test_classification == True:
            y_true = y_true[:, 2]

        losses = []
        for idx in indices:
            embedding_domain = embedding_array[idx]
            y_true_domain = y_true[idx]
            loss_domain = loss.calculate_loss(embedding_domain, y_true_domain)
            losses.append(loss_domain)

        # return loss_train_source, loss_train_target if test classification False
        # return loss_test_source_normal, loss_test_source_anomaly, loss_test_target_normal, loss_test_target_anomaly if test classification True
        return losses

    def embedding_source_target(
        self, embedding_array, y_true, test_classification=False
    ):
        """
        split embedding source and target given embedding array and y_true array
        """
        # get the indices of the source and target in y_true
        source_indices, target_indices = self.get_source_target_indices(
            y_true=y_true, test_classification=test_classification
        )

        # split the embedding
        embedding_source = embedding_array[source_indices]
        embedding_target = embedding_array[target_indices]

        return embedding_source, embedding_target

    def domain_anomaly_score_decision(
        self,
        knn_source,
        knn_target,
        embedding_train_array,
        y_train_true,
        embedding_test_array,
        y_test_true,
        percentile,
    ):
        """
        find the domain of a given embedding test array
        """
        # split the embedding
        embedding_train_source, embedding_train_target = self.embedding_source_target(
            embedding_array=embedding_train_array, y_true=y_train_true
        )

        # fit knn models
        knn_source.fit(embedding_train_source)
        knn_target.fit(embedding_train_target)

        # get the threshold for the anomaly decision
        distance_train_source = knn_source.kneighbors(embedding_train_source)
        distance_train_target = knn_target.kneighbors(embedding_train_target)

        threshold_source = np.percentile(distance_train_source, percentile)
        threshold_target = np.percentile(distance_train_target, percentile)

        # split y_test to index of original time series and label
        windows = y_test_true[:, 0]

        # find the cosine distance to source and target
        distance_test_source, _ = knn_source.kneighbors(embedding_test_array)
        distance_test_source = np.mean(distance_test_source, axis=1)

        distance_test_target, _ = knn_target.kneighbors(embedding_test_array)
        distance_test_target = np.mean(distance_test_target, axis=1)

        distance_test_concat = np.stack((distance_test_source, distance_test_target))

        # get the each time series index from window index. timeseries: [ts1, ts2,...,ts6399] {time series:windows_of_this_timeseries}
        timeseries = np.unique(windows)
        ts_to_window_dict = {
            element: np.where(windows == element)[0] for element in timeseries
        }

        # get the domain, anomaly score, anomaly decision for each time series (time series in range (0,1399)): source 0, target 1, normal 0, anomaly 1. {time series: domain}, {time series: mahalobis}, {time series: decision}
        domain_dict = {}
        anomaly_score = {}
        anomaly_decision = {}

        for ts, ws in ts_to_window_dict.items():

            # distances of each window in a single timeseries
            distance_concat_windows_this_ts = distance_test_concat[:, ws]

            # mean distance of all windows from a sigle timeseries
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

    def compact_y_test(self, y_test_cm, type: None):
        """
        compact y_test_cm to be shorter based on train source, train target, test anomaly, test normal
        y_true = [label of ts1, label of ts2,...]
        """
        # get the labels based on given type
        labels_type = self.labels_analysis_dict[type]

        # compact y_true_cm y_true = {ts1:label_ts1, ts2:label_ts2} -> y_true = [ts 1: label of ts 1]
        y_true = {k: v for k, v, _ in y_test_cm}
        y_true = dict(sorted(y_true.items()))
        y_true = {k: 0 if v in labels_type else 1 for k, v in y_true.items()}
        y_true = np.array(list(y_true.values()))

        return y_true

    def accuracy_domain_test(self, domain_dict, y_test_cm):
        """
        calculate accuracy domain prediction
        """
        # domain prediction of each ts
        domain_pred = np.array(list(domain_dict.values()))

        # get y_true
        y_true = self.compact_y_test(y_test_cm=y_test_cm, type="test_source")

        # calculate accuracy domain
        accuracy_domain = accuracy_score(y_pred=domain_pred, y_true=y_true)

        return accuracy_domain

    def accuracy_decision_test(self, anomaly_decision, y_test_cm):
        """
        calculate accuracy decision prediction
        """
        # decision prediction of each ts
        y_pred = np.array(list(anomaly_decision.values()))

        # y_true
        y_true = self.compact_y_test(y_test_cm=y_test_cm, type="test_normal")

        # calculate accuracy decision
        accuracy_decision = accuracy_score(y_pred=y_pred, y_true=y_true)

        return accuracy_decision

    def aucroc_anomaly_score(self, anomaly_score, y_test_cm):
        """
        calculate auroc given anomaly scire and y_test
        """
        # y score
        # print("anomaly_score", anomaly_score)
        y_score = np.array(list(anomaly_score.values()))
        # print("y_score:", y_score)

        # y_true
        y_true = self.compact_y_test(y_test_cm=y_test_cm, type="test_normal")

        # roc auc
        fpr, tpr, thresholds = roc_curve(y_score=y_score, y_true=y_true)
        roc_auc = auc(fpr, tpr)

        # plot
        fig = self.plot_auroc(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

        return roc_auc, fig

    def plot_auroc(self, fpr, tpr, roc_auc):
        """
        plot the auroc curve
        """
        # Plot ROC curve
        fig = plt.figure(figsize=(10, 10))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")

        return fig

    def confusion_matrix_test(
        self,
        y_true_test,
        anomaly_decision=None,
        domain_dict=None,
        y_pred_test_labels=None,
        test_classification=False,
    ):
        """
        confusion matrix given anomaly_decision and y_test_cm
        """
        # check if test classification
        if test_classification == True:

            # get the indices of the source and target in y_true
            indices = self.get_source_target_indices(
                y_true=y_true_test,
                test_classification=test_classification,
            )
            y_true_test = y_true_test[:, 2]

            source_target_indices = np.concatenate(indices)
            y_pred = y_pred_test_labels[source_target_indices]
            y_true = y_true_test[source_target_indices]

            return confusion_matrix(y_pred=y_pred, y_true=y_true)

        elif test_classification == False:
            # y_pred based on anomaly score
            y_pred = np.array(list(anomaly_decision.values()))

            # y_true
            y_true = self.compact_y_test(y_test_cm=y_true_test, type="test_normal")

            return confusion_matrix(y_pred=y_pred, y_true=y_true)

        elif test_classification == "domain":
            # domain prediction of each ts
            domain_pred = np.array(list(domain_dict.values()))

            # get y_true
            y_true = self.compact_y_test(y_test_cm=y_true_test, type="test_source")

            return confusion_matrix(y_pred=domain_pred, y_true=y_true)

    def speed_perturb(self, X, speed_factors):
        """
        use augmentation as purturb the speed of the input given the speed factors and then down/upsampling the timeseries using cutting, padding
        """
        # load the speed purturb as the augmentation
        augmentation = SpeedPerturbation(self.fs, speed_factors)

        # use the augmentation for inputs
        len_input = X.shape[-1]
        X = augmentation(X)[0]
        len_output = X.shape[-1]

        # downsampling
        if len_output > len_input:
            start_idx = torch.randint(0, len_output - len_input)
            X = X[start_idx : start_idx + len_input]

        # upsampling
        else:
            X = F.pad(X, (0, len_input - len_output), mode="constant", value=0)

        return X

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
        perturb_factors=None,
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
        }

        if speed_purturb == True:
            hyperparameters["perturb_factors"] = perturb_factors

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

            # confustion matrix
            y_train_cm = np.empty(shape=(len_train,))
            y_pred_train_cm = np.empty(shape=(len_train,))

            y_test_cm = np.empty(shape=(len_test, 3))
            y_pred_test_cm = np.empty(shape=(len_test,))

            # embedding array init
            embedding_train_array = np.empty((len_train, emb_size))
            embedding_test_array = np.empty((len_test, emb_size))

            # training mode
            model.train()
            loss.train()
            for batch_train, (X_train, y_train) in enumerate(train_loader):

                print("batch_train", batch_train)

                # to device
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass
                embedding_train = model(X_train)
                embedding_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = (embedding_train.cpu().detach().numpy())

                y_pred_train_label = loss.pred_labels(embedding=embedding_train)

                # calculate the loss
                loss_train_this_batch = loss(embedding_train, y_train)
                loss_train = loss_train + loss_train_this_batch.item()

                # indexing the y_train true, pred labels
                y_train_cm[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_train.cpu().numpy()
                y_pred_train_cm[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_pred_train_label.cpu().numpy()
                # y_pred_train_logits_cm[
                #     batch_train * batch_size : batch_train * batch_size + batch_size
                # ] = y_pred_train_logits.cpu().numpy()

                # gradient decent, backpropagation and update parameters
                optimizer.zero_grad()
                loss_train_this_batch.backward()
                optimizer.step()

                # if batch_train == 200:
                #     break

            # evaluation mode
            model.eval()
            loss.eval()
            with torch.inference_mode():
                for batch_test, (X_test, y_test) in enumerate(test_loader):
                    print("batch_test", batch_test)
                    # split y_test to y_test_na: labels with normal/anomaly and y_test_att: labels of attribute without anomaly, normal

                    # to device
                    X_test = X_test.to(self.device)
                    # y_test_att = y_test_att.to(self.device)

                    # forward pass
                    embedding_test = model(X_test)
                    embedding_test_array[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = embedding_test.cpu().numpy()
                    y_pred_test_label = loss.pred_labels(embedding=embedding_test)

                    # indexing the y_test true, pred labels
                    y_test_cm[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = y_test.cpu().numpy()
                    y_pred_test_cm[
                        batch_test * batch_size : batch_test * batch_size + batch_size
                    ] = y_pred_test_label.cpu().numpy()
                    # if batch_test == 200:
                    #     break

            print("loss train")
            # loss train, loss train source, loss train target classification
            loss_train_source, loss_train_target = self.loss_source_target(
                embedding_array=embedding_train_array,
                y_true=y_train_cm,
                loss=loss,
                test_classification=False,
            )

            loss_train = loss_train / len(train_loader)

            print("loss test")
            # loss_test_source_normal, loss_test_source_anomaly, loss_test_target_normal, loss_test_target_anomaly
            (
                loss_test_source_normal,
                loss_test_source_anomaly,
                loss_test_target_normal,
                loss_test_target_anomaly,
            ) = self.loss_source_target(
                embedding_array=embedding_test_array,
                y_true=y_test_cm,
                loss=loss,
                test_classification=True,
            )

            # accuracy train, train soure adnd train target for classification
            accuracy_train_source, accuracy_train_target = self.accuracy_source_target(
                y_pred=y_pred_train_cm, y_true=y_train_cm, test_classification=False
            )

            accuracy_train = accuracy_score(y_pred=y_pred_train_cm, y_true=y_train_cm)

            # accuracy_test_source_normal, accuracy_test_source_anomaly, accuracy_test_target_normal, accuracy_test_target_anomaly
            (
                accuracy_test_source_normal,
                accuracy_test_source_anomaly,
                accuracy_test_target_normal,
                accuracy_test_target_anomaly,
            ) = self.accuracy_source_target(
                y_pred=y_pred_test_cm,
                y_true=y_test_cm,
                test_classification=True,
            )

            # domain, anomaly score and anomaly decision
            domain_dict, anomaly_score, anomaly_decision = (
                self.domain_anomaly_score_decision(
                    knn_source=knn_source,
                    knn_target=knn_target,
                    embedding_train_array=embedding_train_array,
                    embedding_test_array=embedding_test_array,
                    y_train_true=y_train_cm,
                    y_test_true=y_test_cm,
                    percentile=percentile,
                )
            )

            # accuracy domain test
            accuracy_domain = self.accuracy_domain_test(
                domain_dict=domain_dict, y_test_cm=y_test_cm
            )

            # accuracy anomaly decision
            accuracy_decision = self.accuracy_decision_test(
                anomaly_decision=anomaly_decision, y_test_cm=y_test_cm
            )

            # roc_auc
            roc_auc, roc_auc_fig = self.aucroc_anomaly_score(
                anomaly_score=anomaly_score, y_test_cm=y_test_cm
            )

            # confusion matrix train source target
            cm_train_source_target = confusion_matrix(
                y_true=y_train_cm, y_pred=y_pred_train_cm
            )
            cm_train_source_target_fig = self.plot_confusion_matrix(
                cm=cm_train_source_target, name="train"
            )

            # confusion matrix test source target
            cm_test_source_target = self.confusion_matrix_test(
                y_true_test=y_test_cm,
                y_pred_test_labels=y_pred_test_cm,
                test_classification=True,
            )
            cm_test_source_target_fig = self.plot_confusion_matrix(
                cm=cm_test_source_target, name="test source target"
            )

            # confusion matrix test normal anomaly
            cm_test_normal_anomaly = self.confusion_matrix_test(
                anomaly_decision=anomaly_decision,
                y_true_test=y_test_cm,
                test_classification=False,
            )
            cm_test_normal_anomaly_fig = self.plot_confusion_matrix(
                cm=cm_test_normal_anomaly, name="test normal anomaly"
            )

            # confusion matrix test domain
            cm_test_domain = self.confusion_matrix_test(
                y_true_test=y_test_cm,
                domain_dict=domain_dict,
                test_classification="domain",
            )
            cm_test_domain_fig = self.plot_confusion_matrix(
                cm=cm_test_domain, name="test domain"
            )

            # f1 score
            f1_train = f1_score(
                y_pred=y_pred_train_cm,
                y_true=y_train_cm,
                average="weighted",
            )

            print("epoch {}".format(ep))

            print(
                "loss train = {:.4f}, accuracy train = {:.4f}, f1 train = {:.4f}".format(
                    loss_train, accuracy_train, f1_train
                )
            )

            print(
                "loss train source = {:.4f}, loss train target = {:.4f}".format(
                    loss_train_source, loss_train_target
                )
            )
            print(
                "accuracy train source = {:.4f}, accuracy train target = {:.4f}".format(
                    accuracy_train_source, accuracy_train_target
                )
            )

            print(
                "loss test source normal = {:.4f}, loss test source anomaly = {:.4f}, loss test target normal = {:.4f},loss test target anomaly = {:.4f},".format(
                    loss_test_source_normal,
                    loss_test_source_anomaly,
                    loss_test_target_normal,
                    loss_test_target_anomaly,
                )
            )

            print(
                "accuracy test source normal = {:.4f}, accuracy test source anomaly = {:.4f}, accuracy test target normal = {:.4f}, accuracy test target anomaly = {:.4f},".format(
                    accuracy_test_source_normal,
                    accuracy_test_source_anomaly,
                    accuracy_test_target_normal,
                    accuracy_test_target_anomaly,
                )
            )

            print(
                "accuracy domain = {:.4f},  accuracy decision = {:.4f}, roc_auc = {:.4f}".format(
                    accuracy_domain, accuracy_decision, roc_auc
                )
            )

            # log the metrics in neptune
            metrics = {
                "loss_train": loss_train,
                "loss_train_source": loss_train_source,
                "loss_train_target": loss_train_target,
                "loss_test_source_normal": loss_test_source_normal,
                "loss_test_source_anomaly": loss_test_source_anomaly,
                "loss_test_target_normal": loss_test_target_normal,
                "loss_test_target_anomaly": loss_test_target_anomaly,
                "accuracy_train": accuracy_train,
                "accuracy_train_source": accuracy_train_source,
                "accuracy_train_target": accuracy_train_target,
                "accuracy_test_source_normal": accuracy_test_source_normal,
                "accuracy_test_source_anomaly": accuracy_test_source_anomaly,
                "accuracy_test_target_normal": accuracy_test_target_normal,
                "accuracy_test_target_anomaly": accuracy_test_target_anomaly,
                "f1_train": f1_train,
                "accuracy_domain_test": accuracy_domain,
                "accuracy_decision_test": accuracy_decision,
                "roc_auc_test": roc_auc,
            }

            run["metrics"].append(metrics, step=ep)

            # log the images of confusion matrix
            run["metrics/confusion_matrix_train"].append(
                cm_train_source_target_fig, step=ep
            )
            plt.close()

            run["metrics/confusion_matrix_test_normal_anomaly"].append(
                cm_test_normal_anomaly_fig, step=ep
            )
            plt.close()

            run["metrics/confusion_matrix_test_source_target"].append(
                cm_test_source_target_fig, step=ep
            )
            plt.close()

            run["metrics/confusion_matrix_test_domain"].append(
                cm_test_domain_fig, step=ep
            )
            plt.close()

            run["metrics/roc_auc_plot"].append(roc_auc_fig, step=ep)
            plt.close()

            # break

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
    lr = utils.lr_dev
    emb_size = utils.emb_size_dev
    batch_size = utils.batch_size_dev
    wd = utils.wd_dev
    epochs = utils.epochs_dev
    optimizer_name = utils.optimizer_name_dev
    model_name = utils.model_name_dev
    scale = utils.scale_dev
    margin = utils.margin_dev
    loss_name = utils.loss_name_dev
    classifier_head = utils.classifier_head_dev
    window_size = utils.window_size_dev
    hop_size = utils.hop_size_dev
    k = utils.k_dev
    percentile = utils.percentile_dev
    distance = utils.distance_dev

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
        k=k,
        percentile=percentile,
        scale=scale,
        margin=margin,
        classifier_head=classifier_head,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        window_size=window_size,
        hop_size=hop_size,
        distance=distance,
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
