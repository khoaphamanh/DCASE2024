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
        self.num_classes_train = len(self.unique_labels_train)

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

        # init model knn
        if distance == None:
            distance = "cosine"
        knn_source = NearestNeighbors(n_neighbors=k, metric=distance)
        knn_target = NearestNeighbors(n_neighbors=k, metric=distance)

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

            # y saved array
            y_train_all = np.empty(shape=(len_train, 2))
            y_pred_train_all = np.empty(shape=(len_train,))

            y_test_all = np.empty(shape=(len_test, 3))
            y_pred_test_all = np.empty(shape=(len_test,))

            # training mode
            model.train()
            loss.train()

            # training batch loop
            for 