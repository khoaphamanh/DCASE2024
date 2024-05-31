from transformers import Wav2Vec2ForCTC
import numpy as np
import torch
import sys
import os
from torch import nn
from torchvision.transforms.v2 import MixUp
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, TensorDataset
import neptune
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from neptune.utils import stringify_unsupported

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing, raw_data_path
import utils


# fine-tune model wav2vec
class Wav2VecXLR300MCustom(nn.Module):
    def __init__(
        self, model_name: str, emb_size: int, output_size: int, classifier_head=True
    ):
        super().__init__()
        self.pre_trained_wav2vec = Wav2Vec2ForCTC.from_pretrained(model_name)
        if classifier_head:
            self.out_layer = nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=3168, out_features=emb_size),
                nn.ReLU(),
                nn.Linear(in_features=emb_size, out_features=output_size),
            )
        else:
            self.out_layer = nn.Sequential(
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(in_features=3168, out_features=emb_size),
            )

    def forward(self, x):
        x = self.pre_trained_wav2vec(x).logits
        x = self.out_layer(x)
        return x


# Arcface Loss
class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, emb_size, margin, scale, class_weights=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.margin = margin
        self.scale = scale
        self.w = nn.Parameter(
            data=torch.randn(size=(num_classes, emb_size)), requires_grad=True
        ).to(self.device)
        self.class_weights = class_weights

    def forward(self, embedding, y_true):

        # calculate logits
        logits = self.logits(embedding=embedding, y_true=y_true)

        # combine with cross entropy loss
        ce = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = ce(logits, y_true)

        return loss

    def logits(self, embedding, y_true):

        # cos(phi) =  (x @ w.t) / (||w.t||.||x|| ) = normalize(x) @ normalize(w.t) / 1 beacause (||normalize(w.T)|| = ||normalize(x)|| )
        cosine_logits = F.linear(
            input=F.normalize(embedding), weight=F.normalize(self.w)
        )

        # onehot vector based on y_true
        onehot = self.onehot_true_label(y_true)  # size (B, num_classes)

        # cosine logit of the target class index
        cosine_target = cosine_logits[onehot == 1]  # size (B,)

        # calculate cosine phi in target class index with phi = angle + m
        cosine_phi = self.cosine_angle_plus_margin(
            cosine_target=cosine_target
        )  # size (B,)

        # calculate logit new
        diff = (cosine_phi - cosine_target).unsqueeze(1)
        logits = cosine_logits + (onehot * diff)  # size (B,num_classes)
        logits = self.scale * logits

        return logits

    def onehot_true_label(self, y_true):
        batch_size = y_true.shape[0]
        onehot = torch.zeros(batch_size, self.num_classes).to(self.device)
        onehot.scatter_(1, y_true.unsqueeze(-1), 1)
        return onehot

    def cosine_angle_plus_margin(self, cosine_target):
        eps = 1e-7
        angle = torch.acos(torch.clamp(cosine_target, -1 + eps, 1 - eps))
        phi = angle + self.margin
        cosine_phi = torch.cos(phi)
        return cosine_phi


# main class anomaly detection
class AnomalyDetection:
    def __init__(
        self,
        data_preprocessing: DataPreprocessing,
        seed,
    ):

        # data preprocessing
        self.data_preprocessing = data_preprocessing

        # time series information
        self.fs = data_preprocessing.fs

        # model configuration parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count()
        self.seed = seed
        self.train_size = 0.8
        self.num_workers = 0

    def load_train_data(self, window_size=None, hop_size=None):
        """
        Load train data with labels are attributes given window size and hop size. Defaulf is 16000 for both
        """
        # load data
        X_train, y_train = self.data_preprocessing.load_data(
            window_size=window_size, hop_size=hop_size, train=True, test=False
        )
        # num classes, output size
        self.num_classes_train = len(np.unique(y_train))

        return X_train, y_train

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
        # load train data
        X, y = self.load_train_data(window_size=window_size, hop_size=hop_size)

        # split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.train_size, random_state=self.seed, stratify=y
        )

        X_train, X_val = self.standardize(X_train=X_train, X_val=X_val)

        len_train = len(X_train)
        len_val = len(X_val)

        # compute the class weights
        self.class_weights = (
            torch.tensor(
                class_weight.compute_class_weight(
                    class_weight="balanced", classes=np.unique(y_train), y=y_train
                )
            )
            .float()
            .to(self.device)
        )

        # dataloader
        train_data = TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train).long()
        )
        val_data = TensorDataset(
            torch.tensor(X_val).float(), torch.tensor(y_val).long()
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

        return train_loader, val_loader, len_train, len_val

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

    def speed_perturb(self, min_rate, max_rate, p):
        """
        Speed perturb for stretch of compress audio signal
        """

        return None

    def mix_up(self, alpha: float, num_classes: int):
        return MixUp(alpha=alpha, num_classes=num_classes)

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
        loss_name="cross_entropy",
        optimizer_name="AdamW",
        classifier_head=True,
        scale=None,
        margin=None,
        window_size=None,
        hop_size=None,
    ):
        """
        Train test loop
        """
        # init run
        run = neptune.init_run(project=project, api_token=api_token)

        # load dataloader
        train_loader, val_loader, len_train, len_val = self.data_loader(
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
        if loss_name == "arcface":
            hyperparameters["classifier_head"] = classifier_head
            hyperparameters["scale"] = scale
            hyperparameters["margin"] = margin

        configurations = {
            "seed": self.seed,
            "device": self.device,
            "n_gpus": self.n_gpus,
            "num_workers": self.num_workers,
        }

        run["configurations"] = stringify_unsupported(configurations)
        run["hyperparameters"] = hyperparameters

        # init model
        model = Wav2VecXLR300MCustom(
            model_name=model_name,
            emb_size=emb_size,
            output_size=self.num_classes_train,
            classifier_head=classifier_head,
        )

        # if multiple gpus
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

        # loss
        if loss_name == "cross_entropy":
            loss = nn.CrossEntropyLoss(weight=self.class_weights)

        elif loss_name == "arcface":
            loss = ArcFaceLoss(
                num_classes=self.num_classes_train,
                emb_size=emb_size,
                class_weights=self.class_weights,
                scale=scale,
                margin=margin,
            )

        # optimizer
        parameters = (
            list(model.parameters()) + list(loss.parameters())
            if loss_name == "arcface"
            else model.parameters()
        )

        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(params=parameters, lr=lr, weight_decay=wd)

        elif optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(params=parameters, lr=lr, weight_decay=wd)

        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(params=parameters, lr=lr, weight_decay=wd)

        # training loop:
        for ep in range(epochs):
            loss_train = 0
            accuracy_train = 0
            f1_train = 0

            loss_val = 0
            accuracy_val = 0
            f1_val = 0

            # confustion matrix
            y_train_cm = torch.empty(size=(len_train,))
            y_pred_train_cm = torch.empty(size=(len_train,))

            y_val_cm = torch.empty(size=(len_val,))
            y_pred_val_cm = torch.empty(size=(len_val,))

            # training mode
            model.train()
            for batch_train, (X_train, y_train) in enumerate(train_loader):

                # to device
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass
                if loss_name == "arcface":
                    embedding_train = model(X_train)
                    y_pred_train_logit = loss.logits(
                        embedding=embedding_train, y_true=y_train
                    )

                elif loss_name == "cross_entropy":
                    y_pred_train_logit = model(X_train)

                y_pred_train_label = y_pred_train_logit.argmax(dim=1)

                # calculate the loss, accuracy, f1 score and confusion matrix
                if loss_name == "arcface":
                    loss_train_this_batch = loss(embedding_train, y_train)

                elif loss_name == "cross_entropy":
                    loss_train_this_batch = loss(y_pred_train_logit, y_train)
                loss_train = loss_train + loss_train_this_batch.item()

                accuracy_train_this_batch = accuracy_score(
                    y_pred=y_pred_train_label.cpu().numpy(),
                    y_true=y_train.cpu().numpy(),
                )
                accuracy_train = accuracy_train + accuracy_train_this_batch

                f1_train_this_batch = f1_score(
                    y_pred=y_pred_train_label.cpu().numpy(),
                    y_true=y_train.cpu().numpy(),
                    average="weighted",
                )
                f1_train = f1_train + f1_train_this_batch

                y_train_cm[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_train.cpu()
                y_pred_train_cm[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_pred_train_label.cpu()

                # gradient decent, backpropagation and update parameters
                optimizer.zero_grad()
                loss_train_this_batch.backward()
                optimizer.step()

            # evaluation mode
            model.eval()
            with torch.inference_mode():
                for batch_val, (X_val, y_val) in enumerate(val_loader):

                    # to device
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)

                    # forward pass
                    if loss_name == "arcface":
                        embedding_val = model(X_val)
                        y_pred_val_logit = loss.logits(
                            embedding=embedding_val, y_true=y_val
                        )
                    elif loss_name == "cross_entropy":
                        y_pred_val_logit = model(X_val)

                    y_pred_val_label = y_pred_val_logit.argmax(dim=1)

                    # calculate loss, accuracy, f1 score and confusion matrix
                    if loss_name == "arcface":
                        loss_val_this_batch = loss(embedding_val, y_val)
                    elif loss_name == "cross_entropy":
                        loss_val_this_batch = loss(y_pred_val_logit, y_val)
                    loss_val = loss_val + loss_val_this_batch.item()

                    accuracy_val_this_batch = accuracy_score(
                        y_pred=y_pred_val_label.cpu().numpy(),
                        y_true=y_val.cpu().numpy(),
                    )
                    accuracy_val = accuracy_val + accuracy_val_this_batch

                    f1_val_this_batch = f1_score(
                        y_pred=y_pred_val_label.cpu().numpy(),
                        y_true=y_val.cpu().numpy(),
                        average="weighted",
                    )
                    f1_val = f1_val + f1_val_this_batch

                    y_val_cm[
                        batch_val * batch_size : batch_val * batch_size + batch_size
                    ] = y_val.cpu()
                    y_pred_val_cm[
                        batch_val * batch_size : batch_val * batch_size + batch_size
                    ] = y_pred_val_label.cpu()

            # print out the metrics
            loss_train = loss_train / len(train_loader)
            accuracy_train = accuracy_train / len(train_loader)
            f1_train = f1_train / len(train_loader)

            loss_val = loss_val / len(val_loader)
            accuracy_val = accuracy_val / len(val_loader)
            f1_val = f1_val / len(val_loader)

            cm_train = confusion_matrix(
                y_true=y_train_cm.cpu().numpy(),
                y_pred=y_pred_train_cm.cpu().numpy(),
            )
            cm_val = confusion_matrix(
                y_true=y_val_cm.cpu().numpy(), y_pred=y_pred_val_cm.cpu().numpy()
            )

            print("epoch {}".format(ep))
            print(
                "loss train = {:.4f}, accuracy train = {:.4f}, f1 train = {:.4f}".format(
                    loss_train, accuracy_train, f1_train
                )
            )

            print("confusion matrix train")
            print(cm_train)
            print(
                "loss val = {:.4f}, accuracy val = {:.4f},  f1 val = {:.4f}".format(
                    loss_val, accuracy_val, f1_val
                )
            )
            print("confusion matrix val")
            print(cm_val)
            print()

            # log the metrics in neptune
            metrics = {
                "loss_train": loss_train,
                "accuracy_train": accuracy_train,
                "f1_train": f1_train,
                "loss_val": loss_val,
                "accuracy_val": accuracy_val,
                "f1_val": f1_val,
            }

            run["metrics"].append(metrics, step=ep)

            cm_train_fig = self.plot_confusion_matrix(cm=cm_train, name="train")
            cm_val_fig = self.plot_confusion_matrix(cm=cm_val, name="val")
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
    emb_size = utils.emb_np
    batch_size = utils.batch_size_np
    wd = utils.wd_np
    epochs = utils.epochs_np
    optimizer_name = utils.optimizer_name_np
    model_name = utils.model_name_np
    scale = utils.scale_np
    margin = utils.margin_np
    loss_name = utils.loss_name_np
    classifier_head = utils.classifier_head_np

    project = utils.project
    api_token = utils.api_token

    # data preprocessing
    data_preprocessing = DataPreprocessing(raw_data_path=raw_data_path)
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
