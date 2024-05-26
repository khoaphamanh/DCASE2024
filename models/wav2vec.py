from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
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

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing, raw_data_path
import utils


class Wav2VecXLR300MCustom(nn.Module):
    def __init__(self, fs, emb_size: int, output_size: int):
        super().__init__()
        self.pre_trained_wav2vec = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )

        self.fs = fs
        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=3168, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, output_size),
        )

    def forward(self, x):
        x = self.processor(x, return_tensors="pt", sampling_rate=self.fs).input_values[
            0
        ]
        x = self.pre_trained_wav2vec(x).logits
        x = self.out_layer(x)
        return x


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
        self.seed = seed
        self.train_size = 0.8

    def load_train_data(self, window_size=None, hop_size=None):
        """
        Load train data with labels are attributes given window size and hop size. Defaulf is 16000 for both
        """
        # load data
        X_train, y_train = self.data_preprocessing.load_data(
            window_size=window_size, hop_size=hop_size, train=True, test=False
        )
        # num classes, output size
        self.num_classes_train = np.unique(y_train)

        return X_train, y_train

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
        len_train = len(X_train)
        len_val = len(X_val)

        # compute the class weights
        self.class_weights = class_weight.compute_class_weight(
            class_weight="balance", classes=np.unique(y_train), y=y_train
        ).astype(float)

        # dataloader
        train_data = TensorDataset(torch.tensor(X_train, y_train))
        val_data = TensorDataset(torch.tensor(X_val, y_val))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, len_train, len_val

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
        batch_size: int,
        emb_size: int,
        lr: float,
        wd: int,
        epochs: int,
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
            "loss": "cross_entropy",
            "optimizer": "adamw",
            "window_size": window_size,
            "hop_size": hop_size,
            "weight_decay": wd,
        }
        run["hyperparameter"] = hyperparameters

        # init model
        model = Wav2VecXLR300MCustom(
            fs=self.fs, emb_size=emb_size, output_size=self.num_classes_train
        )
        model = model.to(self.device)

        # loss and optimizer
        loss = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=wd)

        # training loop:
        for ep in epochs:
            loss_train = 0
            accuracy_train = 0
            f1_train = 0

            loss_val = 0
            accuracy_val = 0
            f1_val = 0

            # confustion matrix
            if ep == epochs - 1:
                y_train_cm = torch.empty(size=(1, len_train))
                y_pred_train_cm = torch.empty(size=(1, len_train))

                y_val_cm = torch.empty(size=(1, len_val))
                y_pred_val_cm = torch.empty(size=(1, len_val))

            # training mode
            model.train()
            for batch_train, (X_train, y_train) in enumerate(train_loader):
                # to device
                X_train = X_train.to(self.device)
                y_train = y_train.to(self.device)

                # forward pass
                y_pred_train_logit = model(X_train)
                y_pred_train_label = y_pred_train_logit.argmax(dim=1)

                # calculate the loss, accuracy, f1 score and confusion matrix
                loss_train_this_batch = loss(y_pred_train_logit, y_train)
                loss_train = loss_train + loss_train_this_batch

                accuracy_train_this_batch = accuracy_score(
                    y_pred=y_pred_train_label.numpy(), y_true=y_train.numpy()
                )
                accuracy_train = accuracy_train + accuracy_train_this_batch

                f1_train_this_batch = f1_score(
                    y_pred=y_pred_train_label.numpy(),
                    y_true=y_train.numpy(),
                    average="weighted",
                )
                f1_train = f1_train + f1_train_this_batch

                if ep == epochs - 1:
                    y_train_cm[
                        batch_train * batch_size : batch_train * batch_size + batch_size
                    ] = y_train
                    y_pred_train_cm[
                        batch_train * batch_size : batch_train * batch_size + batch_size
                    ] = y_pred_train_label

                # gradient decent, backpropagation and update parameters
                optimizer.zero_grad()
                loss_train_this_batch.backward()
                optimizer.step()

                # calculate confusion metrics for last epoch

            # evaluation mode
            model.eval()
            with torch.inference_mode():
                for batch_val, (X_val, y_val) in enumerate(val_loader):
                    # to device
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)

                    # forward pass
                    y_pred_val_logit = model(X_val)
                    y_pred_val_label = y_pred_val_logit.argmax(dim=1)

                    # calculate loss, accuracy and f1 score
                    loss_val_this_batcht = loss(y_pred_val_logit, y_val)
                    loss_val = loss_val + loss_val_this_batcht

                    accuracy_val_this_batch = accuracy_score(
                        y_pred=y_pred_val_label.numpy(), y_true=y_val.numpy()
                    )
                    accuracy_val = accuracy_val + accuracy_val_this_batch

                    f1_val_this_batch = f1_score(
                        y_pred=y_pred_val_label.numpy(),
                        y_true=y_val.numpy(),
                        average="weighted",
                    )
                    f1_val = f1_val + f1_val_this_batch

                    if ep == epochs - 1:
                        y_val_cm[
                            batch_val * batch_size : batch_val * batch_size + batch_size
                        ] = y_val
                        y_pred_val_cm[
                            batch_val * batch_size : batch_val * batch_size + batch_size
                        ] = y_pred_val_label

            # print out the metrics
            loss_train = loss_train / len(train_loader)
            accuracy_train = accuracy_train / len(train_loader)
            f1_train = f1_train / len(train_loader)

            loss_val = loss_val / len(val_loader)
            accuracy_val = accuracy_val / len(val_loader)
            f1_val = f1_val / len(val_loader)

            if ep == epochs - 1:
                cm_train = confusion_matrix(
                    y_true=y_train_cm.numpy(), y_pred=y_pred_train_cm.numpy()
                )
                cm_val = confusion_matrix(
                    y_true=y_val_cm.numpy(), y_pred=y_pred_val_cm.numpy()
                )

            print("epoch {}".format(ep))
            print(
                "loss train = {:.4f}, accuracy train = {:.4f}, f1 train ={:.4f}".format(
                    loss_train, accuracy_train, f1_train
                )
            )
            if ep == epochs - 1:
                print("confusion matrix train")
                print(cm_train)
            print(
                "loss val = {:.4f}, accuracy val = {:.4f},  f1 val = {:.4f}".format(
                    loss_val, accuracy_val, f1_val
                )
            )
            if ep == epochs - 1:
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
            if ep == epochs - 1:
                run["metrics/confusion_matrix"].append(cm_train)
                run["metrics/confusion_matrix"].append(cm_val)

        # running time
        run["runing_time"] = run["sys/running_time"]

        # end log neptune
        run.stop()


if __name__ == "__main__":

    # set the seed
    seed = utils.seed
    torch.manual_seed(seed)

    # hyperparameters
    lr = utils.lr_w2v
    emb_size = utils.emb_w2v
    batch_size = utils.batch_size_w2v
    wd = utils.wd_w2v
    epochs = utils.epochs_w2v

    project = utils.project
    api_token = utils.api_token

    # data preprocessing
    data_preprocessing = DataPreprocessing(raw_data_path=raw_data_path)
    anomaly_detection = AnomalyDetection(
        data_preprocessing=data_preprocessing, seed=seed
    )

    # train model
    anomaly_detection.train_test_loop(
        project=project,
        api_token=api_token,
        batch_size=batch_size,
        emb_size=emb_size,
        lr=lr,
        wd=wd,
        epochs=epochs,
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
    # model = Wav2VecXLR300MCustom(fs=fs, emb_size=emb_size, output_size=67)
    # with torch.inference_mode():
    #     out = model(test)
    #     print("out shape:", out.shape)
    #     print("out shape:", out)
