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
from scipy.stats import hmean
from torchvision.transforms import v2

# import random

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing_new_new import DataPreprocessing
import utils
from metrics import AdaCosLoss, ArcFaceLoss


# fine-tune model wav2vec
class Wav2Vec2Custom(nn.Module):
    def __init__(
        self,
        model_name: str,
        emb_size: int,
        output_size: int,
        classifier_head=True,
        window_size=None,
        dropout=0.0,
    ):
        super().__init__()
        if window_size == None:
            window_size = 32000

        self.classifier_head = classifier_head
        flatten_dim = int((((window_size / 16000) * 50) - 1)) * 32

        self.pre_trained_wav2vec = Wav2Vec2ForCTC.from_pretrained(model_name)
        if classifier_head:
            self.out_layer = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
                nn.ReLU(),
                nn.Linear(in_features=flatten_dim, out_features=emb_size),
                nn.ReLU(),
                nn.Linear(in_features=emb_size, out_features=output_size),
            )
        else:
            self.out_layer = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=dropout),
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

        # data analysis
        self.unique_labels_machine_domain = (
            self.data_preprocessing.unique_labels_machine_domain()
        )
        self.full_labels_ts = self.data_preprocessing.full_labels_ts()
        self.ts_analysis = self.data_preprocessing.ts_analysis()
        self.label_analysis = self.data_preprocessing.label_analysis
        self.auc_roc_name = self.data_preprocessing.auc_roc_name
        self.knn_name = self.data_preprocessing.knn_name

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

        X_train, X_test = self.standardize(X_train=X_train, X_test=X_test)

        len_train = len(X_train)
        len_test = len(X_test)

        # compute the class weights
        self.class_weights = torch.tensor(
            class_weight.compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train[:, 1]),
                y=y_train[:, 1],
            )
        ).float()
        # print("class_weights:", self.class_weights)

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
        test_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers
        )

        return train_loader, test_loader, len_train, len_test

    def speed_perturb(self, X, speed_factors):
        """
        use augmentation as purturb the speed of the input given the speed factors and then down/upsampling the timeseries using cutting, padding
        """
        # if apply speed perturb based on random choice
        choice = np.random.choice([True, False])
        # print("choice speed purturb:", choice)

        if choice == True:
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

        else:
            return X

    def mixup(self, X, y, alpha):
        """
        use augmentation for
        """
        # if apply speed perturb based on random choice
        choice = np.random.choice([True, False])
        # print("choice mixup:", choice)

        if choice == True:
            # load the speed purturb as the augmentation
            mixup = v2.MixUp(alpha=alpha, num_classes=self.num_classes_train)

            # reshape from 2D to 4D tensor, mixup can only used to image
            batch_size = X.shape[0]
            y_mixup = y[:, 1]
            X_mixup = X.reshape(batch_size, 1, 1, -1)

            # only take the mixup sample if lamda > 0.7
            max_value_label_mixup_current = 0
            while max_value_label_mixup_current < 0.7:

                X_train_mixup, y_train_mixup = mixup(X_mixup, y_mixup)
                y_train_mixup_flatten = y_train_mixup.flatten()
                y_train_mixup_flatten = y_train_mixup_flatten[
                    y_train_mixup_flatten != 1
                ]

                max_value_label_mixup = torch.max(y_train_mixup_flatten).item()
                max_value_label_mixup_current = max_value_label_mixup
                y_train_mixup = y_train_mixup.argmax(dim=1)

            X_train_mixup = X_train_mixup.reshape(batch_size, -1)
            y[:, 1] = y_train_mixup
            return X_train_mixup, y

        else:
            return X, y

    def get_indices(
        self,
        y_pred_array=None,
        y_true_array=None,
        type_labels=["train_source", "train_target"],
    ):
        """
        get indices of source, target, normal, anomly from y_pred_array or y_train_array
        """

        # y_array if not None:
        if y_true_array is not None:
            y_array = y_true_array[:, 0]
        elif y_pred_array is not None:
            y_array = y_pred_array

        # get the id of ts correspond to the type labels
        ts_ids = [self.ts_analysis[type] for type in type_labels]

        # get the index of id ts from y_array
        indices = []
        for id in ts_ids:
            idx = np.where(np.isin(y_array, id))[0]
            indices.append(idx)

        return indices

    def accuracy_source_target(self, y_pred_array, y_true_array, type_labels):
        """
        calculate the accuracy source, target given y_true and y_pred_label
        """
        # get the indices:
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

    def anomaly_score_decision(
        self,
        k,
        distance,
        percentile,
        embedding_train_array,
        embedding_test_array,
        y_pred_train_array,
        y_pred_test_array,
        y_test_array,
        loss,
    ):
        """
        return the anomaly score and anomaly decision
        """

        # check if all predicted unique machine available
        unique_pred_train, count_pred_train = np.unique(
            y_pred_train_array, return_counts=True
        )
        if len(unique_pred_train) == len(self.unique_labels_machine_domain) and np.all(
            count_pred_train >= k
        ):

            print("this case")

            # fit the knn to pred train data
            knn = []
            threshold_decision = []
            unique_labels_machine_domain_inverse = {
                l: n for n, l in self.unique_labels_machine_domain.items()
            }

            for test_machine_domain in self.knn_name:

                # get pred train data to fit knn
                print("test_machine_domain", test_machine_domain)
                _, machine, domain = test_machine_domain.split("_")
                label_train_string = "{}_{}".format(machine, domain)
                label_train_number = unique_labels_machine_domain_inverse[
                    label_train_string
                ]
                print("label_train_number:", label_train_number)
                indices_train = np.where(y_pred_train_array == label_train_number)[0]
                print("indices_train shape:", indices_train.shape)
                print("indices_train:", indices_train)

                # fit the knn
                embedding_train_data_to_fit = embedding_train_array[indices_train]
                print(
                    "embedding_train_data_to_fit shape:",
                    embedding_train_data_to_fit.shape,
                )
                knn_train = NearestNeighbors(n_neighbors=k, metric=distance)
                knn_train.fit(embedding_train_data_to_fit)

                # find the threshold
                distance_train, _ = knn_train.kneighbors(embedding_train_data_to_fit)
                distance_train = np.mean(distance_train, axis=1)
                threshold = np.percentile(distance_train, percentile)

                # save the knn and threshold
                knn.append(knn_train)
                threshold_decision.append(threshold)

            # # loop through all auc_roc for each machine and domain:
            anomaly_score = {}
            anomaly_decision = {}

            # unique test id
            unique_test_id = np.unique(y_test_array[:, 0]).astype(int)
            print("unique_test_id:", unique_test_id)
            for id in unique_test_id:
                print()
                print("id", id)
                # get the indices of test
                indices_test = np.where(y_test_array[:, 0] == id)[0]
                print("indices_test:", indices_test)

                # get the predicted label of the whole id timeserie
                y_pred = y_pred_test_array[indices_test]
                print("y_pred:", y_pred)
                embedding_test_data_to_evaluate = embedding_test_array[indices_test]

                unique_y_pred, counts_label = np.unique(y_pred, return_counts=True)
                print("unique_y_pred:", unique_y_pred)
                print("counts_label:", counts_label)
                max_count = np.max(counts_label)
                print("max_count:", max_count)
                vote_idx = np.where(counts_label == max_count)[0]
                print("vote_idx:", vote_idx)

                # find the most vote of label for each time series
                if len(vote_idx) == 1:
                    print("len(vote_idx) == 1")
                    argmax_vote = np.argmax(counts_label)
                    pred_label = unique_y_pred[argmax_vote]
                else:
                    print("len(vote_idx) != 1")
                    same_vote_label = unique_y_pred[vote_idx]
                    print("same_vote_label:", same_vote_label)
                    same_vote_indices = np.where(np.isin(y_pred, same_vote_label))[0]
                    print("same_vote_indices:", same_vote_indices)
                    embedding_same_vote = embedding_test_data_to_evaluate[
                        same_vote_indices
                    ]
                    softmax_value = loss.return_softmax_value(embedding_same_vote)
                    max_softmax_index = torch.argmax(softmax_value)
                    max_softmax_index = np.unravel_index(
                        max_softmax_index.item(), softmax_value.shape
                    )
                    argmax_vote = same_vote_indices[max_softmax_index[0]]
                    print("argmax_vote:", argmax_vote)
                    pred_label = y_pred[same_vote_indices[max_softmax_index[0]]]

                # get the pred_label and the embedding test with the prediction same as pred label
                pred_label = int(pred_label)
                print("pred_label:", pred_label)
                indices_pred_label_from_y_pred = np.where(y_pred == pred_label)[0]
                print("indices_pred_label_from_y_pred:", indices_pred_label_from_y_pred)
                embedding_test_data_to_evaluate = embedding_test_data_to_evaluate[
                    indices_pred_label_from_y_pred
                ]

                # evaluate it to knn pretrained models
                knn_vote = knn[pred_label]
                threshold_vote = threshold_decision[pred_label]

                # find the distance test
                distance_test, _ = knn_vote.kneighbors(embedding_test_data_to_evaluate)
                distance_test = distance_test.mean(axis=1)
                distance_test = np.max(distance_test)

                anomaly_decision[id] = 1 if distance_test > threshold_vote else 0
                print("anomaly_decision:", anomaly_decision)
                anomaly_score[id] = distance_test

            print("anomaly_score", anomaly_score)
            print("anomaly_decision", anomaly_decision)

            return anomaly_score, anomaly_decision
        else:
            return None, None

    def y_true_decision_each_id(self, y_test_array):
        """
        get the true decision of each time series. This function should return a dictionary {7000: label of ts 7000, 7001:label of ts 7001,...,}
        """
        # get the label normal and anomaly of y_test
        y_true_decision = {id: label for id, _, label in y_test_array}
        y_true_decision = dict(sorted(y_true_decision.items()))

        return y_true_decision

    def accuracy_decision(self, anomaly_decision, y_test_array):
        """
        calculate accuracy decision between normal and anomaly given anomaly decision and y_test_array
        """
        # get only the value of anomaly decision
        anomaly_decision = np.array(list(anomaly_decision.values()))

        # get the label normal and anomaly of y_test
        y_true_decision = self.y_true_decision_each_id(y_test_array=y_test_array)
        y_true_decision = list(y_true_decision.values())

        # calculate accuracy decision
        accuracy_decision = accuracy_score(
            y_pred=anomaly_decision, y_true=y_true_decision
        )

        return accuracy_decision

    def auc_pauc(self, anomaly_score, anomaly_decision, y_test_array, ep):
        """
        calculate the auc and pauc decision for each machine in each domain
        """
        # loop through all of test machine domain names
        y_true_decision = self.y_true_decision_each_id(y_test_array=y_test_array)
        auc_dict = {}
        pauc_dict = {}
        accuracy_decision_dict = {}

        # create suplots
        n_axes = 21
        n_cols = 7
        n_rows = 3
        auc_roc_name = np.array(self.auc_roc_name).reshape(-1, n_cols)
        axes_place = np.arange(0, n_axes).reshape(-1, n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 15))
        axes = axes.flatten()

        for i in range(n_cols):

            # plot auc and accuracy for machine source and target
            for j in range(n_rows):

                # get test machine name
                test_machine_domain = auc_roc_name[j][i]
                if j in [0, 1]:
                    _, machine, domain = test_machine_domain.split("_")
                else:
                    # print("this case", j)
                    # print("test_machine_domain:", test_machine_domain)
                    domain, machine = test_machine_domain.split("_")

                # get ids of each test machine domain
                ids = self.ts_analysis[test_machine_domain]
                y_score = [anomaly_score[id] for id in ids]
                y_true = [y_true_decision[id] for id in ids]
                y_pred = [anomaly_decision[id] for id in ids]

                # calculate auc and accuracy
                accuraccy_machine_domain = accuracy_score(y_pred=y_pred, y_true=y_true)
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                auc_roc = auc(fpr, tpr)

                # plot the auc
                ax = axes_place[:, i]
                axes[ax[j]].plot(fpr, tpr, label=f"AUC = {auc_roc:.2f}")

                # calculate pauc
                if j == 2:
                    fpr_min = 0.0
                    fpr_max = 0.1
                    indices = np.where((fpr >= fpr_min) & (fpr <= fpr_max))
                    p_auc_roc = auc(fpr[indices], tpr[indices])

                    # plot pauc
                    axes[ax[j]].fill_between(
                        fpr,
                        tpr,
                        where=(fpr >= fpr_min) & (fpr <= fpr_max),
                        color="orange",
                        alpha=0.3,
                        label=f"PAUC = {p_auc_roc:.2f}",
                    )

                # get title, legende and axis
                axes[ax[j]].set_title(
                    "{} {} acc {:.2f}".format(machine, domain, accuraccy_machine_domain)
                )
                axes[ax[j]].legend(loc="lower right")
                axes[ax[j]].set_xlabel("FPR")
                axes[ax[j]].set_ylabel("TPR")

                # save it to dict
                accuracy_decision_dict["accuracy_{}".format(test_machine_domain)] = (
                    accuraccy_machine_domain
                )
                if j in [0, 1]:
                    auc_dict[test_machine_domain] = auc_roc
                else:
                    pauc_dict[test_machine_domain] = p_auc_roc

        # calculate hmean total
        print("auc_dict:", auc_dict)
        print("pauc_dict:", pauc_dict)
        print("accuracy_decision_dict:", accuracy_decision_dict)

        hmean_total = (
            hmean(list(auc_dict.values()) + list(pauc_dict.values()), axis=None) * 100
        )
        fig.suptitle(
            "AUC and PAUC in epoch {} with hmean {:.2f}".format(ep, hmean_total)
        )

        return fig, accuracy_decision_dict, hmean_total

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
        mixup=False,
        alpha_mixup=None,
        loss_name="adacos",
        optimizer_name="AdamW",
        classifier_head=True,
        dropout=0.0,
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
        model = Wav2Vec2Custom(
            model_name=model_name,
            emb_size=emb_size,
            output_size=self.num_classes_train,
            classifier_head=classifier_head,
            window_size=window_size,
            dropout=dropout,
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
            "mixup": mixup,
            "num_params": num_params,
            "epochs": epochs,
            "dropout": dropout,
        }

        if speed_purturb == True:
            hyperparameters["perturb_factors"] = stringify_unsupported(speed_factors)

        if mixup == True:
            hyperparameters["alpha_mixup"] = alpha_mixup

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

        # if multiple gpus
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

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
            y_train_array = np.empty(shape=(len_train, 3))
            y_pred_train_array = np.empty(shape=(len_train,))

            y_test_array = np.empty(shape=(len_test, 3))
            y_pred_test_array = np.empty(shape=(len_test,))

            # embedding array
            embedding_train_array = np.empty((len_train, emb_size))
            embedding_test_array = np.empty((len_test, emb_size))

            # training batch loop
            for batch_train, (X_train, y_train) in enumerate(train_loader):

                print("batch_train", batch_train)
                # save original X_train and y_train, y_pred to array
                with torch.no_grad():
                    model.eval()
                    X_train = X_train.to(self.device)
                    embedding_train = model(X_train)
                    X_train = X_train.to("cpu")

                embedding_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = (embedding_train.cpu().detach().numpy())

                y_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_train.cpu().numpy()

                y_pred_train_label = loss.pred_labels(embedding=embedding_train)
                y_pred_train_array[
                    batch_train * batch_size : batch_train * batch_size + batch_size
                ] = y_pred_train_label.cpu().numpy()

                # speed perturb
                if speed_purturb:
                    X_train = self.speed_perturb(X_train, speed_factors=speed_factors)

                # mixup
                if mixup:
                    X_train, y_train = self.mixup(
                        X=X_train, y=y_train, alpha=alpha_mixup
                    )

                # training mode
                model.train()
                loss.train()

                # to device
                X_train = X_train.to(self.device)
                y_train_loss = y_train[:, 1].to(self.device)

                # forward pass
                embedding_train = model(X_train)

                # calculate the loss
                loss_train_this_batch = loss(embedding_train, y_train_loss)
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

            # # create fake labels for debugging
            # y_pred_train_array = np.random.randint(
            #     low=0, high=14, size=y_pred_train_array.shape
            # )
            # y_pred_test_array = np.random.randint(
            #     low=0, high=14, size=y_pred_test_array.shape
            # )

            # type labels
            type_labels_train = ["train_source", "train_target"]
            type_labels_test = [
                "test_source_normal",
                "test_source_anomaly",
                "test_target_normal",
                "test_target_anomaly",
            ]

            # loss train, loss train source, loss train target
            loss_train = loss_train / len(train_loader)
            loss_train_source, loss_train_target = self.loss_source_target(
                embedding_array=embedding_train_array,
                y_true_array=y_train_array,
                loss=loss,
                type_labels=type_labels_train,
            )

            # loss test, loss_test_source_normal, loss_test_source_anomaly, loss_test_target_normal, loss_test_target_anomaly
            loss_test = loss.calculate_loss(
                embedding=embedding_test_array, y_true=y_test_array[:, 1]
            )
            (
                loss_test_source_normal,
                loss_test_source_anomaly,
                loss_test_target_normal,
                loss_test_target_anomaly,
            ) = self.loss_source_target(
                embedding_array=embedding_test_array,
                y_true_array=y_test_array,
                loss=loss,
                type_labels=type_labels_test,
            )

            # accuracy train
            accuracy_train = accuracy_score(
                y_pred=y_pred_train_array, y_true=y_train_array[:, 1]
            )
            accuracy_train_source, accuracy_train_target = self.accuracy_source_target(
                y_pred_array=y_pred_train_array,
                y_true_array=y_train_array,
                type_labels=type_labels_train,
            )
            # accuracy test, accuracy_test_source_normal, accuracy_test_source_anomaly, accuracy_test_target_normal, accuracy_test_target_anomaly
            accuracy_test = accuracy_score(
                y_pred=y_pred_test_array, y_true=y_test_array[:, 1]
            )
            (
                accuracy_test_source_normal,
                accuracy_test_source_anomaly,
                accuracy_test_target_normal,
                accuracy_test_target_anomaly,
            ) = self.accuracy_source_target(
                y_pred_array=y_pred_test_array,
                y_true_array=y_test_array,
                type_labels=type_labels_test,
            )

            # confusion matrix
            cm_train = confusion_matrix(
                y_pred=y_pred_train_array, y_true=y_train_array[:, 1]
            )
            cm_train_fig = self.plot_confusion_matrix(cm=cm_train, name="train")

            cm_test = confusion_matrix(
                y_pred=y_pred_test_array, y_true=y_test_array[:, 1]
            )
            cm_test_fig = self.plot_confusion_matrix(cm=cm_test, name="test")

            # anomaly score and anomaly decision
            anomaly_score, anomaly_decision = self.anomaly_score_decision(
                k=k,
                distance=distance,
                embedding_train_array=embedding_train_array,
                embedding_test_array=embedding_test_array,
                y_pred_train_array=y_pred_train_array,
                y_pred_test_array=y_pred_test_array,
                y_test_array=y_test_array,
                loss=loss,
                percentile=percentile,
            )

            # accuracy decision
            if anomaly_score is not None and anomaly_decision is not None:

                # accuracy decision
                accuracy_decision = self.accuracy_decision(
                    anomaly_decision=anomaly_decision, y_test_array=y_test_array
                )

                # auc_roc and p_auc_roc
                (
                    fig_auc_roc,
                    accuracy_decision_dict,
                    hmean_total,
                ) = self.auc_pauc(
                    anomaly_score=anomaly_score,
                    anomaly_decision=anomaly_decision,
                    y_test_array=y_test_array,
                    ep=ep,
                )

                # log the metrics
                metrics_decision = {**accuracy_decision_dict}

                metrics_decision["accuracy_decision"] = accuracy_decision
                metrics_decision["hmean_total"] = hmean_total

                run["metrics_decision"].append(metrics_decision, step=ep)

                # log the image of auc_roc and p_auc_roc
                run["metrics_decision/auc_roc_p_auc_roc_hmean"].append(
                    fig_auc_roc, step=ep
                )

            # log the metrics in neptune
            metrics_train = {
                "loss_train": loss_train,
                "loss_train_source": loss_train_source,
                "loss_train_target": loss_train_target,
                "accuracy_train": accuracy_train,
                "accuracy_train_source": accuracy_train_source,
                "accuracy_train_target": accuracy_train_target,
            }

            metrics_test = {
                "loss_test": loss_test,
                "loss_test_source_normal": loss_test_source_normal,
                "loss_test_source_anomaly": loss_test_source_anomaly,
                "loss_test_target_normal": loss_test_target_normal,
                "loss_test_target_anomaly": loss_test_target_anomaly,
                "accuracy_test": accuracy_test,
                "accuracy_test_source_normal": accuracy_test_source_normal,
                "accuracy_test_source_anomaly": accuracy_test_source_anomaly,
                "accuracy_test_target_normal": accuracy_test_target_normal,
                "accuracy_test_target_anomaly": accuracy_test_target_anomaly,
            }

            run["metrics_train"].append(metrics_train, step=ep)
            run["metrics_test"].append(metrics_test, step=ep)

            # log the images of confusion matrix
            run["metrics_train/confusion_matrix_train"].append(cm_train_fig, step=ep)
            plt.close()
            run["metrics_test/confusion_matrix_test"].append(cm_test_fig, step=ep)
            plt.close()


# run this script
if __name__ == "__main__":

    # set the seed
    seed = utils.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("\n")

    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # hyperparameters
    lr = utils.lr_new_mixup
    emb_size = utils.emb_size_new_mixup
    batch_size = utils.batch_size_new_mixup
    wd = utils.wd_new_mixup
    epochs = utils.epochs_new_mixup
    optimizer_name = utils.optimizer_name_new_mixup
    model_name = utils.model_name_new_mixup
    scale = utils.scale_new_mixup
    margin = utils.margin_new_mixup
    loss_name = utils.loss_name_new_mixup
    classifier_head = utils.classifier_head_new_mixup
    window_size = utils.window_size_new_mixup
    dropout = utils.dropout_new_mixup
    hop_size = utils.hop_size_new_mixup
    k = utils.k_new_mixup
    percentile = utils.percentile_new_mixup
    distance = utils.distance_new_mixup
    speed_purturb = utils.speed_purturb_new_mixup
    speed_factors = utils.speed_factors_new_mixup
    mixup = utils.mixup_new_mixup
    alpha_mixup = utils.mixup_alpha_new_mixup

    # general hyperparameters
    project = utils.project
    api_token = utils.api_token
    data_name = utils.data_name_dev

    # data preprocessing
    anomaly_detection = AnomalyDetection(seed=seed, data_name=data_name)

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
        dropout=dropout,
        optimizer_name=optimizer_name,
        loss_name=loss_name,
        window_size=window_size,
        hop_size=hop_size,
        distance=distance,
        speed_factors=speed_factors,
        speed_purturb=speed_purturb,
        mixup=mixup,
        alpha_mixup=alpha_mixup,
    )
