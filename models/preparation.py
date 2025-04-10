import torch
from torch import nn
from beats.beats_custom import BEATsCustom
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import numpy as np
from loss import AdaCosLoss, ArcFaceLoss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingWarmRestarts,
)
from datetime import datetime
from peft import LoraConfig, get_peft_model
import random
import itertools
from scipy.stats import hmean
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# from ..data.preprocessing import DataPreprocessing

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing


class ModelDataPrepraration(DataPreprocessing):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

        # information this directory
        self.path_directory_models = os.path.dirname(os.path.abspath(__file__))

        # pretrained models BEATs
        self.path_pretrained_models_directory = os.path.join(
            self.path_directory_models, "pretrained_models"
        )

        self.path_beat_iter3_state_dict = os.path.join(
            self.path_pretrained_models_directory, "BEATs_iter3.pt"
        )

        if not os.path.exists(
            self.path_pretrained_models_directory
        ) or not os.path.exists(self.path_beat_iter3_state_dict):
            import download_models

        # path hpo
        self.path_hpo_directory = os.path.join(self.path_directory_models, "HPO")

        # configuration of the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            self.n_gpus = torch.cuda.device_count()
            self.vram = (
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
            )
        else:
            self.gpu_name = None
            self.n_gpus = None
            self.vram = None

        # set the seed
        self.set_seed()

    def set_seed(self):
        """
        set the seed for whole project
        """
        # Set seed for PyTorch

        torch.manual_seed(self.seed)

        # Set seed for CUDA (if using GPUs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # For multi-GPU setups

            # Ensure deterministic behavior for PyTorch operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Set seed for Python's random module
        random.seed(self.seed)

        # Set seed for NumPy
        np.random.seed(self.seed)

    def load_model(
        self,
        input_size=10,
        emb_size=None,
        lora=False,
        r=None,
        lora_alpha=None,
        lora_dropout=None,
    ):
        # function to load model beats
        model = BEATsCustom(
            path_state_dict=self.path_beat_iter3_state_dict,
            input_size=input_size,
            emb_size=emb_size,
        )

        # check if lora
        if (
            lora
            and r is not None
            and lora_alpha is not None
            and lora_dropout is not None
        ):
            # freeze all layers except asp layers
            for param in model.parameters():
                param.requires_grad = False
            for param in model.asp.parameters():
                param.requires_grad = True

            # apply lora
            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
                target_modules=[
                    # "k_proj",
                    "v_proj",
                    "q_proj",
                    "out_proj",
                    # "grep_linear",
                ],
            )
            model = get_peft_model(model, lora_config)

        return model

    def load_dataset_tensor(self, k_smote=5):
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
        ) = self.load_data_raw()

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

    def data_loader(self, dataset, batch_size, len_factor=None, uniform_sampling=False):
        """
        convert tensor data to dataloader
        """
        # check if uniform_sampling
        if uniform_sampling and len_factor is not None:
            # total number of instances
            num_instances = int(len_factor * len(dataset))

            # check if a last batch has one instance (avoid error for batchnorm in model)
            if num_instances % batch_size == 1:
                num_instances = num_instances + 1

            # split to get the label
            _, y_train_smote = dataset.tensors

            # instance weight = weight only for smote dataset (same number of labels)
            class_instances_count = torch.tensor(
                [(y_train_smote == l).sum() for l in torch.unique(y_train_smote)]
            )
            weight = 1.0 / class_instances_count
            num_instances_original = len(dataset)

            # instance_weight = torch.tensor([weight[l] for l in y_train_smote])
            instance_weight = torch.tensor(
                [weight[0] for i in range(num_instances_original)]
            )

            # batch uniform sampling
            sampler = WeightedRandomSampler(
                weights=instance_weight,
                num_samples=num_instances,
            )

            dataloader = DataLoader(
                dataset=dataset, sampler=sampler, batch_size=batch_size
            )

        else:
            dataloader = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=False
            )

        return dataloader

    def sample_machines(self, num_train_machines: int = 5, num_splits: int = 5):
        """
        sample the machines for cross validation
        """
        # generate all possible combinations
        combinations = list(itertools.combinations(self.machines, num_train_machines))
        np.random.shuffle(combinations)

        # Sample 5 unique combinations randomly
        index_sampled_combinations = np.random.choice(
            len(combinations), num_splits, replace=False
        )
        sampled_result = [combinations[i] for i in index_sampled_combinations]

        return sampled_result

    def sort_data_machines(self, dataset: TensorDataset, list_machines: list):
        """
        sort the X and y based on machine
        """
        # get the labels of the list machines
        labels = torch.tensor(self.label_machine(list_machines))

        # get X and y from TensorDataset
        X, y = dataset.tensors

        # get the indices of labels from list machines
        if y.ndim == 1:
            mask = torch.isin(y, labels)
        else:
            mask = torch.isin(y[:, 1], labels)
        indices_labels = torch.nonzero(mask, as_tuple=True)[0]

        # sort X and y based on labels
        X = X[indices_labels]
        y = y[indices_labels]

        # turn X and y back to Tensor Dataset
        dataset = TensorDataset(X, y)

        return dataset

    def name_saved_model(self, index_split=None):
        """
        get the model name to save it
        """
        # Get current date and time
        current_datetime = datetime.now()

        # Format as string
        datetime_string = current_datetime.strftime("%Y_%m_%d-%H_%M_%S")

        # create model_name for not HPO
        if index_split == None:
            model_name = "model_{}.pth".format(datetime_string)
        else:
            model_name = "model_checkpoint_{}".format(index_split)

        return model_name

    def load_loss(
        self,
        loss_type: str,
        num_classes: int,
        emb_size: int = None,
        margin: int = None,
        scale: int = None,
    ):
        # load loss based on loss type
        if loss_type == "adacos":
            loss = AdaCosLoss(num_classes=num_classes, emb_size=emb_size)
        elif loss_type == "arcface":
            if margin == None:
                margin = 0.5
            if scale == None:
                scale = 64
            loss = ArcFaceLoss(
                num_classes=num_classes,
                emb_size=emb_size,
                margin=margin,
                scale=scale,
            )

        return loss

    def load_scheduler(
        self, optimizer, scheduler_type: str, step_warmup: int, min_lr: float = None
    ):
        """
        load scheduler for learning rate
        """
        if scheduler_type == "cosine_restarts" and min_lr is not None:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=step_warmup, eta_min=min_lr
            )
        elif scheduler_type == "linear_restarts":

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

        return scheduler

    def hyperparameters_configuration_dict(self, **kwargs):
        """
        hyperparameter dictionary
        """

        # pop function for a given dict
        def dict_pop(dictionary: dict, *arg):
            """
            function to pop the keys and values
            """
            for i in arg:
                dictionary.pop(i, None)
            return dictionary

        # pop some keys for hyperparameters dictionary
        if set({"lora", "HPO", "loss_type", "list_machines"}).issubset(
            set(kwargs.keys())
        ):
            # lora
            lora = kwargs["lora"]
            if not lora:
                dict_pop(kwargs, "r", "lora_alpha", "lora_dropout")

            # HPO
            HPO = kwargs["HPO"]
            if not HPO:
                dict_pop(kwargs, "trial")
                dict_pop(kwargs, "num_train_machines")

            # arcface
            loss_type = kwargs["loss_type"]
            if loss_type != "arcface":
                dict_pop(kwargs, "margin", "scale")

        return kwargs

    def save_pretrained_model_loss(
        self,
        model_pretrained: nn.Module,
        loss_pretrained: nn.Module,
        optimizer: torch.optim.AdamW,
        hyperparameters: dict,
        configuration: dict,
        scheduler: torch.optim.lr_scheduler.LambdaLR = None,
        knn_pretrained: list = None,
        scaler_pretrained: StandardScaler = None,
    ):
        """
        save the pretrained model in pretrained_model directory
        """
        # get model name for hpo and not hpo
        name_saved_model = hyperparameters["name_saved_model"]
        HPO = hyperparameters["HPO"]
        path_pretrained_model_loss = (
            os.path.join(self.path_pretrained_models_directory, name_saved_model)
            if not HPO
            else os.path.join(self.path_hpo_directory, name_saved_model)
        )

        torch.save(
            {
                "model_state_dict": model_pretrained.state_dict(),
                "loss_state_dict": loss_pretrained.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "knn_pretrained": knn_pretrained,
                "scaler_pretrained": scaler_pretrained,
                "hyperparameters": hyperparameters,
                "configuration": configuration,
            },
            path_pretrained_model_loss,
        )

        print(
            "pretrained model, loss, optimizer and hyperparameters saved to ",
            path_pretrained_model_loss,
        )

    def load_pretrained_model(self, pretrained_path: str):
        """
        load the pretrained model
        """
        # load the state_dict given the pretrained_path
        loaded_dict = torch.load(pretrained_path, map_location=self.device)

        # extract the state dict of each model from dictionary keys
        model_state_dict = loaded_dict["model_state_dict"]
        loss_state_dict = loaded_dict["loss_state_dict"]
        optimizer_state_dict = loaded_dict["optimizer_state_dict"]
        scheduler_state_dict = loaded_dict["scheduler_state_dict"]
        knn = loaded_dict["knn_pretrained"]
        scaler = loaded_dict["scaler_pretrained"]
        hyperparameters = loaded_dict["hyperparameters"]
        configuration = loaded_dict["configuration"]

        # load model neural network
        emb_size = hyperparameters["emb_size"]
        lora = hyperparameters["lora"]
        if lora:
            r = hyperparameters["r"]
            lora_alpha = hyperparameters["lora_alpha"]
            lora_dropout = hyperparameters["lora_dropout"]
            model = self.load_model(
                emb_size=emb_size,
                r=r,
                lora=lora,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        else:
            model = self.load_model(emb_size=emb_size)
        model.load_state_dict(model_state_dict)
        model = model.to(self.device)

        # load loss
        loss_type = hyperparameters["loss_type"]
        if loss_type == "adacos":
            loss = AdaCosLoss(
                num_classes=self.num_classes_attribute(), emb_size=emb_size
            )
        elif loss_type == "arcface":
            margin = hyperparameters["margin"]
            scale = hyperparameters["scale"]
            loss = ArcFaceLoss(
                num_classes=self.num_classes_attribute(),
                emb_size=emb_size,
                margin=margin,
                scale=scale,
            )
        loss.load_state_dict(loss_state_dict)
        loss = loss.to(self.device)

        # load optimizer
        learning_rate = hyperparameters["learning_rate"]
        optimizer = self.perform_load_optimizer(
            model=model, loss=loss, learning_rate=learning_rate
        )
        optimizer.load_state_dict(optimizer_state_dict)

        # load scheduler
        step_warmup = hyperparameters["step_warmup"]
        min_lr = hyperparameters["min_lr"]
        scheduler_type = hyperparameters["scheduler_type"]
        scheduler = self.load_scheduler(
            optimizer=optimizer,
            step_warmup=step_warmup,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
        )
        scheduler.load_state_dict(scheduler_state_dict)

        return (
            model,
            loss,
            optimizer,
            scheduler,
            hyperparameters,
            configuration,
            knn,
            scaler,
        )

    def perform_load_data(self, k_smote, HPO, batch_size, len_factor, list_machines):
        """
        perform all steps until data loader
        """
        # load data
        (
            dataset_smote,
            dataset_train_attribute,
            dataset_test_attribute,
        ) = self.load_dataset_tensor(k_smote=k_smote)

        # sort data based on list_machines
        if HPO:
            dataset_smote = self.sort_data_machines(
                dataset=dataset_smote, list_machines=list_machines
            )
            dataset_train_attribute = self.sort_data_machines(
                dataset=dataset_train_attribute, list_machines=list_machines
            )
            dataset_test_attribute = self.sort_data_machines(
                dataset=dataset_test_attribute, list_machines=list_machines
            )
            # name_trial = f"trial {trial.number} split {index_split}"

        dataloader_smote_attribute = self.data_loader(
            dataset=dataset_smote,
            batch_size=batch_size,
            len_factor=len_factor,
            uniform_sampling=True,
        )

        dataloader_train_attribute = self.data_loader(
            dataset=dataset_train_attribute, batch_size=batch_size
        )
        dataloader_test_attribute = self.data_loader(
            dataset=dataset_test_attribute, batch_size=batch_size
        )

        # get the num instances smote
        num_instances_smote = int(len(dataset_smote) * len_factor)
        if num_instances_smote % batch_size == 1:
            num_instances_smote = num_instances_smote + 1

        return (
            dataloader_smote_attribute,
            dataloader_train_attribute,
            dataloader_test_attribute,
            num_instances_smote,
        )

    def perform_load_model(
        self, input_size, emb_size, lora, r, lora_alpha, lora_dropout
    ):
        """
        perform load model to device, get emb_size and number of parameters, trainable parameters
        """
        # load model
        model = self.load_model(
            input_size=input_size,
            emb_size=emb_size,
            lora=lora,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        if emb_size == None:
            emb_size = model.embedding_asp

        # model to device
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)

        # number trainable parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_params_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        return model, num_params, num_params_trainable

    def perform_load_loss(self, loss_type, emb_size, margin, scale):
        """
        perform load adacos or arcface loss
        """
        # loss
        num_classes_train = self.num_classes_attribute()
        loss = self.load_loss(
            loss_type=loss_type,
            num_classes=num_classes_train,
            emb_size=emb_size,
            margin=margin,
            scale=scale,
        )

        return loss

    def perform_load_optimizer(self, model, loss, learning_rate):
        """
        perform the load optimizer
        """
        # load optimizer
        parameters = list(model.parameters()) + list(loss.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

        return optimizer

    def accuracy_attribute(
        self,
        y_true_array: np.array,
        y_pred_label_array: np.array,
        type_labels=["train_source_normal", "train_target_normal"],
    ):
        """
        get the accuracy for attribute given y_true_array shape (index, attribute, condition)
        and y_pred_label_array shape (pred_attribute)
        """
        # get the indices
        indices = self.get_indices(
            y_true_array=y_true_array,
            type_labels=type_labels,
        )
        y_true_array = y_true_array[:, 1]

        # calculate accuracy
        accuracy = []
        for idx in indices:
            y_pred = y_pred_label_array[idx]
            y_true = y_true_array[idx]
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
        ts_ids = [self.id_timeseries_analysis(keys=typ) for typ in type_labels]

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
        # plot the confusion matrix
        fig = plt.figure(figsize=(35, 16))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        titel = "Confusion Matrix {}".format(type_data)
        plt.title(titel, fontsize=18)
        plt.xlabel("Predicted Labels", fontsize=15)
        plt.ylabel("True Labels", fontsize=15)

        return fig

    def decision_knn(
        self,
        k_neighbors,
        embedding_train_array,
        embedding_test_array,
        y_pred_train_array,
        y_pred_test_array,
        y_true_test_array,
    ):
        """
        use knn to make decision if timeseries in test data normal or anomaly
        """
        # knn model with cosine distance
        knn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")

        # fit the train data.
        knn.fit(embedding_train_array)
        distance_train, _ = knn.kneighbors(embedding_train_array)
        distance_train = np.mean(distance_train[:, 1:], axis=1)
        # distance_train = np.mean(distance_train, axis=1)
        # print("distance_train shape:", distance_train.shape)

        # normalize the distances
        scaler = StandardScaler()
        scaler.fit(distance_train.reshape(-1, 1))
        distance_train = scaler.transform(distance_train.reshape(-1, 1)).reshape(
            len(distance_train),
        )
        threshold = 3

        # ids of the test data
        ids = self.id_timeseries_analysis(keys="test")
        ids = y_true_test_array[:, 0]
        # indices = self.get_indices(y_true_array=y_pred_test_array,type_labels=["test"])

        # anomaly detection score list
        decision_anomaly_score_test = np.empty(shape=(len(ids), 4))
        decision_anomaly_score_test[:, 0] = ids
        decision_anomaly_score_test[:, 3] = [threshold for i in range(len(ids))]

        # get the distance test
        distance_test, _ = knn.kneighbors(embedding_test_array)
        distance_test = distance_test[:, :-1]
        distance_test = np.mean(distance_test, axis=1)

        # normalize as anomaly score and compare with the threshold the make the decision
        distance_test = scaler.transform(distance_test.reshape(-1, 1)).reshape(
            len(distance_test),
        )
        # distance_test = distance_test[0]
        # print("distance_test:", distance_test)
        decision_anomaly_score_test[:, 2] = distance_test

        # get the decision
        decisions = [0 if d < threshold else 1 for d in distance_test]
        decision_anomaly_score_test[:, 1] = decisions

        # decision_anomaly_score_test = np.array(decision_anomaly_score_test)
        # print("decision_anomaly_score_test:", decision_anomaly_score_test)

        return decision_anomaly_score_test, knn, scaler

    def true_test_condition_array(self):
        """
        y_true_test_condition shape (id, condition_true)
        """
        # y_true_test_condition_array shape (id, condition_true)
        y_true_test_condition = self.timeseries_information()["Condition"].to_numpy()
        y_true_test_condition_array = np.array(
            [
                [id, y_true_test_condition[id]]
                for id in self.id_timeseries_analysis(keys="test")
            ]
        )
        return y_true_test_condition_array

    def accuracy_decision(self, decision_anomaly_score_test):
        """
        accuracy decision given the prediction of the condition (normal or anomaly)
        given the decision test shape (id, condition_pred)
        """
        # type_labels of test data
        type_labels = [
            "test_source_normal",
            "test_target_normal",
            "test_source_anomaly",
            "test_target_anomaly",
        ]

        # y_true_test_condition_array shape (id, condition_true)
        y_true_test_condition_array = self.true_test_condition_array()

        # get the indices for each type_labels
        indices = self.get_indices(
            y_true_array=y_true_test_condition_array, type_labels=type_labels
        )

        # calculate accuracy
        accuracy = []
        for idx in indices:
            y_pred = decision_anomaly_score_test[idx, 1]
            y_true = y_true_test_condition_array[idx, 1]
            acc = accuracy_score(y_pred=y_pred, y_true=y_true)
            accuracy.append(acc)

        return accuracy

    def auc_pauc_hmean(
        self, decision_anomaly_score_test, list_machines=None, y_true_test_array=None
    ):
        """
        calculate auc pauc of test machine domain
        given anomaly score shape (id, anomaly score) and dicision test shape (id, condition pred)
        with type_labels_hmean
        """
        # y_true_test_condition_array shape (id, condition_true)
        y_true_test_condition_array = self.true_test_condition_array()

        # get the indices for each type_labels_hmean
        type_labels_hmean, type_labels_hmean_auc, type_labels_hmean_pauc = (
            self.type_labels_hmean(list_machines=list_machines)
        )
        indices = self.get_indices(
            y_true_array=y_true_test_condition_array, type_labels=type_labels_hmean
        )

        # fpr_max and fpr_min for pauc
        fpr_min = 0
        fpr_max = 0.1

        # create suplots
        n_cols = len(self.machines) if list_machines == None else len(list_machines)
        n_rows = 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 20))
        axes = axes.flatten()

        # list for auc, pauc method1 and 2
        auc_test = []
        pauc_test_1 = []
        pauc_test_2 = []

        # loop through all the axes, each axes is plot of auc and pauc of machine_domain
        for i in range(len(type_labels_hmean)):

            # get the name of label, y_score, y_true
            test_machine_domain = type_labels_hmean[i]
            # idx = indices[i]
            id_test_machine_domain = self.id_timeseries_analysis(
                keys=test_machine_domain
            )
            idx = np.where(np.isin(y_true_test_array[:, 0], id_test_machine_domain))[0]
            y_score = decision_anomaly_score_test[idx, 2]
            y_true = y_true_test_condition_array[idx, 1]
            y_pred = decision_anomaly_score_test[idx, 1]

            # calculate auc pauc
            fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
            auc = roc_auc_score(y_true=y_true, y_score=y_score)
            # print("auc:", auc)

            # calculate p auc roc
            pauc = roc_auc_score(y_true=y_true, y_score=y_score, max_fpr=fpr_max)
            # print("pauc:", pauc)

            # accuracy machine domain
            accuraccy_machine_domain = accuracy_score(y_pred=y_pred, y_true=y_true)

            # plot the auc
            name_auc = (
                "AUC_12" if test_machine_domain in type_labels_hmean_auc else "AUC"
            )
            axes[i].plot(fpr, tpr, label="{} {:.4f}".format(name_auc, auc))

            # plot the pauc
            name_pauc = (
                "PAUC_1" if test_machine_domain in type_labels_hmean_auc else "PAUC_2"
            )
            axes[i].fill_between(
                fpr,
                tpr,
                where=(fpr >= fpr_min) & (fpr <= fpr_max),
                color="orange",
                alpha=0.3,
                label="{} {:.4f}".format(name_pauc, pauc),
            )

            # get title and axis
            axes[i].set_title(
                "{}\nacc {:.4f}".format(test_machine_domain, accuraccy_machine_domain)
            )
            axes[i].set_xlabel("FPR")
            axes[i].set_ylabel("TPR")
            axes[i].legend(loc="lower right")

            # save to list
            if test_machine_domain in type_labels_hmean_auc:
                pauc_test_1.append(pauc)
                auc_test.append(auc)

            else:
                pauc_test_2.append(pauc)

        # calculate hmean total
        # print("pauc_test_1:", pauc_test_1)
        # print("pauc_test_2:", pauc_test_2)
        # print("auc_test", auc_test)
        hmean_total_1 = hmean(auc_test + pauc_test_1)
        hmean_total_2 = hmean(auc_test + pauc_test_2)

        # suplite of the fig to report the hmean
        fig.suptitle(
            "Hmean_1 {:.4f} Hmean_2 {:.4f}".format(hmean_total_1, hmean_total_2)
        )

        return fig, hmean_total_2


if __name__ == "__main__":

    data_name = "develop"
    seed = 1998
    cv_class = ModelDataPrepraration(data_name=data_name, seed=seed)

    # sample_machines = cv_class.sample_machines(num_splits=5, num_train_machines=5)
    # print("sample_machines len:", len(sample_machines))
    # print("sample_machines:", sample_machines)

    # from collections import Counter

    # flattened = [elem for combo in sample_machines for elem in combo]
    # # Count occurrences of each element
    # element_counts = Counter(flattened)

    # print("Element counts:", element_counts)

    cv_class.cross_validation()

    # hyperparameters_configuration_dict = cv_class.check_function()
    # print("hyperparameters_configuration_dict:", hyperparameters_configuration_dict)
