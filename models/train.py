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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import neptune
from neptune.utils import stringify_unsupported
from torch.optim.lr_scheduler import LambdaLR
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hmean

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
        if not os.path.exists(self.path_pretrained_models_directory):
            import download_models

        self.path_beat_iter3_state_dict = os.path.join(
            self.path_pretrained_models_directory, "BEATs_iter3.pt"
        )

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

        # knn models
        self.name_knn = [
            "{}_{}_{}".format("train", m, d)
            for m in self.machines
            for d in self.domain_data
        ]

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
            dataloader = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=False
            )

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
        k_neighbors,
        emb_size=None,
    ):
        """
        main function to find the result
        """
        # fix hyperparameter to suit with vram
        if self.vram < 23:
            step_warmup = 480
            batch_size = 8
            step_accumulation = 32

        # init neptune
        run = neptune.init_run(project=project, api_token=api_token)

        # load data
        (
            dataset_smote,
            train_dataset_attribute,
            test_dataset_attribute,
        ) = self.load_dataset_tensor(k_smote=k_smote)

        # dataloader
        dataloader_smote_uniform = self.data_loader(
            dataset=dataset_smote,
            batch_size=batch_size,
            num_instances=num_instances,
            uniform_sampling=True,
        )
        dataloader_smote_attribute = self.data_loader(
            dataset=dataset_smote, batch_size=batch_size
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
            print("emb_size:", emb_size)
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
        configuration["device"] = stringify_unsupported(self.device)
        configuration["gpu_name"] = self.gpu_name
        configuration["vram"] = self.vram
        run["configuration"] = configuration

        # training attribute classification
        self.training_loop(
            run=run,
            dataloader_smote_uniform=dataloader_smote_uniform,
            dataloader_smote_attribute=dataloader_smote_attribute,
            dataloader_train_attribute=dataloader_train_attribute,
            dataloader_test_attribute=dataloader_test_attribute,
            batch_size=batch_size,
            model=model,
            step_accumulation=step_accumulation,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            k_neighbors=k_neighbors,
            emb_size=emb_size,
        )

    def training_loop(
        self,
        run: neptune.init_run,
        dataloader_smote_uniform: DataLoader,
        dataloader_smote_attribute: DataLoader,
        dataloader_train_attribute: DataLoader,
        dataloader_test_attribute: DataLoader,
        batch_size: int,
        model: nn.Module,
        step_accumulation: int,
        loss: AdaCosLoss,
        optimizer: torch.optim.AdamW,
        scheduler: LambdaLR,
        k_neighbors: int,
        emb_size: int,
    ):
        """
        training loop for smote data
        """

        # step report and evaluation
        step_eval = step_accumulation * 50

        # loss train
        loss_smote_uniform_total = 0

        for iter_smote_uniform, (X_smote_uniform, y_smote_uniform) in enumerate(
            dataloader_smote_uniform
        ):
            # model in traning model
            model.train()
            loss.train()

            # data to device
            X_smote_uniform = X_smote_uniform.to(self.device)
            y_smote_uniform = y_smote_uniform.to(self.device)

            # forward pass
            embedding_smote_uniform = model(X_smote_uniform)

            # calculate the loss
            loss_smote_uniform = loss(embedding_smote_uniform, y_smote_uniform)
            loss_smote_uniform_total = (
                loss_smote_uniform_total + loss_smote_uniform.item()
            )

            # update the loss
            loss_smote_uniform.backward()

            # gradient accumulated and report loss
            if (iter_smote_uniform + 1) % step_accumulation == 0:
                # update model weights and zero grad
                optimizer.step()
                optimizer.zero_grad()

                # report loss
                loss_smote_uniform_total = loss_smote_uniform_total / step_accumulation
                run["smote_uniform/loss_smote_total"].append(
                    loss_smote_uniform_total, step=iter_smote_uniform
                )
                loss_smote_uniform_total = 0

            # update scheduler
            scheduler.step()

            # report the loss and evaluation mode every 1000 iteration
            if (iter_smote_uniform + 1) % step_eval == 0:
                # type_labels
                type_labels_train = ["train_source_normal", "train_target_normal"]
                type_labels_test = [
                    "test_source_normal",
                    "test_target_normal",
                    "test_source_anomaly",
                    "test_target_anomaly",
                ]
                type_labels_smote_knn = ["SmoteAttributeKNN"]

                # evaluation mode for train data attribute
                (
                    accuracy_train_dict,
                    embedding_train_array,
                    y_true_train_array,
                    y_pred_label_train_array,
                ) = self.evaluation_mode(
                    run=run,
                    iter_smote_uniform=iter_smote_uniform,
                    batch_size=batch_size,
                    model=model,
                    loss=loss,
                    dataloader_attribute=dataloader_train_attribute,
                    emb_size=emb_size,
                    type_labels=type_labels_train,
                )

                # evaluation mode for test data attribute
                (
                    accuracy_test_dict,
                    embedding_test_array,
                    y_true_test_array,
                    y_pred_label_test_array,
                ) = self.evaluation_mode(
                    run=run,
                    iter_smote_uniform=iter_smote_uniform,
                    batch_size=batch_size,
                    model=model,
                    loss=loss,
                    dataloader_attribute=dataloader_test_attribute,
                    emb_size=emb_size,
                    type_labels=type_labels_test,
                )

                # only apply knn for anomly detechtion for a good performance
                if all(acc > 0.9 for acc in accuracy_train_dict.values()):
                    (
                        accuracy_smote_dict,
                        embedding_smote_array,
                        y_true_smote_array,
                        y_pred_label_smote_array,
                    ) = self.evaluation_mode(
                        run=run,
                        iter_smote_uniform=iter_smote_uniform,
                        batch_size=batch_size,
                        model=model,
                        loss=loss,
                        dataloader_attribute=dataloader_smote_attribute,
                        emb_size=emb_size,
                        type_labels=type_labels_smote_knn,
                    )

                    # only use knn if accuracy smote has good performance
                    if all(acc > 0.9 for acc in accuracy_smote_dict.values()):

                        # use knn to get the decision and anomaly score
                        decision_anomaly_score_test = self.decision_knn(
                            k_neighbors=k_neighbors,
                            embedding_train_array=embedding_smote_array,
                            embedding_test_array=embedding_test_array,
                            y_pred_train_array=y_pred_label_smote_array,
                            y_pred_test_array=y_pred_label_test_array,
                            y_true_test_array=y_true_test_array,
                        )

                        # accuracy decision
                        accuracy_decisions = self.accuracy_decision(
                            decision_anomaly_score_test=decision_anomaly_score_test
                        )
                        for typ_l, acc in zip(type_labels_test, accuracy_decisions):
                            run["{}/accuracy_{}".format("decision", typ_l)].append(
                                acc, step=iter_smote_uniform
                            )

                        # anomaly score and hmean
                        hmean_img, hmean_total = self.auc_pauc_hmean(
                            decision_anomaly_score_test=decision_anomaly_score_test
                        )

                        run["score/hmean"].append(hmean_total, step=iter_smote_uniform)
                        run["score/auc_pauc_hmean"].append(
                            hmean_img, step=iter_smote_uniform
                        )
                        plt.close()

            current_lr = optimizer.param_groups[0]["lr"]
            run["smote_uniform/current_lr"].append(current_lr, step=iter_smote_uniform)

    def evaluation_mode(
        self,
        run: neptune.init_run,
        iter_smote_uniform: int,
        batch_size: int,
        model: nn.Module,
        loss: AdaCosLoss,
        dataloader_attribute: DataLoader,
        emb_size: int = None,
        type_labels=["train_source_normal", "train_target_normal"],
    ):
        """
        evaluation mode in training loop
        """
        # model in evaluation and no grad mode
        model.eval()
        loss.eval()

        # y_true, y_pred, embedding array
        type_data = type_labels[0].split("_")[0]
        len_dataset = dataloader_attribute.dataset.tensors[0].shape[0]
        y_pred_label_array = np.empty(shape=(len_dataset,))
        if type_data in ["train", "test"]:
            y_true_array = np.empty(shape=(len_dataset, 3))
        else:
            y_true_array = np.empty(shape=(len_dataset,))

        embedding_array = np.empty(shape=(len_dataset, emb_size))

        with torch.no_grad():
            for iter_eval, (X, y) in enumerate(dataloader_attribute):
                # data to device
                X = X.to(self.device)

                # forward pass
                embedding = model(X)

                # pred the label
                y_pred_label = loss.pred_labels(embedding=embedding)

                # save to array
                embedding_array[
                    iter_eval * batch_size : iter_eval * batch_size + batch_size
                ] = embedding.cpu().numpy()
                y_true_array[
                    iter_eval * batch_size : iter_eval * batch_size + batch_size
                ] = y.cpu().numpy()
                y_pred_label_array[
                    iter_eval * batch_size : iter_eval * batch_size + batch_size
                ] = y_pred_label.cpu().numpy()

            # calculate accuracy for train and test data
            if type_data in ["train", "test"]:
                accuracy_type_labels = self.accuracy_attribute(
                    y_true_array=y_true_array,
                    y_pred_label_array=y_pred_label_array,
                    type_labels=type_labels,
                )

            # calculate accuracy for train data smote
            else:
                accuracy_type_labels = [
                    accuracy_score(y_true=y_true_array, y_pred=y_pred_label_array)
                ]

            # calculate confusion matrix
            if type_data in ["train", "test"]:
                cm = confusion_matrix(
                    y_true=y_true_array[:, 1], y_pred=y_pred_label_array
                )
            else:
                cm = confusion_matrix(y_true=y_true_array, y_pred=y_pred_label_array)
            cm_img = self.plot_confusion_matrix(cm=cm, type_data=type_data)

            # save metrics in run
            accuracy_dict = {}
            for typ_l, acc in zip(type_labels, accuracy_type_labels):
                run["{}/accuracy_{}".format(type_data, typ_l)].append(
                    acc, step=iter_smote_uniform
                )
                accuracy_dict[typ_l] = acc

            run["{}/confusion_matrix".format(type_data)].append(
                cm_img, step=iter_smote_uniform
            )
            plt.close()

        return accuracy_dict, embedding_array, y_true_array, y_pred_label_array

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

        # list to save z score scaler and pretrained knn
        knn_train = []
        threshold_train = []
        scaler_train = []

        # train each knn based on the predicted y_pred_train
        for label in range(self.num_classes_attribute()):

            print("label", label)
            # get the indices of each label from embedding and y pred
            indices_train = np.where(y_pred_train_array == label)[0]
            # print("indices_train:", indices_train)

            # data and fít knn
            embedding_train_fit_knn = embedding_train_array[indices_train]
            knn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")
            knn.fit(embedding_train_fit_knn)

            # calculate distance
            distance_train, _ = knn.kneighbors(embedding_train_fit_knn)
            distance_train = np.mean(distance_train[:, 1:], axis=1)
            # print("distance_train:", distance_train)

            # normalize the distance train in range 0 and 1
            scaler = MinMaxScaler()
            scaler.fit(distance_train.reshape(-1, 1))
            distance_train = scaler.transform(distance_train.reshape(-1, 1)).reshape(
                len(distance_train),
            )
            # print("distance_train_normalize", distance_train)
            print("distance_train_normalize max", max(distance_train))
            print("distance_train_normalize min", min(distance_train))

            # calculate the threshold using percentile
            # threshold = np.percentile(distance_train, 99)
            # print("threshold:", threshold)

            # save to list
            knn_train.append(knn)
            threshold_train.append(threshold)
            scaler_train.append(scaler)

            print()

        decision_anomaly_score_test = []

        # loop through each id in test data
        for id in self.id_timeseries_analysis(keys="test"):

            # find the index of each id and their predict and embedding
            index_id_test = np.where(y_true_test_array[:, 0] == id)[0]
            print("id", id)
            print("index_id_test:", index_id_test)
            embedding_test_fit_knn = embedding_test_array[index_id_test]
            label_pred = int(y_pred_test_array[index_id_test][0])
            print("label_pred:", label_pred)

            # use knn, scaler and threshold from label pred
            knn = knn_train[label_pred]
            scaler = scaler_train[label_pred]
            # threshold = threshold_train[label_pred]
            # print("threshold:", threshold)
            threshold = 1

            # find the distance test of embedding test with correspond pred label
            distance_test, _ = knn.kneighbors(embedding_test_fit_knn)
            distance_test = np.mean(distance_test, axis=1)
            # print("distance_test:", distance_test)

            # normalize as anomaly score and compare with the threshold the make the decision
            distance_test = scaler.transform(distance_test.reshape(-1, 1)).reshape(
                len(distance_test),
            )
            distance_test = distance_test[0]
            print("distance_test:", distance_test)
            decision = 0 if distance_test < threshold else 1
            print("decision:", decision)

            # append to list
            decision_anomaly_score_test.append([id, decision, distance_test])

        print("decision_anomaly_score_test:", decision_anomaly_score_test)
        print()

        # convert decision test, anomaly score test to array
        decision_anomaly_score_test = np.array(decision_anomaly_score_test)

        return decision_anomaly_score_test

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

    def auc_pauc_hmean(self, decision_anomaly_score_test):
        """
        calculate auc pauc of test machine domain
        given anomaly score shape (id, anomaly score) and dicision test shape (id, condition pred)
        with type_labels_hmean
        """
        # y_true_test_condition_array shape (id, condition_true)
        y_true_test_condition_array = self.true_test_condition_array()

        # get the indices for each type_labels_hmean
        indices = self.get_indices(
            y_true_array=y_true_test_condition_array, type_labels=self.type_labels_hmean
        )

        # fpr_max and fpr_min for pauc
        fpr_min = 0
        fpr_max = 0.1

        # create suplots
        n_cols = 7
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 15))
        axes = axes.flatten()

        # list for auc, pauc, hmean
        auc_test = []
        pauc_test = []

        # loop through all the axes, each axes is plot of auc and pauc of machine_domain
        for i in range(len(self.type_labels_hmean)):

            # get the name of label, y_score, y_true
            test_machine_domain = self.type_labels_hmean[i]
            idx = indices[i]
            y_score = decision_anomaly_score_test[idx, 2]
            y_true = y_true_test_condition_array[idx, 1]
            y_pred = decision_anomaly_score_test[idx, 1]

            # calculate auc pauc
            fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
            auc = roc_auc_score(y_true=y_true, y_score=y_score)

            # calculate p auc roc
            pauc = roc_auc_score(y_true=y_true, y_score=y_score, max_fpr=fpr_max)

            # calculate h mean
            hmean_machine_domain = hmean([auc, pauc])

            # accuracy machine domain
            accuraccy_machine_domain = accuracy_score(y_pred=y_pred, y_true=y_true)

            # plot the auc
            axes[i].plot(fpr, tpr, label=f"AUC = {auc:.4f}")

            # plot the pauc
            axes[i].fill_between(
                fpr,
                tpr,
                where=(fpr >= fpr_min) & (fpr <= fpr_max),
                color="orange",
                alpha=0.3,
                label=f"PAUC = {pauc:.4f}",
            )

            # get title and axis
            axes[i].set_title(
                "{}\nacc {:.4f} hmean {:4f}".format(
                    test_machine_domain, accuraccy_machine_domain, hmean_machine_domain
                )
            )
            axes[i].set_xlabel("FPR")
            axes[i].set_ylabel("TPR")
            axes[i].legend(loc="lower right")

            # save to list
            auc_test.append(auc)
            pauc_test.append(pauc)

        # calculate hmean total
        hmean_total = hmean(auc_test + pauc_test)

        # suplite of the fig to report the hmean
        fig.suptitle("Hmean {:.4f}".format(hmean_total))

        return fig, hmean_total


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

    # knn_name = ad.name_knn
    # print("knn_name:", knn_name)

    """
    test model anomaly detection
    """
    project = "DCASE2024/wav-test"
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
    k_smote = 5
    batch_size = 32  # 8
    num_instances = 320000
    learning_rate = 0.0001
    step_warmup = 120  # 480
    step_accumulation = 8  # 32
    k_neighbors = 2
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
        k_neighbors=k_neighbors,
        emb_size=emb_size,
    )
