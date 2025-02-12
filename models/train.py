import torch
from torch import nn
from beats.beats_custom import BEATsCustom
import sys
import os
import numpy as np
from loss import AdaCosLoss, ArcFaceLoss
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import neptune
from neptune.utils import stringify_unsupported
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hmean
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingWarmRestarts,
)
import optuna
from optuna.trial import TrialState
import random
from preparation import ModelDataPrepraration


# class Anomaly Detechtion
class AnomalyDetection(ModelDataPrepraration):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

    def anomaly_detection(
        self,
        project="DCASE2024/wav-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ==",
        k_smote: int = 5,
        batch_size: int = 8,
        num_iterations: int = 100,
        lora: bool = False,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        emb_size: int = None,
        loss_type: str = "adacos",
        margin: int = None,
        scale: int = None,
        learning_rate: float = 1e-5,
        scheduler_type: str = "linear_restarts",
        step_warmup: int = 8,
        min_lr: float = None,
        step_accumulation: int = 8,
        k_neighbors: int = 2,
        HPO: bool = False,
        trial: optuna.trial.Trial = None,
        index_split=None,
        num_train_machines: int = 5,
        num_splits: int = 5,
        list_machines=None,
    ):
        """
        main function to find the result
        """

        if self.vram > 40:
            batch_size = 64
        elif self.vram > 11:
            batch_size = 32
        print("List machine", list_machines)

        # init neptune
        run = neptune.init_run(project=project, api_token=api_token)

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
            print("sort data test")
            dataset_test_attribute = self.sort_data_machines(
                dataset=dataset_test_attribute, list_machines=list_machines
            )
            name_trial = f"trial {trial.number} split {index_split}"
            run["name_trial"] = name_trial

        num_classes = self.num_classes_attribute()

        # dataloader
        num_instances = batch_size * num_iterations
        dataloader_smote_uniform = self.data_loader(
            dataset=dataset_smote,
            batch_size=batch_size,
            num_iterations=num_iterations,
            uniform_sampling=True,
        )
        dataloader_smote_attribute = self.data_loader(
            dataset=dataset_smote, batch_size=batch_size
        )
        dataloader_train_attribute = self.data_loader(
            dataset=dataset_train_attribute, batch_size=batch_size
        )
        dataloader_test_attribute = self.data_loader(
            dataset=dataset_test_attribute, batch_size=batch_size
        )

        # load model
        input_size = dataloader_train_attribute.dataset.tensors[0].shape[1] // self.fs
        model = self.load_model(
            input_size=input_size,
            emb_size=emb_size,
            lora=lora,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # model to device
        if self.n_gpus > 1:
            model = nn.DataParallel(model, device_ids=list(range(self.n_gpus)), dim=0)
        model = model.to(self.device)
        num_params = sum(p.numel() for p in model.parameters())
        num_params_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # loss
        if emb_size == None:
            emb_size = model.embedding_asp
        loss = self.load_loss(
            loss_type=loss_type,
            num_classes=num_classes,
            emb_size=emb_size,
            margin=margin,
            scale=scale,
        )

        # optimizer
        parameters = list(model.parameters()) + list(loss.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)

        # scheduler
        scheduler = self.load_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            step_warmup=step_warmup,
            min_lr=min_lr,
        )

        # save the hyperparameters and configuration
        name_saved_model = self.name_saved_model()
        hyperparameters = self.hyperparameters_configuration_dict(
            k_smote=k_smote,
            lora=lora,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            num_iterations=num_iterations,
            num_instances=num_instances,
            loss_type=loss_type,
            learning_rate=learning_rate,
            step_warmup=step_warmup,
            step_accumulation=step_accumulation,
            k_neighbors=k_neighbors,
            HPO=HPO,
            emb_size=emb_size,
            margin=margin,
            scale=scale,
            name_saved_model=name_saved_model,
            input_size=input_size,
            trial=trial,
            trial_number=None if trial == None else trial.number,
            index_split=index_split,
            num_train_machines=num_train_machines,
            num_splits=num_splits,
            list_machines=list_machines,
        )
        run["hyperparameters"] = hyperparameters

        configuration = self.hyperparameters_configuration_dict(
            seed=self.seed,
            num_params=num_params,
            num_params_trainable=num_params_trainable,
            n_gpus=self.n_gpus,
            device=stringify_unsupported(self.device),
            vram=self.vram,
        )
        run["configuration"] = configuration

        # training attribute classification
        output_training_loop = self.training_loop(
            run=run,
            dataloader_smote_uniform=dataloader_smote_uniform,
            dataloader_smote_attribute=dataloader_smote_attribute,
            dataloader_train_attribute=dataloader_train_attribute,
            dataloader_test_attribute=dataloader_test_attribute,
            hyperparameters=hyperparameters,
            model=model,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # stop the run
        run.stop()

        # save the pretrained mode, loss, optimizer and hyperparameters
        if not HPO:
            (
                model_pretrained,
                loss_pretrained,
                optimizer_pretrained,
                knn_pretrained,
                scaler_pretrained,
            ) = output_training_loop

            self.save_pretrained_model_loss(
                model_pretrained=model_pretrained,
                loss_pretrained=loss_pretrained,
                optimizer=optimizer_pretrained,
                knn_pretrained=knn_pretrained,
                scaler_pretrained=scaler_pretrained,
                hyperparameters=hyperparameters,
            )

        else:
            hmean_split = output_training_loop
            return hmean_split

    def training_loop(
        self,
        run: neptune.init_run,
        dataloader_smote_uniform: DataLoader,
        dataloader_smote_attribute: DataLoader,
        dataloader_train_attribute: DataLoader,
        dataloader_test_attribute: DataLoader,
        hyperparameters: dict,
        model: nn.Module,
        loss: AdaCosLoss,
        optimizer: torch.optim.AdamW,
        scheduler: LambdaLR,
    ):
        """
        training loop for smote data
        """

        # get the hyperparameters
        step_accumulation = hyperparameters["step_accumulation"]
        batch_size = hyperparameters["batch_size"]
        emb_size = hyperparameters["emb_size"]
        k_neighbors = hyperparameters["k_neighbors"]
        num_iterations = hyperparameters["num_iterations"]
        list_machines = hyperparameters["list_machines"]
        HPO = hyperparameters["HPO"]
        if HPO:
            trial = hyperparameters["trial"]
            index_split = hyperparameters["index_split"]

        # step report and evaluation
        step_eval = 20
        step_lr = 0
        step_hpo = 0

        # loss train
        loss_smote_uniform_total = 0

        # accuracy as objective function for HPO
        if HPO:
            y_pred_smote_uniform_array = np.empty(
                batch_size * step_accumulation,
            )
            y_true_smote_uniform_array = np.empty(
                batch_size * step_accumulation,
            )

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

            # save to array if HPO
            if HPO:
                # iter for saved array y_true y_pred
                iter_smote_uniform_accumulated = iter_smote_uniform % step_accumulation
                y_true_smote_uniform_array[
                    iter_smote_uniform_accumulated
                    * batch_size : iter_smote_uniform_accumulated
                    * batch_size
                    + batch_size
                ] = y_smote_uniform.cpu().numpy()

                y_pred_smote_uniform = loss.pred_labels(
                    embedding=embedding_smote_uniform, y_true=y_smote_uniform
                )
                y_pred_smote_uniform_array[
                    iter_smote_uniform_accumulated
                    * batch_size : iter_smote_uniform_accumulated
                    * batch_size
                    + batch_size
                ] = y_pred_smote_uniform.cpu().numpy()

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

                # pruned for hpo
                if HPO:
                    # accuracy smote as ojective function
                    accuracy_smote_uniform = accuracy_score(
                        y_true=y_true_smote_uniform_array,
                        y_pred=y_pred_smote_uniform_array,
                    )
                    run["smote_uniform/accuracy_smote_uniform"].append(
                        accuracy_smote_uniform, step=iter_smote_uniform
                    )

                # reset loss_smote_uniform_total
                loss_smote_uniform_total = 0

                # log the learning rate
                current_lr = optimizer.param_groups[0]["lr"]
                run["smote_uniform/current_lr"].append(current_lr, step=step_lr)
                step_lr = step_lr + 1

                # update scheduler
                scheduler.step()

            # report the loss and evaluation mode every 1000 iteration
            if (
                iter_smote_uniform + 1
            ) % step_eval == 0 or iter_smote_uniform + 1 == num_iterations:

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
                if not HPO:
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

                # evaluation mode for data smote attribute
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

                # use knn to get the decision, anomaly score and knn_pretrained
                decision_anomaly_score_test, knn_train, scaler = self.decision_knn_1(
                    k_neighbors=k_neighbors,
                    embedding_train_array=embedding_smote_array,
                    embedding_test_array=embedding_test_array,
                    y_pred_train_array=y_pred_label_smote_array,
                    y_pred_test_array=y_pred_label_test_array,
                    y_true_test_array=y_true_test_array,
                )

                # accuracy decision
                if not HPO:
                    accuracy_decisions = self.accuracy_decision(
                        decision_anomaly_score_test=decision_anomaly_score_test
                    )
                    for typ_l, acc in zip(type_labels_test, accuracy_decisions):
                        run["{}/accuracy_{}".format("decision", typ_l)].append(
                            acc, step=iter_smote_uniform
                        )

                # anomaly score and hmean
                hmean_img, hmean_total = self.auc_pauc_hmean(
                    decision_anomaly_score_test=decision_anomaly_score_test,
                    list_machines=list_machines,
                    y_true_test_array=y_true_test_array,
                )

                run["score/hmean"].append(hmean_total, step=iter_smote_uniform)
                run["score/auc_pauc_hmean"].append(hmean_img, step=iter_smote_uniform)
                plt.close()

                # check pruning for HPO
                if HPO:
                    trial.report(
                        hmean_total, step=index_split * num_iterations + step_hpo
                    )
                    if trial.should_prune() or np.isnan(loss_smote_uniform_total):
                        raise optuna.exceptions.TrialPruned()
                    step_hpo = step_hpo + 1

        if HPO:
            return hmean_total
        else:
            # knn_train = None
            # scaler = None
            return model, loss, optimizer, knn_train, scaler

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

        # evaluation mode
        with torch.no_grad():
            for iter_eval, (X, y) in enumerate(dataloader_attribute):

                # data to device
                X = X.to(self.device)

                # forward pass
                embedding = model(X)

                # pred the label
                y_true_label = (
                    y.clone()[:, 1] if type_data in ["train", "test"] else y.clone()
                )
                y_true_label = y_true_label.to(self.device)
                y_pred_label = loss.pred_labels(
                    embedding=embedding, y_true=y_true_label
                )

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

    def cross_validation(
        self,
        project="DCASE2024/wav-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ==",
        k_smote: int = 5,
        batch_size: int = 8,
        num_iterations: int = 100,
        lora: bool = False,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        emb_size: int = None,
        loss_type: str = "adacos",
        margin: int = None,
        scale: int = None,
        learning_rate: float = 1e-5,
        scheduler_type: str = "linear_restarts",
        step_warmup: int = 8,
        min_lr: float = None,
        step_accumulation: int = 8,
        k_neighbors: int = 2,
        HPO: bool = False,
        trial: optuna.trial.Trial = None,
        num_train_machines: int = 5,
        num_splits: int = 5,
    ):

        # get the combinations of machines (list of the machines)
        list_machines_combination = self.sample_machines(
            num_train_machines=num_train_machines, num_splits=num_splits
        )

        # cross validation
        hmean_list_hpo = []
        for index_split, list_machines in enumerate(list_machines_combination):

            # calculate the hmean split
            hmean_split = self.anomaly_detection(
                project=project,
                api_token=api_token,
                k_smote=k_smote,
                batch_size=batch_size,
                num_iterations=num_iterations,
                lora=lora,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                emb_size=emb_size,
                loss_type=loss_type,
                margin=margin,
                scale=scale,
                learning_rate=learning_rate,
                scheduler_type=scheduler_type,
                step_warmup=step_warmup,
                min_lr=min_lr,
                step_accumulation=step_accumulation,
                k_neighbors=k_neighbors,
                HPO=HPO,
                trial=trial,
                index_split=index_split,
                num_train_machines=num_train_machines,
                num_splits=num_splits,
                list_machines=list_machines,
            )
            hmean_list_hpo.append(hmean_split)
        print("hmean_list_hpo:", hmean_list_hpo)

        return np.mean(hmean_list_hpo)

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

    def decision_knn_1(
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
        print("distance_train shape:", distance_train.shape)

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
        print("distance_test:", distance_test)
        decision_anomaly_score_test[:, 2] = distance_test

        # get the decision
        decisions = [0 if d < threshold else 1 for d in distance_test]
        decision_anomaly_score_test[:, 1] = decisions

        # decision_anomaly_score_test = np.array(decision_anomaly_score_test)
        # print("decision_anomaly_score_test:", decision_anomaly_score_test)
        for i in decision_anomaly_score_test:
            print(i)

        return decision_anomaly_score_test, knn, scaler

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

            # print("label", label)
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
            scaler = StandardScaler()
            scaler.fit(distance_train.reshape(-1, 1))
            distance_train = scaler.transform(distance_train.reshape(-1, 1)).reshape(
                len(distance_train),
            )
            # print("distance_train_normalize", distance_train)
            # print("distance_train_normalize max", max(distance_train))
            # print("distance_train_normalize min", min(distance_train))

            # calculate the threshold using percentile
            threshold = 3
            # print("threshold:", threshold)

            # save to list
            knn_train.append(knn)
            threshold_train.append(threshold)
            scaler_train.append(scaler)

            # print()

        decision_anomaly_score_test = []

        # loop through each id in test data
        for id in self.id_timeseries_analysis(keys="test"):

            # find the index of each id and their predict and embedding
            index_id_test = np.where(y_true_test_array[:, 0] == id)[0]
            # print("id", id)
            # print("index_id_test:", index_id_test)
            embedding_test_fit_knn = embedding_test_array[index_id_test]
            label_pred = int(y_pred_test_array[index_id_test][0])
            # print("label_pred:", label_pred)

            # use knn, scaler and threshold from label pred
            knn = knn_train[label_pred]
            scaler = scaler_train[label_pred]
            threshold = threshold_train[label_pred]
            # print("threshold:", threshold)
            # threshold = 1

            # find the distance test of embedding test with correspond pred label
            distance_test, _ = knn.kneighbors(embedding_test_fit_knn)
            # print("distance_test:", distance_test)

            # elminate the max value in each row, only consider the distance to k_neighbors-1 neighbor (same as in training)
            max_indices_distance_test = np.argmax(distance_test, axis=1)
            distance_test = np.array(
                [
                    np.delete(row, max_idx)
                    for row, max_idx in zip(distance_test, max_indices_distance_test)
                ]
            )
            # print("distance_test_delete:", distance_test)
            distance_test = np.mean(distance_test, axis=1)
            # print("distance_test_mean:", distance_test)

            # normalize as anomaly score and compare with the threshold the make the decision
            distance_test = scaler.transform(distance_test.reshape(-1, 1)).reshape(
                len(distance_test),
            )
            distance_test = distance_test[0]
            # print("distance_test:", distance_test)
            decision = 0 if distance_test < threshold else 1
            # print("decision:", decision)

            # append to list
            decision_anomaly_score_test.append([id, decision, distance_test])
            # print(
            #     "[id, decision, distance_test, threshold]",
            #     [id, decision, distance_test, threshold],
            # )
            # print()

        print("decision_anomaly_score_test:", decision_anomaly_score_test)
        print()

        # convert decision anomaly score test to array
        decision_anomaly_score_test = np.array(decision_anomaly_score_test)

        return decision_anomaly_score_test, knn_train, scaler_train

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


# run this script
if __name__ == "__main__":

    #  seed and data_name
    seed = 1998
    develop_name = "develop"

    # load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # anomaly detection
    ad = AnomalyDetection(data_name=develop_name, seed=seed)

    """
    test model anomaly detection
    """
    project = "DCASE2024/dcase-HPO"
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
    k_smote: int = 5
    batch_size: int = 64
    num_iterations: int = 1000
    lora: bool = False
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    emb_size: int = None
    loss_type: str = "adacos"
    margin: int = None
    scale: int = None
    learning_rate: float = 1e-5
    scheduler_type: str = "linear_restarts"
    step_warmup: int = 8
    min_lr: float = None
    step_accumulation: int = 8
    k_neighbors: int = 2
    trial: optuna.trial.Trial = None
    index_split = None
    num_train_machines: int = 5
    num_splits: int = 5
    list_machines = None
    HPO: bool = True

    # hyperparameters optimization
    if HPO:
        # create directory for HPO
        directory_hpo = "HPO"
        path_directory_models = ad.path_directory_models
        path_directory_HPO = os.path.join(path_directory_models, directory_hpo)
        os.makedirs(path_directory_HPO, exist_ok=True)

        # data base file for hpo
        db_hpo = "hpo.db"
        path_db_hpo = os.path.join(path_directory_HPO, db_hpo)
        db_hpo_sqlite = "sqlite:///{}".format(path_db_hpo)

        # csv file for hpo
        csv_hpo = "hpo.csv"
        path_csv_hpo = os.path.join(path_directory_HPO, csv_hpo)

        # optuna configuration
        sampler = optuna.samplers.TPESampler(seed=ad.seed)
        pruner = optuna.pruners.MedianPruner()
        study_name = "dcase24_hpo"
        n_trials_total = 100

        # fix hyperparameters
        project = "DCASE2024/dcase-HPO"
        api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
        k_smote = 5
        batch_size = 64
        num_train_machines = 5
        num_splits = 5

        # objective functions
        def objective(trial: optuna.trial.Trial):

            # tuned hyperparamters
            num_iterations = trial.suggest_int(
                name="num_iterations",
                low=500,
                high=10000,
                step=100,
            )

            learning_rate = trial.suggest_float(
                name="learning_rate", low=1e-7, high=1e-1, log=True
            )

            step_warmup = trial.suggest_int(
                name="step_warmup", low=2, high=500, step=10
            )

            emb_size = trial.suggest_int(name="emb_size", low=2, high=2048, step=2)

            loss_type = trial.suggest_categorical(
                name="loss_type", choices=["adacos", "arcface"]
            )
            margin = trial.suggest_float(name="margin", low=0, high=5, step=0.1)
            scale = trial.suggest_int(name="scale", low=2, high=256, step=2)

            scheduler_type = trial.suggest_categorical(
                name="scheduler_type", choices=["cosine_restarts", "linear_restarts"]
            )
            min_lr = trial.suggest_float(name="min_lr", low=1e-7, high=1e-1, log=True)

            lora = trial.suggest_categorical(name="lora", choices=[True, False])
            r = trial.suggest_int(name="r", low=8, high=256, step=2)
            lora_alpha = trial.suggest_int(name="lora_alpha", low=2, high=128, step=2)
            lora_dropout = trial.suggest_float(
                name="lora_dropout", low=0.1, high=1, step=0.1
            )

            step_accumulation = trial.suggest_int(
                name="step_accumulation", low=2, high=128, step=2
            )

            k_neighbors = trial.suggest_int(name="k_neighbors", low=2, high=128, step=1)

            # mean for cv
            hmean_cv = ad.cross_validation(
                project=project,
                api_token=api_token,
                k_smote=k_smote,
                batch_size=batch_size,
                num_iterations=num_iterations,
                lora=lora,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                emb_size=emb_size,
                loss_type=loss_type,
                margin=margin,
                scale=scale,
                learning_rate=learning_rate,
                scheduler_type=scheduler_type,
                step_warmup=step_warmup,
                min_lr=min_lr,
                step_accumulation=step_accumulation,
                k_neighbors=k_neighbors,
                HPO=HPO,
                trial=trial,
                num_train_machines=num_train_machines,
                num_splits=num_splits,
            )

            return hmean_cv

        # create or load a exist study
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=db_hpo_sqlite,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

        # Export the study trials to a pandas DataFrame
        df_trial = study.trials_dataframe()

        # Save the DataFrame to a CSV file
        df_trial.to_csv(path_csv_hpo, index=False)

        # run trial if not enough n_trials_total
        if len(study.trials) < n_trials_total:

            # reload traila if in fail or running state
            if len(study.trials) >= 1 and study.trials[-1].state in [
                TrialState.FAIL,
                TrialState.RUNNING,
            ]:
                failed_trial_params = study.trials[-1].params
                study.enqueue_trial(failed_trial_params)

            # perform HPO
            study.optimize(objective, n_trials=n_trials_total)

        else:
            pruned_trials = study.get_trials(
                deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
            )
            complete_trials = study.get_trials(
                deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
            )
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            best_trial_score = study.best_trial
            best_trial_params = best_trial_score.params

            print("  Value: ", best_trial_score.value)

            print("  Params: ")
            for key, value in best_trial_params.items():
                print("    {}: {}".format(key, value))

            # apply the best hyperparamters for final test
            ad.anomaly_detection(
                project=project,
                api_token=api_token,
                k_smote=k_smote,
                batch_size=batch_size,
                num_train_machines=num_train_machines,
                num_splits=num_splits,
                **best_trial_params,
            )

    else:
        ad.anomaly_detection(
            project=project,
            api_token=api_token,
            k_smote=k_smote,
            lora=lora,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            num_iterations=num_iterations,
            loss_type=loss_type,
            learning_rate=learning_rate,
            step_warmup=step_warmup,
            step_accumulation=step_accumulation,
            k_neighbors=k_neighbors,
            scheduler_type=scheduler_type,
            min_lr=min_lr,
            margin=margin,
            scale=scale,
            emb_size=emb_size,
        )
