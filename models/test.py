import torch
from torch import nn
from beats.beats_custom import BEATsCustom
import os
import numpy as np
from loss import AdaCosLoss
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import neptune
from neptune.utils import stringify_unsupported
from torch.optim.lr_scheduler import (
    LambdaLR,
)
import multiprocessing
import optuna
from optuna.trial import TrialState
from preparation import ModelDataPrepraration
import matplotlib.pyplot as plt
import argparse


# class Anomaly Detechtion
class AnomalyDetection(ModelDataPrepraration):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

    def preparation_before_training(
        self,
        k_smote: int = 5,
        batch_size: int = 8,
        epochs: int = 50,
        len_factor: float = 0.1,
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
        list_machines: list = None,
        k_neighbors: int = 2,
        index_split=None,
        HPO: bool = False,
        trial: optuna.trial.Trial = None,
        num_train_machines: int = 5,
        num_splits: int = 5,
        dict_data_shared: dict = None,
    ):
        """
        perform all steps are needed before training, load model, load loss, optimizer, hyperapameters dict
        """
        # batchsize based on vram
        if self.vram < 11:
            batch_size = 12
        elif 11 <= self.vram < 25:
            batch_size = 25
        else:
            batch_size = 64

        # load data
        (
            dataloader_smote_attribute,
            dataloader_train_attribute,
            dataloader_test_attribute,
            num_instances_smote,
        ) = self.perform_load_data(
            k_smote=k_smote,
            batch_size=batch_size,
            len_factor=len_factor,
            HPO=HPO,
            list_machines=list_machines,
        )

        # load model
        input_size = dataloader_train_attribute.dataset.tensors[0].shape[1] // self.fs
        model, num_params, num_params_trainable = self.perform_load_model(
            input_size=input_size,
            emb_size=emb_size,
            lora=lora,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        if emb_size == None:
            emb_size = model.embedding_asp

        # load loss
        loss = self.perform_load_loss(
            loss_type=loss_type, emb_size=emb_size, margin=margin, scale=scale
        )

        # load optimizer
        optimizer = self.perform_load_optimizer(
            model=model, loss=loss, learning_rate=learning_rate
        )

        # scheduler
        scheduler = self.load_scheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_type,
            step_warmup=step_warmup,
            min_lr=min_lr,
        )

        # save the hyperparameters and configuration
        name_saved_model = self.name_saved_model(index_split=index_split)

        # hyperparameters dict and and configuration dict
        hyperparameters = self.hyperparameters_configuration_dict(
            k_smote=k_smote,
            lora=lora,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            batch_size=batch_size,
            epochs=epochs,
            len_factor=len_factor,
            list_machines=list_machines,
            num_instances_smote=num_instances_smote,
            loss_type=loss_type,
            learning_rate=learning_rate,
            step_warmup=step_warmup,
            min_lr=min_lr,
            k_neighbors=k_neighbors,
            HPO=HPO,
            emb_size=emb_size,
            margin=margin,
            scale=scale,
            scheduler_type=scheduler_type,
            name_saved_model=name_saved_model,
            input_size=input_size,
            num_classes_train=self.num_classes_attribute(),
            trial=trial,
            trial_number=None if trial == None else trial.number,
            index_split=index_split,
            num_train_machines=num_train_machines,
            num_splits=num_splits,
        )

        configuration = self.hyperparameters_configuration_dict(
            seed=self.seed,
            num_params=num_params,
            num_params_trainable=num_params_trainable,
            n_gpus=self.n_gpus,
            device=stringify_unsupported(self.device),
            vram=self.vram,
        )

        # shared dictionary
        if HPO and dict_data_shared is not None:
            # init run
            run = neptune.init_run(project=project, api_token=api_token)
            run_id = run["sys/id"].fetch()
            print("run_id:", run_id)

            dict_data_shared[index_split] = {
                "smote": dataloader_smote_attribute,
                "train": dataloader_train_attribute,
                "test": dataloader_test_attribute,
            }
            dict_data_shared[f"id_run_{index_split}"] = run_id

            run.stop()

            # save model, loss and hyperparameters
            self.save_pretrained_model_loss(
                model_pretrained=model,
                loss_pretrained=loss,
                optimizer=optimizer,
                scheduler=scheduler,
                hyperparameters=hyperparameters,
                configuration=configuration,
            )

        # return for not HPO
        return (
            dataloader_smote_attribute,
            dataloader_train_attribute,
            dataloader_test_attribute,
            model,
            loss,
            optimizer,
            scheduler,
            hyperparameters,
            configuration,
        )

    def anomaly_detection(
        self,
        project="DCASE2024/wav-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ==",
        k_smote: int = 5,
        batch_size: int = 8,
        len_factor: float = 0.1,
        epochs: int = 100,
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
        k_neighbors: int = 2,
        HPO: bool = False,
        trial: optuna.trial.Trial = None,
        index_split=None,
        num_train_machines: int = 5,
        num_splits: int = 5,
        list_machines: list = None,
        dict_shared: dict = None,
    ):
        """
        main function to find the result
        """
        # init neptune run
        run = neptune.init_run(project=project, api_token=api_token)

        # load data, loss, optimizer and model
        (
            dataloader_smote_attribute,
            dataloader_train_attribute,
            dataloader_test_attribute,
            model,
            loss,
            optimizer,
            scheduler,
            hyperparameters,
            configuration,
        ) = self.preparation_before_training(
            k_smote=k_smote,
            batch_size=batch_size,
            epochs=epochs,
            len_factor=len_factor,
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
            list_machines=list_machines,
            k_neighbors=k_neighbors,
            index_split=index_split,
            HPO=HPO,
            trial=trial,
            num_train_machines=num_train_machines,
            num_splits=num_splits,
            dict_data_shared=dict_shared,
        )
        run["hyperparameters"] = hyperparameters
        run["configuration"] = configuration

        # training attribute classification
        output_training_loop = self.training_loop(
            run=run,
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
        k_neighbors = hyperparameters["k_neighbors"]
        epochs = hyperparameters["epochs"]
        list_machines = hyperparameters["list_machines"]
        HPO = hyperparameters["HPO"]

        # trials for hpo
        if HPO:
            trial = hyperparameters["trial"]
            index_split = hyperparameters["index_split"]

        # type_labels
        type_labels_train = ["train_source_normal", "train_target_normal"]
        type_labels_test = [
            "test_source_normal",
            "test_target_normal",
            "test_source_anomaly",
            "test_target_anomaly",
        ]
        type_labels_smote_knn = ["smote_attribute"]

        for ep in range(epochs):

            # training mode for data smote
            (
                accuracy_smote_dict,
                embedding_smote_array,
                y_true_smote_array,
                y_pred_label_smote_array,
                loss_smote_attribute_total,
            ) = self.iteration_loop(
                run=run,
                ep=ep,
                model=model,
                loss=loss,
                scheduler=scheduler,
                dataloader_attribute=dataloader_smote_attribute,
                optimizer=optimizer,
                hyperparameters=hyperparameters,
                type_labels=type_labels_smote_knn,
            )

            # evaluation mode for train data attribute
            if not HPO:
                (
                    accuracy_train_dict,
                    embedding_train_array,
                    y_true_train_array,
                    y_pred_label_train_array,
                    _,
                ) = self.iteration_loop(
                    run=run,
                    ep=ep,
                    model=model,
                    loss=loss,
                    scheduler=scheduler,
                    dataloader_attribute=dataloader_train_attribute,
                    optimizer=optimizer,
                    hyperparameters=hyperparameters,
                    type_labels=type_labels_train,
                )

            # evaluation mode for test data attribute
            (
                accuracy_test_dict,
                embedding_test_array,
                y_true_test_array,
                y_pred_label_test_array,
                _,
            ) = self.iteration_loop(
                run=run,
                ep=ep,
                model=model,
                loss=loss,
                scheduler=scheduler,
                dataloader_attribute=dataloader_test_attribute,
                optimizer=optimizer,
                hyperparameters=hyperparameters,
                type_labels=type_labels_test,
            )

            # use knn to get the decision, anomaly score and knn_pretrained
            decision_anomaly_score_test, knn_train, scaler = self.decision_knn(
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
                    run["{}/accuracy_{}".format("decision", typ_l)].append(acc, step=ep)

            # anomaly score and hmean
            hmean_img, hmean_total = self.auc_pauc_hmean(
                decision_anomaly_score_test=decision_anomaly_score_test,
                list_machines=list_machines,
                y_true_test_array=y_true_test_array,
            )

            # log the hmean image
            run["score/hmean"].append(hmean_total, step=ep)
            run["score/auc_pauc_hmean"].append(hmean_img, step=ep)
            plt.close()

            # check pruning for HPO
            if HPO:
                trial.report(hmean_total, step=index_split * epochs + ep)
                if trial.should_prune() or np.isnan(loss_smote_attribute_total):
                    run.stop()
                    raise optuna.exceptions.TrialPruned()

        if HPO:
            return hmean_total
        else:
            # knn_train = None
            # scaler = None
            return model, loss, optimizer, knn_train, scaler

    def iteration_loop(
        self,
        run: neptune.init_run,
        ep: int,
        model: nn.Module,
        loss: AdaCosLoss,
        scheduler: torch.optim.lr_scheduler,
        optimizer: torch.optim.AdamW,
        hyperparameters: dict,
        type_labels=["train_source_normal", "train_target_normal"],
    ):
        """
        evaluation mode in training loop
        """

        # hyperparameters
        emb_size = hyperparameters["emb_size"]
        batch_size = hyperparameters["batch_size"]
        num_instances_smote = hyperparameters["num_instances_smote"]
        list_machines = hyperparameters["list_machines"]

        # y_true, y_pred, embedding array
        type_data = type_labels[0].split("_")[0]
        len_dataset = (
            dataloader_attribute.dataset.tensors[0].shape[0]
            if type_data in ["train", "test"]
            else num_instances_smote
        )

        y_pred_label_array = np.empty(shape=(len_dataset,))

        # evaluation mode or no grad mode
        if type_data in ["train", "test"]:
            # mode for evaluation
            mode = torch.no_grad()
            model.eval()
            loss.eval()
            y_true_array = np.empty(shape=(len_dataset, 3))

        else:
            # mode for training for smote data
            mode = torch.enable_grad()
            model.train()
            loss.train()
            y_true_array = np.empty(shape=(len_dataset,))

            # loss train smote
            loss_smote_attribute_total = 0

        embedding_array = np.empty(shape=(len_dataset, emb_size))

        # evaluation mode
        with mode:
            for iter_mode, (X, y) in enumerate(dataloader_attribute):

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
                    iter_mode * batch_size : iter_mode * batch_size + len(X)
                ] = (embedding.detach().cpu().numpy())
                y_true_array[
                    iter_mode * batch_size : iter_mode * batch_size + len(X)
                ] = y.cpu().numpy()
                y_pred_label_array[
                    iter_mode * batch_size : iter_mode * batch_size + len(X)
                ] = y_pred_label.cpu().numpy()

                # calculate loss for training mode for smote data
                if type_data == "smote":
                    loss_smote_attribute = loss(embedding, y_true_label)
                    loss_smote_attribute_total = (
                        loss_smote_attribute_total + loss_smote_attribute.item()
                    )

                    # calculate gradient
                    loss_smote_attribute.backward()

                    # update weights with optimizer
                    optimizer.step()
                    optimizer.zero_grad()

                    # update scheduler
                    scheduler.step()

                    # log the learning rate
                    current_lr = optimizer.param_groups[0]["lr"]
                    run["smote_uniform/current_lr"].append(current_lr, step=ep)

            # calculate accuracy, confusion matrix for train and test data
            if type_data in ["train", "test"]:
                accuracy_type_labels = self.accuracy_attribute(
                    y_true_array=y_true_array,
                    y_pred_label_array=y_pred_label_array,
                    type_labels=type_labels,
                )
                cm = confusion_matrix(
                    y_true=y_true_array[:, 1], y_pred=y_pred_label_array
                )
                # for ignoring error
                loss_smote_attribute_total = None

            # calculate loss smote total, accuracy, confusion matrix for data smote
            else:
                accuracy_type_labels = [
                    accuracy_score(y_true=y_true_array, y_pred=y_pred_label_array)
                ]

                cm = confusion_matrix(y_true=y_true_array, y_pred=y_pred_label_array)

                loss_smote_attribute_total = loss_smote_attribute_total / len(
                    dataloader_attribute
                )
                run["{}/loss_smote_attribute".format(type_data)].append(
                    loss_smote_attribute_total, step=ep
                )

            cm_img = self.plot_confusion_matrix(cm=cm, type_data=type_data)

            # save metrics in run
            accuracy_dict = {}
            for typ_l, acc in zip(type_labels, accuracy_type_labels):
                run["{}/accuracy_{}".format(type_data, typ_l)].append(acc, step=ep)
                accuracy_dict[typ_l] = acc

            run["{}/confusion_matrix".format(type_data)].append(cm_img, step=ep)
            plt.close()

        return (
            accuracy_dict,
            embedding_array,
            y_true_array,
            y_pred_label_array,
            loss_smote_attribute_total,
        )

    def hmean_calculation_one_epoch_hpo(
        self,
        project: str,
        api_token: str,
        number_trial: int,
        ep: int,
        index_split: int,
        list_machines: list,
        k_neighbors: int,
        dict_data_shared: dict = None,
        list_shared_hmean: list = None,
    ):
        """
        calcualate hmean for one epoch
        """
        # reinit run
        print("rund_id hmean calculation")
        id_run = dict_data_shared[f"id_run_{index_split}"]
        run = neptune.init_run(project=project, api_token=api_token, with_id=id_run)

        # load data for this split
        dataloader_smote_attribute = dict_data_shared[index_split]["smote"]
        dataloader_test_attribute = dict_data_shared[index_split]["test"]

        # load model, loss, optimizer, scheduler, hyperparameters for this split
        pretrained_path = os.path.join(
            self.path_hpo_directory, self.name_saved_model(index_split=index_split)
        )
        model, loss, optimizer, scheduler, hyperparameters, configuration, _, _ = (
            self.load_pretrained_model(pretrained_path=pretrained_path)
        )

        # log the hyperaparameters, configuration and name trial
        if ep == 0:
            name_trial = f"trial {number_trial} split {index_split}"
            run["name_trial"] = name_trial
            run["hyperparameters"] = hyperparameters
            run["configuration"] = configuration

        # type_labels for iteration loop
        type_labels_test = [
            "test_source_normal",
            "test_target_normal",
            "test_source_anomaly",
            "test_target_anomaly",
        ]
        type_labels_smote_knn = ["smote_attribute"]

        # training mode for data smote
        (
            accuracy_smote_dict,
            embedding_smote_array,
            y_true_smote_array,
            y_pred_label_smote_array,
            loss_smote_attribute_total,
        ) = self.iteration_loop(
            run=run,
            ep=ep,
            model=model,
            loss=loss,
            scheduler=scheduler,
            dataloader_attribute=dataloader_smote_attribute,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
            type_labels=type_labels_smote_knn,
        )

        # evaluation mode for test data attribute
        (
            accuracy_test_dict,
            embedding_test_array,
            y_true_test_array,
            y_pred_label_test_array,
            _,
        ) = self.iteration_loop(
            run=run,
            ep=ep,
            model=model,
            loss=loss,
            scheduler=scheduler,
            dataloader_attribute=dataloader_test_attribute,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
            type_labels=type_labels_test,
        )

        # use knn to get the decision, anomaly score and knn_pretrained
        decision_anomaly_score_test, knn_train, scaler = self.decision_knn(
            k_neighbors=k_neighbors,
            embedding_train_array=embedding_smote_array,
            embedding_test_array=embedding_test_array,
            y_pred_train_array=y_pred_label_smote_array,
            y_pred_test_array=y_pred_label_test_array,
            y_true_test_array=y_true_test_array,
        )

        # update scheduler
        scheduler.step()

        # log the learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        run["smote_uniform/current_lr"].append(current_lr, step=ep)

        # anomaly score and hmean
        hmean_img, hmean_total = self.auc_pauc_hmean(
            decision_anomaly_score_test=decision_anomaly_score_test,
            list_machines=list_machines,
            y_true_test_array=y_true_test_array,
        )

        # log the hmean image
        run["score/hmean"].append(hmean_total, step=ep)
        run["score/auc_pauc_hmean"].append(hmean_img, step=ep)
        plt.close()

        # stop the run
        run.stop()

        # Save updated model and optimizer
        self.save_pretrained_model_loss(
            model_pretrained=model,
            loss_pretrained=loss,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
            configuration=configuration,
            scheduler=scheduler,
            knn_pretrained=knn_train,
            scaler_pretrained=scaler,
        )

        # save the hmean total to shared list
        list_shared_hmean.append(hmean_total)

    def run_hmean_calculation_one_epoch_hpo(
        self,
        list_machines_combinations: list,
        project: str,
        api_token: str,
        number_trial: int,
        ep: int,
        k_neighbors: int,
        dict_data_shared: dict = None,
    ):
        """
        run the function hmean_calculation_one_epoch_hpo for all splits in parallel using multiprocessing
        """
        # create the process
        processes = []
        manager = multiprocessing.Manager()
        list_shared_hmean = manager.list()

        for index_split, list_machines in enumerate(list_machines_combinations):

            p = multiprocessing.Process(
                target=self.hmean_calculation_one_epoch_hpo,
                args=(
                    project,
                    api_token,
                    number_trial,
                    ep,
                    index_split,
                    list_machines,
                    k_neighbors,
                    dict_data_shared,
                    list_shared_hmean,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        return list(list_shared_hmean)

    def cross_validation(
        self,
        project="DCASE2024/wav-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ==",
        k_smote: int = 5,
        batch_size: int = 8,
        epochs: int = 50,
        len_factor: float = 0.5,
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
        k_neighbors: int = 2,
        HPO: bool = False,
        trial: optuna.trial.Trial = None,
        num_train_machines: int = 5,
        num_splits: int = 5,
    ):

        # get the combinations of machines (list of the machines)
        list_machines_combinations = self.sample_machines(
            num_train_machines=num_train_machines, num_splits=num_splits
        )

        # load data, model, loss, optimizer and hyperparameters
        manager = multiprocessing.Manager()
        dict_data_shared = manager.dict()

        for index_split, list_machines in enumerate(list_machines_combinations):

            self.preparation_before_training(
                k_smote=k_smote,
                batch_size=batch_size,
                epochs=epochs,
                len_factor=len_factor,
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
                list_machines=list_machines,
                k_neighbors=k_neighbors,
                index_split=index_split,
                HPO=HPO,
                trial=trial,
                num_train_machines=num_train_machines,
                num_splits=num_splits,
                dict_data_shared=dict_data_shared,
            )

        for ep in range(epochs):

            hmean_test_this_epoch = self.run_hmean_calculation_one_epoch_hpo(
                project=project,
                api_token=api_token,
                list_machines_combinations=list_machines_combinations,
                number_trial=trial.number,
                ep=ep,
                k_neighbors=k_neighbors,
                dict_data_shared=dict_data_shared,
            )

            # check for pruning
            hmean_test_this_epoch = np.mean(hmean_test_this_epoch)
            print("hmean_test_this_epoch cross vali:", hmean_test_this_epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return hmean_test_this_epoch


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
    # hyperaparameter
    project = "DCASE2024/wav-test"
    api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
    k_smote: int = 5
    batch_size: int = 12
    len_factor: float = 0.2
    epochs: int = 50
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
    k_neighbors: int = 2

    # fix hyperaparameter for hpo
    trial: optuna.trial.Trial = None
    index_split = None
    num_train_machines: int = 5
    num_splits: int = 5
    list_machines = None

    # hpo argument argparse
    parser = argparse.ArgumentParser(description="Ein einfaches Beispiel für argparse.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="choose run mode between 'reimplement' or 'hpo', default is 'reimplement'",
        choices=["reimplement", "hpo"],
        default="reimplement",
    )
    args = parser.parse_args()

    if args.mode == "reimplement":
        HPO: bool = False
    elif args.mode == "hpo":
        HPO: bool = True

    # hyperparameters optimization
    if HPO:
        # Use 'spawn' to avoid CUDA issues
        multiprocessing.set_start_method("spawn", force=False)

        # create directory for HPO
        path_hpo_directory = ad.path_hpo_directory
        os.makedirs(path_hpo_directory, exist_ok=True)

        # data base file for hpo
        db_hpo = "hpo.db"
        path_db_hpo = os.path.join(path_hpo_directory, db_hpo)
        db_hpo_sqlite = "sqlite:///{}".format(path_db_hpo)

        # csv file for hpo
        csv_hpo = "hpo.csv"
        path_csv_hpo = os.path.join(path_hpo_directory, csv_hpo)

        # optuna configuration
        sampler = optuna.samplers.TPESampler(seed=ad.seed)
        pruner = optuna.pruners.MedianPruner()
        study_name = "dcase24_hpo"
        n_trials_total = 100

        # fix hyperparameters
        project = "DCASE2024/dcase-HPO"
        api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ=="
        k_smote = 5
        batch_size = 12
        num_train_machines = 5
        num_splits = 5
        epochs = 50
        len_factor = 0.5

        # objective functions
        def objective(trial: optuna.trial.Trial):

            # tuned hyperparamters
            learning_rate = trial.suggest_float(
                name="learning_rate", low=1e-7, high=1e-1, log=True
            )

            step_warmup = trial.suggest_int(name="step_warmup", low=2, high=256, step=2)

            emb_size = trial.suggest_int(name="emb_size", low=2, high=2048, step=2)

            loss_type = trial.suggest_categorical(
                name="loss_type", choices=["arcface", "adacos"]
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

            k_neighbors = trial.suggest_int(name="k_neighbors", low=2, high=32, step=1)

            # mean for cv
            hmean_cv = ad.cross_validation(
                project=project,
                api_token=api_token,
                k_smote=k_smote,
                batch_size=batch_size,
                len_factor=len_factor,
                epochs=epochs,
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
            len_factor=len_factor,
            batch_size=batch_size,
            epochs=epochs,
            loss_type=loss_type,
            learning_rate=learning_rate,
            step_warmup=step_warmup,
            k_neighbors=k_neighbors,
            scheduler_type=scheduler_type,
            min_lr=min_lr,
            margin=margin,
            scale=scale,
            emb_size=emb_size,
        )
