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
import neptune
from neptune.utils import stringify_unsupported
from torch.optim.lr_scheduler import (
    LambdaLR,
    CosineAnnealingWarmRestarts,
)
from datetime import datetime
from peft import LoraConfig, get_peft_model
import optuna
import random
import itertools

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

    def load_dataset_tensor(self, k_smote=5, kind=None):
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

        # return data based on kind
        if kind == "train":
            return train_dataset_attribute
        elif kind == "test":
            return test_dataset_attribute
        elif kind == "smote":
            return dataset_smote
        else:
            return dataset_smote, train_dataset_attribute, test_dataset_attribute

    def data_loader(
        self, dataset, batch_size, num_iterations=None, uniform_sampling=False
    ):
        """
        convert tensor data to dataloader
        """
        # check if uniform_sampling
        if uniform_sampling and isinstance(num_iterations, int):
            # total number of instances
            num_instances = num_iterations * batch_size

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
        print("y shape:", y.shape)

        # turn X and y back to Tensor Dataset
        dataset = TensorDataset(X, y)

        return dataset

    def name_saved_model(self):
        """
        get the model name to save it
        """
        # Get current date and time
        current_datetime = datetime.now()

        # Format as string
        datetime_string = current_datetime.strftime("%Y_%m_%d-%H_%M_%S")

        # create model_name
        model_name = "model_{}_embsize_3.pth".format(datetime_string)

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
            function to pop the
            """
            for i in arg:
                dictionary.pop(i, None)
            return dictionary

        # pop some keys for hyperparameters dictionary
        if set({"lora", "HPO", "loss_type"}).issubset(set(kwargs.keys())):
            # lora
            lora = kwargs["lora"]
            if not lora:
                dict_pop(kwargs, "r", "lora_alpha", "lora_dropout")

            # HPO
            HPO = kwargs["HPO"]
            if not HPO:
                dict_pop(kwargs, "trial")
                dict_pop(kwargs, "num_train_machines")
                dict_pop(kwargs, "num_splits")
                dict_pop(kwargs, "list_machines")

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
        knn_pretrained: list,
        scaler_pretrained: StandardScaler,
        hyperparameters: dict,
    ):
        """
        save the pretrained model in pretrained_model directory
        """
        # get model name
        model_name = hyperparameters["model_name"]
        path_pretrained_model_loss = os.path.join(
            self.path_pretrained_models_directory, model_name
        )
        torch.save(
            {
                "model_state_dict": model_pretrained.state_dict(),
                "loss_state_dict": loss_pretrained.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "knn_pretrained": knn_pretrained,
                "scaler_pretrained": scaler_pretrained,
                "hyperparameters": hyperparameters,
            },
            path_pretrained_model_loss,
        )

        print(
            "pretrained model, loss, optimizer and hyperparameters saved to ",
            path_pretrained_model_loss,
        )

    def cross_validation(
        self,
        project="DCASE2024/wav-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiODUwOWJmNy05M2UzLTQ2ZDItYjU2MS0yZWMwNGI1NDI5ZjAifQ==",
        num_train_machines: int = 5,
        num_splits: int = 5,
        k_smote: int = 5,
        batch_size: int = 8,
        num_instances_factor: int = 100,
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
        HPO: bool = True,
        trial: optuna.trial.Trial = None,
    ):
        """
        perform cross validation
        """
        # init run

        # get the combinations of machines (list of the machines)
        combinations = self.sample_machines(
            num_train_machines=num_train_machines, num_splits=num_splits
        )

        # get data
        dataset_smote = self.load_dataset_tensor(k_smote=k_smote, kind="smote")
        dataset_test = self.load_dataset_tensor(k_smote=k_smote, kind="test")

        # cross validation
        for idx_split, list_machines in enumerate(combinations):

            # sort data based on list_machines
            num_classes = self.label_machine(list_machines=list_machines)
            dataset_smote = self.sort_data_machines(
                dataset=dataset_smote, list_machines=list_machines
            )
            dataset_test = self.sort_data_machines(
                dataset=dataset_test, list_machines=list_machines
            )

            # turn to data loader
            num_instances = num_instances_factor * batch_size
            dataloader_smote_uniform = self.data_loader(
                dataset=dataset_smote,
                batch_size=batch_size,
                num_iterations=num_instances_factor,
                uniform_sampling=True,
            )
            dataloader_smote_attritbute = self.data_loader(
                dataset=dataset_smote, batch_size=batch_size
            )
            dataloader_test_attribute = self.data_loader(
                dataset=dataset_test, batch_size=batch_size
            )

            # init run
            run = neptune.init_run(project=project, api_token=api_token)

            # load model
            name_saved_model = self.name_saved_model()
            input_size = (
                dataloader_test_attribute.dataset.tensors[0].shape[1] // self.fs
            )
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
                model = nn.DataParallel(
                    model, device_ids=list(range(self.n_gpus)), dim=0
                )
            model = model.to(self.device)
            num_params = sum(p.numel() for p in model.parameters())
            num_params_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            # load loss
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

            # load scheduler
            scheduler = self.load_scheduler(
                optimizer=optimizer,
                scheduler_type=scheduler_type,
                step_warmup=step_warmup,
                min_lr=min_lr,
            )

            # hyperparameters and configurations
            hyperparameters = self.hyperparameters_configuration_dict(
                k_smote=k_smote,
                lora=lora,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                batch_size=batch_size,
                num_instances_factor=num_instances_factor,
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
                trial=trial,
                model_name=name_saved_model,
                input_size=input_size,
                trial_number=trial.number,
                index_split=idx_split,
            )

            configuration = self.hyperparameters_configuration_dict(
                seed=self.seed,
                num_params=num_params,
                num_params_trainable=num_params_trainable,
                n_gpus=self.n_gpus,
                device=stringify_unsupported(self.device),
                vram=self.vram,
            )

    def check_function(self):
        return self.hyperparameters_configuration_dict(seed=self.seed)


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
