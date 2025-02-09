# import torch
from train import AnomalyDetection

# import optuna
import random
import itertools
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import neptune


class CrossValidation(AnomalyDetection):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

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
            dataloader_smote_uniform = self.data_loader(
                dataset=dataset_smote,
                batch_size=batch_size,
                num_instances_factor=num_instances_factor,
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

    def check_function(self):
        return self.hyperparameters_configuration_dict(seed=self.seed)


if __name__ == "__main__":

    data_name = "develop"
    seed = 1998
    cv_class = CrossValidation(data_name=data_name, seed=seed)

    # sample_machines = cv_class.sample_machines(num_splits=5, num_train_machines=5)
    # print("sample_machines len:", len(sample_machines))
    # print("sample_machines:", sample_machines)

    # from collections import Counter

    # flattened = [elem for combo in sample_machines for elem in combo]
    # # Count occurrences of each element
    # element_counts = Counter(flattened)

    # print("Element counts:", element_counts)

    # cv_class.cross_validation()

    # hyperparameters_configuration_dict = cv_class.check_function()
    # print("hyperparameters_configuration_dict:", hyperparameters_configuration_dict)
