import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from beats.beats_custom import BEATsCustom
import os
import numpy as np
from train import AnomalyDetection
from loss import AdaCosLoss, ArcFaceLoss


# class Visualize Embedding for emb_size = 3
class VisualizeEmbedding(AnomalyDetection):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

    def load_pretrained_model(self, pretrained_file: str):
        """
        load the pretrained model
        """
        # load the state_dict given the pretrained_file
        pretrained_path = os.path.join(
            self.path_pretrained_models_directory, pretrained_file
        )
        loaded_dict = torch.load(pretrained_path, map_location=self.device)

        # extract the state dict of each model from dictionary keys
        model_state_dict = loaded_dict["model_state_dict"]
        loss_state_dict = loaded_dict["loss_state_dict"]
        knn_pretrained = loaded_dict["knn_pretrained"]
        hyperparameters = loaded_dict["hyperparameters"]

        # load model neural network
        emb_size = hyperparameters["emb_size"]
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

        return model, loss, knn_pretrained, hyperparameters

    def get_pretrained_embedding(self, pretrained_file: str):
        """
        evaluation model and loss to get the embedding, y_true and y_pred array
        """
        # path of pretrained embedding
        path_pretrained_embedding = self.name_pretrained_embedding(
            pretrained_file=pretrained_file
        )

        # check if exsist
        if not os.path.exists(path_pretrained_embedding):

            # load pretrained model
            model, loss, knn, hyperparameters = self.load_pretrained_model(
                pretrained_file=pretrained_file
            )

            # load dataset as TensorDataset
            k_smote = hyperparameters["k_smote"]
            dataset_smote, train_dataset_attribute, test_dataset_attribute = (
                self.load_dataset_tensor(k_smote=k_smote)
            )

            # load dataset as Dataloader
            batch_size = 8 if self.vram < 23 else hyperparameters["batch_size"]

            dataloader_smote_attribute = self.data_loader(
                dataset=dataset_smote, batch_size=batch_size
            )
            dataloader_train_attribute = self.data_loader(
                dataset=train_dataset_attribute, batch_size=batch_size
            )
            dataloader_test_attribute = self.data_loader(
                dataset=test_dataset_attribute, batch_size=batch_size
            )

            # get the embedding, y_true and y_pred_label_array
            embedding_smote, y_true_smote, y_pred_smote = self.get_prediction(
                dataloader_attribute=dataloader_smote_attribute,
                model=model,
                loss=loss,
                hyperparameters=hyperparameters,
            )

            embedding_train, y_true_train, y_pred_train = self.get_prediction(
                dataloader_attribute=dataloader_train_attribute,
                model=model,
                loss=loss,
                hyperparameters=hyperparameters,
            )

            embedding_test, y_true_test, y_pred_test = self.get_prediction(
                dataloader_attribute=dataloader_test_attribute,
                model=model,
                loss=loss,
                hyperparameters=hyperparameters,
            )

            # save the file in directory pretrained_models
            pretrained_embedding = {
                "smote": [embedding_smote, y_true_smote, y_pred_smote],
                "train": [embedding_train, y_true_train, y_pred_train],
                "test": [embedding_test, y_true_test, y_pred_test],
            }

            torch.save(pretrained_embedding, path_pretrained_embedding)

        else:
            pretrained_embedding = torch.load(path_pretrained_embedding)

        return pretrained_embedding

    def name_pretrained_embedding(self, pretrained_file: str):
        """
        get the name of pretrained embedding given the pretrained_file
        """
        # get the pretrained_embedding_name
        pretrained_file = pretrained_file.split(".pth")[0]
        name_pretrained_emnbedding = pretrained_file + "_pretrained_embedding" + ".pth"

        # path pretrained embedding
        path_pretrained_embedding = os.path.join(
            self.path_pretrained_models_directory, name_pretrained_emnbedding
        )

        return path_pretrained_embedding

    def get_prediction(
        self,
        dataloader_attribute: DataLoader,
        model: BEATsCustom,
        loss: AdaCosLoss,
        hyperparameters: dict,
    ):
        """
        get the label prediction given dataset, model, loss
        """
        # create array for y_pred and y_true
        emb_size = hyperparameters["emb_size"]
        len_dataset = dataloader_attribute.dataset.tensors[0].shape[0]
        check_y_shape = len(dataloader_attribute.dataset.tensors[1].shape)

        batch_size = 8 if self.vram < 23 else hyperparameters["batch_size"]
        y_pred_label_array = np.empty(shape=(len_dataset,))
        y_true_array = np.empty(shape=(len_dataset,))
        embedding_array = np.empty(shape=(len_dataset, emb_size))

        # evaluation mode
        model.eval()
        loss.eval()

        with torch.no_grad():
            for iter_eval, (X, y) in enumerate(dataloader_attribute):

                # X to device
                X = X.to(self.device)
                y = y.to(self.device) if check_y_shape == 1 else y[:, 1].to(self.device)

                # forward pass
                embedding = model(X)

                # pred the label
                y_pred_label = loss.pred_labels(embedding=embedding, y_true=y)

                # save it to array
                embedding_array[
                    iter_eval * batch_size : iter_eval * batch_size + batch_size
                ] = embedding.cpu().numpy()
                y_true_array[
                    iter_eval * batch_size : iter_eval * batch_size + batch_size
                ] = y.cpu().numpy()
                y_pred_label_array[
                    iter_eval * batch_size : iter_eval * batch_size + batch_size
                ] = y_pred_label.cpu().numpy()

        return embedding_array, y_true_array, y_pred_label_array


# run this script
if __name__ == "__main__":

    from timeit import default_timer

    start = default_timer()

    # create the seed
    seed = 1998
    develop_name = "develop"

    visualize_embedding = VisualizeEmbedding(data_name=develop_name, seed=seed)

    # seed = visualize_embedding.seed
    # print("seed:", seed)

    pretrained_file = "k_smote_5-batch_size_32-num_instances_320000-num_iterations_1250-learning_rate_0.0001-step_warmup_120-step_accumulation_8-k_neighbors_2-emb_size_992-loss_type_adacos-2024_12_19-11_01_41.pth"
    # visualize_embedding.get_pretrained_embedding(pretrained_file=pretrained_file)

    pretrained_embedding = visualize_embedding.name_pretrained_embedding(
        pretrained_file
    )
    print("pretrained_embedding:", pretrained_embedding)

    end = default_timer()
    print(end - start)
