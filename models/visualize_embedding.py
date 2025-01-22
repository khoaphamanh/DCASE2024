import torch
from torch.utils.data import DataLoader
from beats.beats_custom import BEATsCustom
import os
import numpy as np
from train import AnomalyDetection
from loss import AdaCosLoss, ArcFaceLoss
from umap import UMAP
from sklearn.manifold import TSNE
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


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
        path_pretrained_embedding = self.path_pretrained_embedding(
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
                "hyperparameters": hyperparameters,
            }

            torch.save(pretrained_embedding, path_pretrained_embedding)

        else:
            pretrained_embedding = torch.load(path_pretrained_embedding)

        return pretrained_embedding

    def path_pretrained_embedding(self, pretrained_file: str):
        """
        path and name of pretrained embedding given the pretrained_file
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

    def dimension_reduction(self, pretrained_file: str, method: str = "umap"):
        """
        visualize the embedding given the pretrained_file, type_data and method
        """
        # path demension reduction
        path_embedding_dimension_reduction = self.path_embedding_dimension_reduction(
            pretrained_file=pretrained_file, method=method
        )

        # if not then train with method to reduce the dimension
        if not os.path.exists(path_embedding_dimension_reduction):
            if method == "umap":
                method = UMAP(n_components=3, random_state=self.seed)
            elif method == "tsne":
                method = TSNE(n_components=3, random_state=self.seed)

            # load the pretrained embedding
            pretrained_embedding = self.get_pretrained_embedding(
                pretrained_file=pretrained_file
            )

            # calculate the dimension reduction and normalize
            embedding_dimension_reduction = {}
            for type_data, array in pretrained_embedding.items():
                embedding, y_true, y_pred = array
                embedding_reduction = method.fit_transform(embedding)
                embedding_reduction_normalize = self.l2_normalization(
                    embedding=embedding_reduction
                )
                embedding_dimension_reduction[type_data] = embedding_reduction_normalize

            torch.save(
                embedding_dimension_reduction, path_embedding_dimension_reduction
            )

        else:
            embedding_dimension_reduction = torch.load(
                path_embedding_dimension_reduction
            )

        return embedding_dimension_reduction

    def path_embedding_dimension_reduction(
        self, pretrained_file: str, method: str = "umap"
    ):
        """
        path and name of pretrained mbedding given the pretrained_file and method
        """
        # get the pretrained_embedding_name
        pretrained_file = pretrained_file.split(".pth")[0]
        name_demension_reduction = (
            pretrained_file + "_pretrained_embedding" + "_{}".format(method) + ".pth"
        )

        # path pretrained embedding
        path_demension_reduction = os.path.join(
            self.path_pretrained_models_directory, name_demension_reduction
        )

        return path_demension_reduction

    def l2_normalization(self, embedding):
        """
        normalize the embedding (array) to have len 1 each row
        """
        # Compute L2 norm for each row: l2 = sqrt (row)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)

        # Avoid division by zero (add a small epsilon)
        epsilon = 1e-10
        norms = np.maximum(norms, epsilon)

        # Normalize each row to have L2 norm = 1
        normalized_array = embedding / norms

        return normalized_array

    def visualize(
        self,
        pretrained_file: str,
        method: str = None,
        type_data: str = None,
        type_label: str = None,
    ):
        """ """
        # type data to plot
        type_data = ["smote", "train", "test"]

        # create a fig plot for plotly
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=type_data,
            # specs=[[{"type": "scatter3d"}]],  # One 3D scatter plot
        )
        positions = [[0, 0], [0, 1], [1, 0], [1, 1]]

        # Create a color palette
        colors = px.colors.qualitative.Plotly

        # embedding pretrained from pretrained model
        embedding_pretrained = self.get_pretrained_embedding(
            pretrained_file=pretrained_file
        )
        hyperparmeters = embedding_pretrained["hyperparameters"]

        # check emb_size if use demension reduction or not
        emb_size = hyperparmeters["emb_size"]
        if emb_size != 3:
            embedding_dimension_reduction = self.dimension_reduction(
                pretrained_file=pretrained_file, method=method
            )

        for idx, typ in enumerate(type_data):
            y_true = embedding_pretrained[typ][1]
            y_pred = embedding_pretrained[typ][2]

            if emb_size != 3:
                embedding = embedding_dimension_reduction[type_data]

            else:
                embedding = embedding_pretrained[typ][0]

            # y_check = self.y_check_array(y_true_array=y_true, y_pred_array=y_pred)

            # Create a figure with one subplot (only "Label Prediction")

            # Define the scatter plot for Label Prediction
            scatter = go.Scatter3d(
                x=embedding[:, 0],
                y=embedding[:, 1],
                z=embedding[:, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=colors[i % len(colors)],  # Use y_true for coloring the points
                    colorscale="Viridis",  # You can change the colorscale here
                    colorbar=dict(title="Labels"),
                ),
                name="Label Prediction",  # Specify the name for the legend (no "trace 0")
                customdata=np.stack(
                    [y_true, y_pred], axis=-1
                ),  # Include y_true and y_pred
                hovertemplate=(
                    "X: %{x:.2f}<br>"
                    "Y: %{y:.2f}<br>"
                    "Z: %{z:.2f}<br>"
                    "True Label: %{customdata[0]}<br>"  # Display y_true
                    "Predicted Label: %{customdata[1]}"  # Display y_pred
                ),
            )

            # Add scatter plot to the subplot
            pos = positions[idx]
            fig.add_trace(scatter, row=pos[0], col=pos[1])

        # Update layout for the figure
        fig.update_layout(
            title=f"Visualize Embedding {type_data} {method}",
        )

        # Show the figure
        fig.show()

    def plot_3d_embedding_label_prediction(self):
        """
        plot the 3D label prediction from embedding
        """

    def plot_gray_sphere(self, ax):
        """
        plot the gray sphere with radius 1
        """
        # Define the grid for spherical coordinates
        u = np.linspace(0, 2 * np.pi, 100)  # Longitude
        v = np.linspace(0, np.pi, 100)  # Latitude

        # Parametric equations for a sphere
        x = np.sin(v)[:, None] * np.cos(u)  # X = sin(latitude) * cos(longitude)
        y = np.sin(v)[:, None] * np.sin(u)  # Y = sin(latitude) * sin(longitude)
        z = np.cos(v)[:, None]  # Z = cos(latitude)

        # Plot the surface of the sphere
        ax.plot_surface(x, y, z, color="gray", alpha=0.1)

    def y_check_array(self, y_true_array: np.array, y_pred_array: np.array):
        """
        get the check between y_true and y_pred with 1 is correct prediction and 0 is false prediction
        """
        # check the prediction with 1 correct prediction and 0 false prediction
        y_check = np.equal(y_true_array, y_pred_array)
        y_check = y_check.astype(int)
        return y_check


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
    embedding_dimension_reduction_umap = visualize_embedding.dimension_reduction(
        pretrained_file=pretrained_file, method="umap"
    )

    print(embedding_dimension_reduction_umap.keys())
    for i in embedding_dimension_reduction_umap:

        check = embedding_dimension_reduction_umap[i]
        print("check shape:", check.shape)

    embedding_dimension_reduction_tsne = visualize_embedding.dimension_reduction(
        pretrained_file=pretrained_file, method="tsne"
    )

    print(embedding_dimension_reduction_tsne.keys())

    fig = visualize_embedding.visualize(
        pretrained_file=pretrained_file,
        method="umap",
        type_data="smote",
    )

    end = default_timer()
    print(end - start)
