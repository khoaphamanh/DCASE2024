import torch
from torch import nn
from beats.beats import BEATs, BEATsConfig
from beats.beats_custom import BEATsCustom
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import sys
import os
import numpy as np
from loss import AdaCosLoss
from torchinfo import summary

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing


# class Anomaly Detechtion
class AnomalyDetection:
    def __init__(self, data_name, seed):

        # information this class
        self.path_models_directory = os.path.dirname(os.path.abspath(__file__))

        # pretrained models BEATs
        self.path_pretrained_models_directory = os.path.join(
            self.path_models_directory, "pretrained_models"
        )
        if os.path.exists(self.path_pretrained_models_directory):
            import download_models

        self.path_beat_iter3_state_dict = os.path.join(
            self.path_pretrained_models_directory, "BEATs_iter3.pt"
        )

        # preprocessing class
        self.data_preprocessing = DataPreprocessing(data_name)

        # configuration of the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count()
        self.seed = seed

    def load_raw_data(self):
        """
        function to load raw data as np.array, label as [attritbute, condition]
        """
        # load raw data as numpy
        train_data, train_label, test_data, test_label = (
            self.data_preprocessing.load_data()
        )
        return train_data, train_label, test_data, test_label

    def load_data_attribute(self):
        """
        function to sort the instances in test data, which only have the same label attribute with train data
        """
        # load raw data
        train_data, train_label, test_data, test_label = self.load_raw_data()

        # find the unique label in train data
        label_train_attribute = train_label[:, 1]
        self.label_train_attribute_unique = np.unique(label_train_attribute)
        label_test_attribute = test_label[:, 1]
        index_label_unique_attribute_in_test = [
            i
            for i in range(len(test_data))
            if label_test_attribute[i] in self.label_train_attribute_unique
        ]

        test_data = test_data[index_label_unique_attribute_in_test]
        test_label = test_label[index_label_unique_attribute_in_test]

        # conver to tensor dataset
        return train_data, train_label, test_data, test_label

    def smote(self):
        train_data, train_label, test_data, test_label = self.load_data_attribute()
        train_label = train_label[:, 1]
        smote = SMOTE(random_state=self.seed)
        train_data, train_label = smote.fit_resample(train_data, train_label)
        return train_data, train_label, test_data, test_label

    def load_model(self, input_size=12, embedding_dim=None):
        # function to load model beats
        model = BEATsCustom(
            path_state_dict=self.path_beat_iter3_state_dict,
            input_size=input_size,
            embedding_dim=embedding_dim,
        )
        return model


# run this script
if __name__ == "__main__":
    seed = 1998
    develop_name = "develop"
    ad = AnomalyDetection(data_name=develop_name, seed=seed)

    path_beat_iter3_state_dict = ad.path_beat_iter3_state_dict
    # print("path_beat_iter3_state_dict:", path_beat_iter3_state_dict)

    # model = BEATsCustom(path_state_dict=path_beat_iter3_state_dict)

    # a = torch.randn(2, 496, 768)
    # asp = AttentiveStatisticsPooling(input_size=10)

    # out = asp(a)
    # print("out shape:", out.shape)

    # a = torch.randn(2, 12 * 16000)
    # model = BEATsCustom(path_state_dict=path_beat_iter3_state_dict, input_size=12,embedding_dim=1024)
    # out = model(a)
    # print("out shape:", out.shape)

    # summary(model)

    # train_data, train_label, test_data, test_label = ad.load_data()
    # print("train_data:", train_data.shape)
    # print("train_data", train_data.dtype)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("train_data", test_data.dtype)
    # print("test_label:", test_label.shape)

    train_data, train_label, test_data, test_label = ad.load_data_attribute()

    print("train_data:", train_data.shape)
    print("train_data", train_data.dtype)
    print("train_label:", train_label.shape)
    print("test_data:", test_data.shape)
    print("train_data", test_data.dtype)
    print("test_label:", test_label.shape)

    label_train_unique, count = np.unique(train_label[:, 1], return_counts=True)
    print("label_train_unique:", label_train_unique)
    print("count:", count)

    label_test_unique, count = np.unique(test_label[:, 1], return_counts=True)
    print("label_test_unique:", label_test_unique)
    print("count:", count)

    # train_data, train_label, test_data, test_label = ad.smote()

    # print("train_data:", train_data.shape)
    # print("train_data", train_data.dtype)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("train_data", test_data.dtype)
    # print("test_label:", test_label.shape)
