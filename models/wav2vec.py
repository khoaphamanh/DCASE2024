from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
import numpy as np
import torch
import sys
import os
from torch import nn
import random
from torchvision.transforms.v2 import MixUp
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.utils.data import DataLoader, TensorDataset
import neptune

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing, raw_data_path
import utils


class Wav2VecXLR300MCustom(nn.Module):
    def __init__(self, fs, emb_size, output_size):
        super().__init__()
        self.pre_trained_wav2vec = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )

        self.fs = fs
        self.out_layer = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=3168, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, output_size),
        )

    def forward(self, x):
        x = self.processor(x, return_tensors="pt", sampling_rate=self.fs).input_values[
            0
        ]
        x = self.pre_trained_wav2vec(x).logits
        x = self.out_layer(x)
        return x


class Wav2VecXLR300M:
    def __init__(self, data_preprocessing: DataPreprocessing, seed):

        # data preprocessing
        self.data_preprocessing = data_preprocessing

        # time series information
        self.fs = data_preprocessing.fs
        # self.num_classes_train = 67

        # model configuration parameter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_size = 0.8

    def load_train_data(self, window_size=None, hop_size=None):
        """
        Load train data with labels are attributes given window size and hop size. Defaulf is 16000 for both
        """
        # load data
        X_train, y_train = self.data_preprocessing.load_data(
            window_size=window_size, hop_size=hop_size, train=True, test=False
        )
        # num classes, output size
        self.num_classes_train = np.unique(y_train)

        return X_train, y_train

    def data_loader(self, batch_size, window_size=None, hop_size=None):
        """
        Turn data to pytorch,split it to train val and turn it to dataloader
        """
        # load train data
        X, y = self.load_train_data(window_size=window_size, hop_size=hop_size)

        # split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=self.train_size, random_state=self.seed, stratify=y
        )

        # compute the class weights
        self.class_weights = class_weight.compute_class_weight(
            class_weight="balance", classes=self.num_classes_train, y=y_train
        )

        # dataloader
        train_data = TensorDataset(torch.tensor(X_train, y_train))
        val_data = TensorDataset(torch.tensor(X_val, y_val))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader

    def speed_perturb(self, min_rate, max_rate, p):
        """
        Speed perturb for stretch of compress audio signal
        """

        return None

    def mix_up(self, alpha, num_classes):
        return MixUp(alpha=alpha, num_classes=num_classes)

    def train_test_loop(self, emb_size, lr):
        """
        Train test loop
        """
        # split data
        # init model
        model = Wav2VecXLR300MCustom(
            fs=self.fs, emb_size=emb_size, output_size=self.num_classes_train
        )

        # loss and optimizer
        loss = 1


if __name__ == "__main__":

    # set the seed
    seed = utils.seed
    torch.manual_seed(seed)

    # hyperparameters
    lr = utils.lr_w2v
    emb_size = utils.emb_w2v

    data_preprocessing = DataPreprocessing(raw_data_path=raw_data_path)
    w2v = Wav2VecXLR300M(data_preprocessing=data_preprocessing, seed=seed)
    fs = w2v.fs
    print("fs:", fs)

    train_data, train_label = w2v.load_train_data()
    print("train_data shape:", train_data.shape)

    unique = w2v.num_classes_train
    print("unique:", unique)
    coumt = w2v.count
    print("coumt:", coumt)
    # processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")

    # test = train_data[0:5]
    # print("test shape:", test.shape)
    # test = torch.tensor(test)
    # # test = processor(test, return_tensors="pt", sampling_rate=fs).input_values
    # print("test shape:", test.shape)
    # # print("test:", test)
    # model = Wav2VecXLR300MCustom(fs=fs, emb_size=emb_size, output_size=67)
    # with torch.inference_mode():
    #     out = model(test)
    #     print("out shape:", out.shape)
    #     print("out shape:", out)
