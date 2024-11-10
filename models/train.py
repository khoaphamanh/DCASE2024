import torch
from torch import nn
from beats.beats import BEATs, BEATsConfig
from torchinfo import summary
import sys
import os
from loss import AdaCosLoss
from torchinfo import summary


# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.preprocessing import DataPreprocessing


# class custom model
class BEATsCustom(nn.Module):
    def __init__(self, path_state_dict, input_size, embedding_dim=None):
        super().__init__()

        # beats
        self.path_state_dict = path_state_dict
        self.beats = self.load_beats_model()

        # Attentive Stat Pooling
        self.asp = AttentiveStatisticsPooling(input_size=input_size)
        self.embedding_asp = self.asp.num_features

        # in case embedding is not None
        self.embedding_dim = embedding_dim
        if embedding_dim is not None:
            self.embedding_output = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    in_features=self.embedding_asp, out_features=self.embedding_dim
                ),
            )

    def load_beats_model(self):
        # load state_dict
        beats_state_dict = torch.load(self.path_state_dict)
        cfg = BEATsConfig(beats_state_dict["cfg"])
        beats = BEATs(cfg)
        beats.load_state_dict(state_dict=beats_state_dict["model"])

        return beats

    def forward(self, x):

        # beats and asp
        x = self.beats.extract_features(x)[0]
        x = self.asp(x)

        # change the embedding dim
        if self.embedding_dim is not None:
            x = self.embedding_output(x)

        return x


# class Attentive Statistcs Pooling, source https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py#L96
class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # input size of the original clip in seconds
        if input_size == 12:
            self.input_size = 592
            self.num_features = 1184
        elif input_size == 10:
            self.input_size = 496
            self.num_features = 992

        # Attentive Statistcs Pooling layer
        self.attention = nn.Sequential(
            nn.Conv1d(self.input_size, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),  # I add this layer
            nn.Conv1d(256, self.input_size, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn = nn.BatchNorm1d(num_features=self.num_features)

    def forward(self, x):
        w = self.attention(x)
        mu = torch.sum(w * x, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4))
        x = torch.cat((mu, sg), dim=1)
        x = self.bn(x)
        return x


# class Anomaly Detechtion
class AnomalyDetection:
    def __init__(self, data_name):

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

    def load_model(self):
        # function to load model

        # load state dict
        beats_state_dict = torch.load(self.path_beat_iter3_state_dict)
        cfg = BEATsConfig(beats_state_dict["cfg"])
        beat = BEATs(cfg)
        beat.load_state_dict(state_dict=beats_state_dict["model"])


# run this script
if __name__ == "__main__":

    develop_name = "develop"
    ad = AnomalyDetection(data_name=develop_name)

    path_beat_iter3_state_dict = ad.path_beat_iter3_state_dict
    # print("path_beat_iter3_state_dict:", path_beat_iter3_state_dict)

    # model = BEATsCustom(path_state_dict=path_beat_iter3_state_dict)

    # a = torch.randn(2, 496, 768)
    # asp = AttentiveStatisticsPooling(input_size=10)

    # out = asp(a)
    # print("out shape:", out.shape)

    a = torch.randn(2, 12 * 16000)
    model = BEATsCustom(path_state_dict=path_beat_iter3_state_dict, input_size=12)
    out = model(a)
    print("out shape:", out.shape)

    summary(model)
