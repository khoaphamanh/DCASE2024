import torch
from torch import nn
from .beats import BEATs, BEATsConfig
from torchinfo import summary
import torch.nn.functional as F


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


# class custom model
class BEATsCustom(nn.Module):
    def __init__(self, path_state_dict, input_size, emb_size=None):
        super().__init__()

        # beats
        self.path_state_dict = path_state_dict
        self.beats = self.load_beats_model()

        # Attentive Stat Pooling
        self.asp = AttentiveStatisticsPooling(input_size=input_size)
        self.embedding_asp = self.asp.num_features

        # in case embedding is not None
        self.emb_size = emb_size
        if emb_size not in [None, self.embedding_asp]:
            self.embedding_output = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features=self.embedding_asp, out_features=self.emb_size),
            )

    def load_beats_model(self):
        # load state_dict
        beats_state_dict = torch.load(self.path_state_dict)
        cfg = BEATsConfig(beats_state_dict["cfg"])
        beats = BEATs(cfg)
        beats.load_state_dict(state_dict=beats_state_dict["model"])

        return beats

    def forward(self, x=None, input_ids=None, **kwargs):

        # Determine the input tensor
        if input_ids is not None:
            x = input_ids

        # beats and asp
        x = self.beats.extract_features(x)[0]
        x = self.asp(x)

        # change the embedding dim
        if self.emb_size not in [None, self.embedding_asp]:
            x = self.embedding_output(x)

        # normalize x to have length equal to 1
        x = F.normalize(x)

        return x
