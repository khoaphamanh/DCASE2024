import torch
from torch import nn
from beats.beats_custom import BEATsCustom
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from train import AnomalyDetection


# class Visualize Embedding for emb_size = 3
class VisualizeEmbedding(AnomalyDetection):
    def __init__(self, data_name, seed):
        super().__init__(data_name, seed)

    def load_pretrained_model(self, pretrained_file):
        """
        load the pretrained model
        """


# run this script
if __name__ == "__main__":

    # create the seed
    seed = 1998
    develop_name = "develop"

    vis_emb = VisualizeEmbedding(data_name=develop_name, seed=seed)

    seed = vis_emb.seed
    print("seed:", seed)
