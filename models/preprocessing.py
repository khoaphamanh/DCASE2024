import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import librosa
import sys

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import data.download_data


# data preprocessing
class DataPreprocessing:
    def __init__(self, data_name):

        # directory information
        self.data_name = data_name
        self.path_data_preprocessing = os.path.abspath(__file__)
        self.path_models_directory = os.path.dirname(self.path_data_preprocessing)
        self.path_main_directory = os.path.dirname(self.path_models_directory)

        # data information
        self.path_data_directory = os.path.join(
            self.path_main_directory, "data", data_name
        )
        self.machines = os.listdir(self.path_data_directory)
        self.path_machines = [
            os.path.join(self.path_data_directory, i) for i in self.machines
        ]
        self.type_data = ["train", "test"]

        # timeseries information
        self.fs = 16000

    def read_data(self):
        """
        Read features, labels from .wav files
        """

        # train data
        train_data = []
        train_label = []

        # test data
        test_data = []
        test_label = []

        # loop type train test
        for t in self.type_data:

            # loop machines
            for path_m in self.path_machines:

                path_type_machine = os.path.join(path_m, t)
                name_ts = os.listdir(path_type_machine)
                # print("name_ts:", name_ts)

                # loop name ts
                for n_ts in name_ts:

                    a = n_ts.split("_")
                    print("a:", a)
                    break
                break
            break


if __name__ == "__main__":

    from timeit import default_timer

    start = default_timer()
    print()
    data_name = "develop"
    data_preprocessing = DataPreprocessing(data_name=data_name)

    # path_models_directory = data_preprocessing.path_models_directory
    # print("path_models_directory:", path_models_directory)

    # path_data_preprocessing = data_preprocessing.path_data_preprocessing
    # print("path_data_preprocessing:", path_data_preprocessing)

    # path_main_directory = data_preprocessing.path_main_directory
    # print("path_main_directory:", path_main_directory)

    # path_data_directory = data_preprocessing.path_data_directory
    # print("path_data_directory:", path_data_directory)

    # machines = data_preprocessing.machines
    # print("machines:", machines)

    # path_machines = data_preprocessing.path_machines
    # print("path_machines:", path_machines)

    data_preprocessing.read_data()
