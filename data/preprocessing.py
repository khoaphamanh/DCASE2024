import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import librosa
import sys

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import download_data


# data preprocessing
class DataPreprocessing:
    def __init__(self, data_name):

        # directory information
        self.data_name = data_name
        self.path_data_preprocessing = os.path.abspath(__file__)
        self.path_data_directory = os.path.dirname(self.path_data_preprocessing)
        self.path_main_directory = os.path.dirname(self.path_data_directory)

        # data information
        self.path_data_name_directory = os.path.join(
            self.path_main_directory, "data", data_name
        )
        self.machines = sorted(os.listdir(self.path_data_name_directory), key=str.lower)
        self.path_machines = [
            os.path.join(self.path_data_name_directory, i) for i in self.machines
        ]
        self.path_attribute_file = [
            os.path.join(m, "attributes_00.csv") for m in self.path_machines
        ]
        self.type_data = ["train", "test"]

        # timeseries information
        self.fs = 16000
        self.machine_no_attribute = ["slider", "gearbox", "ToyTrain"]
        self.duration = {
            tuple(i for i in self.machines if i not in ["ToyCar", "ToyTrain"]): 10,
            ("ToyCar", "ToyTrain"): 12,
        }
        self.label_condition_number = {"normal": 0, "anomaly": 1}

        # dict, list path
        # self.path_label_unique = os.path.join(f"{}")

    def read_data(self):
        """
        Read features, labels from .wav files
        """

        # train data
        train_raw_data = []
        train_label = []

        # test data
        test_raw_data = []
        test_label = []

        # dict for data timeseries information with syntax [index,name_ts,label_attributt, label_condition ]
        data_timeseries_information = []
        label_unique = []

        # loop type train test
        idx_data = 0  # index of the each timeseries, from 0 to 999 (total 1000)

        for t in self.type_data:

            # loop machines
            for path_m in self.path_machines:

                path_type_machine = os.path.join(path_m, t)
                name_ts = os.listdir(path_type_machine)

                # loop name ts
                for n_ts in name_ts:

                    # split the name of timeseries
                    n_ts_split = n_ts.split("_")
                    machine = path_m.split("/")[-1]
                    domain = n_ts_split[2]
                    condition = n_ts_split[4]
                    attribute = "_".join(n_ts_split[6:]).replace(".wav", "")

                    # label attribute as string
                    label_attribute = f"{machine}_{domain}_{attribute}"

                    # add to label_unique list
                    if label_attribute not in label_unique:
                        label_unique.append(label_attribute)

                    # label condition
                    label_condition_number = self.label_condition_number[condition]

                    # append to data_timeseries_information
                    label_attribute_number = label_unique.index(label_attribute)
                    data_timeseries_information.append(
                        [idx_data, n_ts, label_attribute_number, label_condition_number]
                    )

                    # next index
                    idx_data = idx_data + 1

                    # break
                # break
            # break

        # convert
        label_unique = [[i, label_unique[i]] for i in range(len(label_unique))]

        print("data_timeseries_information:", data_timeseries_information)
        print("label_unique:", label_unique)


if __name__ == "__main__":

    from timeit import default_timer

    start = default_timer()
    print()
    data_name = "develop"
    data_preprocessing = DataPreprocessing(data_name=data_name)

    # path_models_directory = data_preprocessing.path_models_directory
    # print("path_models_directory:", path_models_directory)

    # machine = data_preprocessing.machines
    # print("machine:", machine)

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

    # duration = data_preprocessing.duration
    # print("duration:", duration)
    data_preprocessing.read_data()
