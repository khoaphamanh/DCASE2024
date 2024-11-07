import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import librosa
import sys
import pickle

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# data preprocessing
class DataPreprocessing:
    def __init__(self, data_name):

        # directory information
        self.data_name = data_name
        self.path_preprocessing = os.path.abspath(__file__)
        self.path_data_directory = os.path.dirname(self.path_preprocessing)
        self.path_main_directory = os.path.dirname(self.path_data_directory)

        # data information
        self.path_data_name_directory = os.path.join(
            self.path_main_directory, "data", data_name
        )
        if not os.path.exists(self.path_data_name_directory):
            import download_data

        self.machines = sorted(
            [
                i
                for i in os.listdir(self.path_data_name_directory)
                if os.path.isdir(os.path.join(self.path_data_name_directory, i))
            ],
            key=str.lower,
        )
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
        self.path_label_unique = os.path.join(
            self.path_data_directory,
            self.data_name,
            "{}_label_unique.csv".format(self.data_name),
        )
        # print("self:", self.path_label_unique)
        self.path_data_timeseries_information = os.path.join(
            self.path_data_directory,
            self.data_name,
            "{}_data_timeseries_information.csv".format(self.data_name),
        )
        self.path_train_data = os.path.join(
            self.path_data_directory,
            self.data_name,
            "{}_train_data.pkl".format(self.data_name),
        )
        self.path_test_data = os.path.join(
            self.path_data_directory,
            self.data_name,
            "{}_test_data.pkl".format(self.data_name),
        )
        # print("self:", self.path_data_timeseries_information)

    def read_data(self):
        """
        Read features, labels from .wav files
        """

        # train test data list
        train_data = []
        test_data = []

        # dict for data timeseries information with syntax [index,name_ts,label_attributt, label_condition ]
        data_timeseries_information = []
        label_unique = []

        # loop type train test
        idx_data = 0  # index of the each timeseries, from 0 to 8399 (total 8400)

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

                    # read .wav file
                    path_n_ts = os.path.join(
                        self.path_data_name_directory, machine, t, n_ts
                    )
                    fs, ts = wavfile.read(path_n_ts)

                    # label attribute as string
                    label_attribute = f"{machine}_{domain}_{attribute}"

                    # add to label_unique list
                    if label_attribute not in label_unique:
                        label_unique.append(label_attribute)

                    # label condition
                    label_condition_number = self.label_condition_number[condition]

                    # append to data_timeseries_information
                    label_attribute_number = label_unique.index(label_attribute)
                    path_n_ts = os.path.join(machine, n_ts)
                    data_timeseries_information.append(
                        [
                            idx_data,
                            path_n_ts,
                            label_attribute_number,
                            label_condition_number,
                            len(ts),
                        ]
                    )

                    # append to data list
                    ts_lb = [ts, label_attribute_number, label_condition_number]
                    if t == "train":
                        train_data.append(ts_lb)
                    elif t == "test":
                        test_data.append(ts_lb)

                    # next index
                    idx_data = idx_data + 1

                # break
            # break

        # save train test data list
        with open(self.path_train_data, "wb") as file:
            pickle.dump(train_data, file)

        with open(self.path_test_data, "wb") as file:
            pickle.dump(test_data, file)

        # convert to csv
        label_unique = [[i, label_unique[i]] for i in range(len(label_unique))]
        column_names = ["Number", "Attribute"]
        label_unique = pd.DataFrame(label_unique, columns=column_names)

        column_names = ["Index", "Name", "Attribute", "Condition", "Length"]
        data_timeseries_information = pd.DataFrame(
            data_timeseries_information, columns=column_names
        )

        # save the data frame to data name directory
        label_unique.to_csv(self.path_label_unique, index=False)
        data_timeseries_information.to_csv(
            self.path_data_timeseries_information, index=False
        )

    def label_unique(self):
        # load label unique as csv
        if not os.path.exists(self.path_label_unique):
            self.read_data()
        else:
            label_unique = pd.read_csv(self.path_label_unique)

        return label_unique

    def data_timeseries_information(self):
        # load data_timeseries_information as csv
        if not os.path.exists(self.path_data_timeseries_information):
            self.read_data()

        data_timeseries_information = pd.read_csv(self.path_data_timeseries_information)

        return data_timeseries_information

    def load_data(self):
        # load data .pkl file as list
        if not os.path.exists(self.path_train_data) or os.path.exists(
            self.path_test_data
        ):
            self.read_data()
        with open(self.path_train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.path_test_data, "rb") as file:
            test_data = pickle.load(file)

        return train_data, test_data


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

    # data_preprocessing.read_data()

    label_unique = data_preprocessing.label_unique()
    print("label_unique:", label_unique)

    data_timeseries_information = data_preprocessing.data_timeseries_information()
    print("data_timeseries_information:", data_timeseries_information)

    # path_label_unique = data_preprocessing.path_label_unique
    # print("path_label_unique:", path_label_unique)

    # path_train_data = data_preprocessing.path_train_data
    # print("path_train_data:", path_train_data)
