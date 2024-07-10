import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import copy


# data preprocessing
class DataPreprocessing:
    def __init__(
        self,
        data_name,
    ):

        # directory information
        self.data_name = data_name
        self.preprocessing_path = os.path.abspath(__file__)
        self.data_path = os.path.dirname(self.preprocessing_path)
        self.dcase_path = os.path.dirname(self.data_path)
        if data_name == "develop":
            data_name = "10902294"
        self.raw_data_path = os.path.join(self.dcase_path, data_name)

        # data information
        self.machines = [i for i in os.listdir(self.raw_data_path) if "." not in i]
        self.machines_path = [
            os.path.join(self.raw_data_path, m) for m in self.machines
        ]
        self.attribute_file_path = [
            os.path.join(m, "attributes_00.csv") for m in self.machines_path
        ]
        self.no_attribute_machine = ["slider", "gearbox"]

        # time series information
        self.fs = 16000
        self.duration = {
            tuple(i for i in self.machines if i not in ["ToyCar", "ToyTrain"]): 10,
            ("ToyCar", "ToyTrain"): 12,
        }
        self.len_ts = {
            tuple(i for i in self.machines if i not in ["ToyCar", "ToyTrain"]): 10
            * self.fs,
            ("ToyCar", "ToyTrain"): 12 * self.fs,
        }

        # analysis
        self.num_classes_train = len(self.machines) * 2
        self.unique_labels_machine_domain_path = os.path.join(
            self.data_path, "unique_labels_machine_domain.csv"
        )
        self.full_labels_ts_path = os.path.join(self.data_path, "full_labels_ts.csv")
        self.ts_analysis_path = os.path.join(self.data_path, "ts_analysis123.csv")
        self.label_analysis = {
            d: [k for k, v in self.unique_labels_machine_domain().items() if d in v]
            for d in ["source", "target"]
        }
        self.auc_roc_name = [
            "test_{}_{}".format(m, d)
            for m in self.machines
            for d in ["source", "target"]
        ]

    def read_data(self):
        """
        Read the feature and labels from train and test wave.mp4 files
        """
        train_data = []
        train_label = []

        test_data = []
        test_label = []

        idx_ts = 0
        unique_labels_machine_domain = []
        full_labels_ts = {}

        # type data train and test
        for type in ["train", "test"]:
            # loop every machine in 7 machines
            for i, machine_path in enumerate(self.machines_path):

                # get the path of machine and name of each timeseries
                machine = self.machines[i]
                type_path = os.path.join(machine_path, type)
                name_ts = os.listdir(type_path)

                for name in name_ts:

                    # get the array of the full timeseries
                    name_path = os.path.join(type_path, name)
                    fs, ts = wavfile.read(name_path)

                    # get the domain
                    if "source" in name:
                        dommain = "source"
                    elif "target" in name:
                        dommain = "target"

                    # get the name of the labels train and save in the unique labels dict
                    label_machine_domain = "{}_{}".format(machine, dommain)
                    if label_machine_domain not in unique_labels_machine_domain:
                        unique_labels_machine_domain.append(label_machine_domain)

                    # save array to data list and labels to label list with syntax: [idx,label_machine_domain,condition]
                    if type == "train":
                        train_data.append(ts)
                        train_label.append([idx_ts, label_machine_domain, 0])

                    # get the name of the labels condition (only in test data)
                    elif type == "test":
                        if "normal" in name:
                            condition = 0
                        elif "anomaly" in name:
                            condition = 1
                        test_data.append(ts)
                        test_label.append([idx_ts, label_machine_domain, condition])

                    # save full label of timeseries to a dict
                    full_labels_ts[idx_ts] = "{}_".format(machine) + name.replace(
                        ".wav", ""
                    ).replace("section_00_", "")

                    # update index of ts
                    idx_ts = idx_ts + 1

        # unique machine labels to csv and save it
        unique_labels_machine_domain = {
            i: l for i, l in enumerate(unique_labels_machine_domain)
        }
        unique_labels_machine_domain = pd.DataFrame(
            list(unique_labels_machine_domain.items()), columns=["Index", "Label"]
        )
        if not os.path.exists(self.unique_labels_machine_domain_path):
            unique_labels_machine_domain.to_csv(
                self.unique_labels_machine_domain_path, index=False
            )

        # full labels to csv and save it
        full_labels_ts = pd.DataFrame(
            list(full_labels_ts.items()), columns=["Index", "Label"]
        )
        if not os.path.exists(self.full_labels_ts_path):
            full_labels_ts.to_csv(self.full_labels_ts_path, index=False)

        return train_data, train_label, test_data, test_label

    def unique_labels_machine_domain(self):
        """
        load unique_labels_machine_domain as dict from csv
        """
        # check if unique_labels_machine_domain_path available
        if not os.path.exists(self.unique_labels_machine_domain_path):
            self.read_data()

        # read ts_lb.csv
        unique_labels_machine_domain = pd.read_csv(
            self.unique_labels_machine_domain_path
        )
        unique_labels_machine_domain = unique_labels_machine_domain["Label"].to_dict()

        return unique_labels_machine_domain

    def full_labels_ts(self):
        """
        load full_labels_ts as dict from csv
        """
        # check if unique_labels_machine_domain_path available
        if not os.path.exists(self.full_labels_ts_path):
            self.read_data()

        # read ts_lb.csv
        full_labels_ts = pd.read_csv(self.full_labels_ts_path)
        full_labels_ts = full_labels_ts["Label"].to_dict()

        return full_labels_ts

    def windowing(self, window_size=None, hop_size=None):
        """
        Cut time series into segment with given window_size and hop_size
        """
        # default window_size and hop_size if not given
        if window_size is None:
            window_size = self.fs * 2
        if hop_size is None:
            hop_size = self.fs

        # get data
        train_data, train_label, test_data, test_label = self.read_data()

        # unque label machine domain
        unique_labels_machine_domain = self.unique_labels_machine_domain()
        unique_labels_machine_domain = {
            v: k for (k, v) in unique_labels_machine_domain.items()
        }

        # windowing
        train_windows = []
        test_windows = []
        train_label_windows = []
        test_label_windows = []

        data = [
            (train_data, train_label, train_windows, train_label_windows),
            (test_data, test_label, test_windows, test_label_windows),
        ]

        # windowing loop
        for type_data, type_label, windows, label_windows in data:
            for ts, lb in zip(type_data, type_label):
                n_samples = len(ts)
                n_windows = (n_samples - window_size) // hop_size + 1

                for n in range(n_windows):
                    # get the window timeseries
                    ts_window = ts[n * hop_size : n * hop_size + window_size]
                    windows.append(ts_window)

                    # get label for each window timeseries
                    idx, label, condtion = lb
                    label = unique_labels_machine_domain[label]
                    label_windows.append([idx, label, condtion])

        return (
            np.array(train_windows),
            np.array(train_label_windows),
            np.array(test_windows),
            np.array(test_label_windows),
        )

    def load_data(self, window_size=None, hop_size=None, train=True, test=True):
        """
        load data if data already availabel, else will read, windowing, saved and load data
        """
        # default window_size and hop_size if not given
        if window_size is None:
            window_size = self.fs * 2
        if hop_size is None:
            hop_size = self.fs

        # check if data available:
        name_train_data = "train_data_{}_{}_new.npy".format(window_size, hop_size)
        name_train_label = "train_label_{}_{}_new.npy".format(window_size, hop_size)
        name_test_data = "test_data_{}_{}_new.npy".format(window_size, hop_size)
        name_test_label = "test_label_{}_{}_new.npy".format(window_size, hop_size)

        name = [name_train_data, name_train_label, name_test_data, name_test_label]
        path_data_files = [os.path.join(self.data_path, i) for i in name]

        check_data_available = [
            True if i in os.listdir(self.data_path) else False for i in name
        ]

        if all(check_data_available):
            train_data, train_label, test_data, test_label = [
                np.load(i) for i in path_data_files
            ]

        # if not create it and saved it
        else:
            train_data, train_label, test_data, test_label = self.windowing(
                window_size=window_size, hop_size=hop_size
            )
            np.save(path_data_files[0], train_data)
            np.save(path_data_files[1], train_label)
            np.save(path_data_files[2], test_data)
            np.save(path_data_files[3], test_label)

        if train == True and test == True:
            return (
                train_data.astype(np.float32),
                train_label.astype(np.int32),
                test_data.astype(np.float32),
                test_label.astype(np.int32),
            )
        elif train == True and test == False:
            return train_data.astype(np.float32), train_label.astype(np.int32)

    def ts_analysis(self):
        """
        sort the id of ts with labels of machine
        """
        # load full labels ts
        full_labels_ts = self.full_labels_ts()

        # create ts analysis
        if not os.path.exists(self.ts_analysis_path):
            ts_analysis = {}
            for type in ["train", "test"]:
                for domain in ["source", "target"]:
                    key = "{}_{}".format(type, domain)
                    ts_analysis[key] = [
                        i
                        for i, l in full_labels_ts.items()
                        if type in l and domain in l
                    ]
                    if type == "test":
                        for condition in ["normal", "anomaly"]:
                            key = "{}_{}_{}".format(type, domain, condition)
                            ts_analysis[key] = [
                                i
                                for i, l in full_labels_ts.items()
                                if type in l and domain in l and condition in l
                            ]

                            key = "{}_{}".format(type, condition)
                            ts_analysis[key] = [
                                i
                                for i, l in full_labels_ts.items()
                                if type in l and condition in l
                            ]

                    for machine in self.machines:
                        key = "{}_{}_{}".format(type, machine, domain)
                        ts_analysis[key] = [
                            i
                            for i, l in full_labels_ts.items()
                            if type in l and domain in l and machine in l
                        ]

            # save it to .csv for better visualize
            ts_analysis_csv = copy.deepcopy(ts_analysis)
            max_len = max([len(v) for v in ts_analysis.values()])

            for k, v in ts_analysis_csv.items():
                while len(v) < max_len:
                    v.append(None)

            ts_analysis_csv = pd.DataFrame(ts_analysis_csv)
            ts_analysis_csv.to_csv(self.ts_analysis_path, index=False)

        else:
            ts_analysis = pd.read_csv(self.ts_analysis_path)
            ts_analysis = ts_analysis.to_dict(orient="list")
            ts_analysis = {
                k: [int(x) for x in v if pd.notna(x)] for k, v in ts_analysis.items()
            }

        return ts_analysis


if __name__ == "__main__":

    data_name = "develop"
    data_preprocessing = DataPreprocessing(data_name=data_name)
    raw_data_path = data_preprocessing.raw_data_path
    # /home/phamanh/nobackup/DCASE2024/10902294
    # print("raw_data_path:", raw_data_path)

    machine_path = data_preprocessing.machines_path
    # machne_path: ['/home/phamanh/nobackup/DCASE2024/10902294/bearing', '/home/phamanh/nobackup/DCASE2024/10902294/fan', '/home/phamanh/nobackup/DCASE2024/10902294/gearbox', '/home/phamanh/nobackup/DCASE2024/10902294/slider', '/home/phamanh/nobackup/DCASE2024/10902294/ToyCar', '/home/phamanh/nobackup/DCASE2024/10902294/ToyTrain', '/home/phamanh/nobackup/DCASE2024/10902294/valve']
    # print("machne_path:", machne_path)

    data_path = data_preprocessing.data_path
    # /home/phamanh/nobackup/DCASE2024/data
    # print("data_path:", data_path)

    num_classes_train = data_preprocessing.num_classes_train
    # print("num_classes_train:", num_classes_train) 14

    # train_data, train_label, test_data, test_label = read_data = (
    #     data_preprocessing.read_data()
    # )

    # for i in read_data:
    #     print(len(i))

    unique_labels_machine_domain = data_preprocessing.unique_labels_machine_domain()
    # print("unique_labels_machine_domain:", unique_labels_machine_domain)

    full_labels_ts = data_preprocessing.full_labels_ts()
    # print("full_labels_ts:", full_labels_ts)

    # train_data, train_label, test_data, test_label = windowing = (
    #     data_preprocessing.windowing()
    # )
    # for i in windowing:
    #     print(i.shape)
    #     print(i)

    train_data, train_label, test_data, test_label = load_data = (
        data_preprocessing.load_data()
    )
    # for i in load_data:
    #     print(i.shape)
    train_label_ts = train_label[:, 0]
    # print("train_label_ts:", train_label_ts)

    train_label = train_label[:, 1]
    check_unique_train = np.unique(train_label, return_counts=True)
    # print("check_unique_train:", check_unique_train)

    test_label = test_label[:, 1]
    check_unique_test = np.unique(test_label, return_counts=True)
    # print("check_unique_test:", check_unique_test)

    ts_analysis = data_preprocessing.ts_analysis()
    # print("ts_analysis:", ts_analysis)
    # print(ts_analysis.keys())
    # for k, v in ts_analysis.items():
    #     print(len(v))

    # from sklearn.utils import class_weight

    # cw = class_weight.compute_class_weight(
    #     class_weight="balanced", classes=np.unique(train_label), y=train_label
    # )
    # print("cw:", cw)

    label_analysis = data_preprocessing.label_analysis
    print("label_analysis:", label_analysis)

    auc_roc_type = data_preprocessing.auc_roc_name
    print("auc_roc_type:", auc_roc_type)
