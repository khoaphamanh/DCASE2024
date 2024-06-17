import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import pickle


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
        self.timeseries_labels_path = os.path.join(self.data_path, "ts_lb.csv")
        self.timeseries_analysis_path = os.path.join(self.data_path, "ts_analysis.csv")

    def read_data(self):
        """
        Read the feature and labels from train and test wave.mp4 files
        """
        train_data = []
        train_label = []

        test_data = []
        test_label = []
        test_label_normal_anomaly = []

        idx_ts = 0
        timeseries_labels_dict = {}

        # type data train and test
        for type in ["train", "test"]:
            # loop every machine in 7 machines
            for i, machine_path in enumerate(self.machines_path):

                # get the path of machine and name of each timeseries
                machine = self.machines[i]
                type_path = os.path.join(machine_path, type)
                name_ts = os.listdir(type_path)

                for name in name_ts:
                    # soure label 0, target label 1
                    if "source" in name:
                        label = 0
                    elif "target" in name:
                        label = 1

                    # path of each time series
                    name_path = os.path.join(type_path, name)

                    # get the timeseries in array
                    fs, ts = wavfile.read(name_path)

                    # save it in list
                    if type == "train":
                        train_data.append(ts)
                        train_label.append(label)
                    elif type == "test":
                        test_data.append(ts)
                        test_label.append(label)

                        # label for normal anomaly
                        if "normal" in name:
                            test_label_normal_anomaly.append(0)
                        elif "anomaly" in name:
                            test_label_normal_anomaly.append(1)

                    # get the information of each timeseries and their label
                    timeseries_labels_dict[idx_ts] = "{}_".format(
                        machine
                    ) + name.replace(".wav", "").replace("section_00_", "")
                    idx_ts = idx_ts + 1

        # save the timeseries labels
        timeseries_labels_csv = pd.DataFrame(
            list(timeseries_labels_dict.items()), columns=["Index", "Label"]
        )
        if not os.path.exists(self.timeseries_labels_path):
            timeseries_labels_csv.to_csv(self.timeseries_labels_path, index=False)

        return train_data, train_label, test_data, test_label

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
        # , test_label_normal_anomaly

        # timeseries to labels
        ts_lb = self.timeseries_labels()

        # timeseries analysis
        timeseries_analysis = {
            "train_source": [],
            "train_target": [],
            "test_source": [],
            "test_target": [],
            "test_normal": [],
            "test_anomaly": [],
            "test_source_normal": [],
            "test_source_anomaly": [],
            "test_target_normal": [],
            "test_target_anomaly": [],
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

        idx_ts = 0
        for idx_data, (type_data, type_label, windows, label_windows) in enumerate(
            data
        ):
            for ts, lb in zip(type_data, type_label):
                n_samples = len(ts)
                n_windows = (n_samples - window_size) // hop_size + 1
                for n in range(n_windows):
                    # get the windowing timeseries
                    ts_window = ts[n * hop_size : n * hop_size + window_size]
                    windows.append(ts_window)

                    # get labels
                    if idx_data == 0:
                        label_windows.append([idx_ts, lb])

                        # timeseries analysis
                        if (
                            lb == 0
                            and idx_ts not in timeseries_analysis["train_source"]
                        ):
                            timeseries_analysis["train_source"].append(idx_ts)
                        elif (
                            lb == 1
                            and idx_ts not in timeseries_analysis["train_target"]
                        ):
                            timeseries_analysis["train_target"].append(idx_ts)

                    elif idx_data == 1:
                        # get labels for normal anomaly
                        lb_na_full = ts_lb[idx_ts]
                        if "normal" in lb_na_full:
                            lb_na = 0
                        elif "anomaly" in lb_na_full:
                            lb_na = 1
                        label_windows.append([idx_ts, lb, lb_na])

                        # timeseries analysis
                        if (
                            lb == 0
                            and lb_na == 0
                            and idx_ts not in timeseries_analysis["test_source_normal"]
                        ):
                            timeseries_analysis["test_source_normal"].append(idx_ts)
                        elif (
                            lb == 0
                            and lb_na == 1
                            and idx_ts not in timeseries_analysis["test_source_anomaly"]
                        ):
                            timeseries_analysis["test_source_anomaly"].append(idx_ts)
                        elif (
                            lb == 1
                            and lb_na == 0
                            and idx_ts not in timeseries_analysis["test_target_normal"]
                        ):
                            timeseries_analysis["test_target_normal"].append(idx_ts)
                        elif (
                            lb == 1
                            and lb_na == 1
                            and idx_ts not in timeseries_analysis["test_target_anomaly"]
                        ):
                            timeseries_analysis["test_target_anomaly"].append(idx_ts)

                        elif (
                            lb == 0 and idx_ts not in timeseries_analysis["test_source"]
                        ):
                            timeseries_analysis["test_source"].append(idx_ts)
                        elif (
                            lb == 1 and idx_ts not in timeseries_analysis["test_target"]
                        ):
                            timeseries_analysis["test_target"].append(idx_ts)
                        elif (
                            lb_na == 0
                            and idx_ts not in timeseries_analysis["test_normal"]
                        ):
                            timeseries_analysis["test_normal"].append(idx_ts)
                        elif (
                            lb_na == 1
                            and idx_ts not in timeseries_analysis["test_anomaly"]
                        ):
                            timeseries_analysis["test_anomaly"].append(idx_ts)

                idx_ts = idx_ts + 1

        # save the timeseries analysis
        max_len_value = max([len(v) for v in timeseries_analysis.values()])
        for k in timeseries_analysis.keys():
            while len(timeseries_analysis[k]) < max_len_value:
                timeseries_analysis[k].append(None)

        if not os.path.exists(self.timeseries_analysis_path):
            timeseries_analysis = pd.DataFrame.from_dict(timeseries_analysis)
            timeseries_analysis.to_csv(self.timeseries_analysis_path, index=False)

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

    def timeseries_labels(self):
        """
        load ts_lb csv and conver to dict
        """
        # timeseries to labels
        if os.path.exists(self.timeseries_labels_path):

            # read ts_lb.csv
            timeseries_labels = pd.read_csv(self.timeseries_labels_path)
            timeseries_labels = timeseries_labels["Label"].to_dict()

            return timeseries_labels

        else:
            self.read_data()
            self.timeseries_labels()

    def timeseries_analysis(self):
        """
        load time series analysis from csv as dict
        """
        # timeseries analysis
        if os.path.exists(self.timeseries_analysis_path):

            # read ts analysis
            timeseries_analysis = pd.read_csv(self.timeseries_analysis_path)
            timeseries_analysis = timeseries_analysis.to_dict(orient="list")
            timeseries_analysis = {
                k: [int(x) for x in v if pd.notna(x)]
                for k, v in timeseries_analysis.items()
            }

            return timeseries_analysis

        else:
            self.windowing()
            self.timeseries_analysis


if __name__ == "__main__":

    data_name = "develop"
    data_preprocessing = DataPreprocessing(data_name=data_name)
    raw_data_path = data_preprocessing.raw_data_path
    # print("raw_data_path:", raw_data_path)
    machne_path = data_preprocessing.machines_path
    # print("machne_path:", machne_path)

    train_data, train_label, test_data, test_label = read_data = (
        data_preprocessing.read_data()
    )
    for i in read_data:
        print(len(i))

    # ts_lb = data_preprocessing.timeseries_labels()
    # print("ts_lb:", ts_lb)

    train_data, train_label, test_data, test_label = windowing = (
        data_preprocessing.windowing()
    )
    for i in windowing:
        print(i.shape)

    # print(test_label)

    ts_analysis = data_preprocessing.timeseries_analysis()
    print("ts_analysis:", ts_analysis)
    # train_data, train_label, test_data, test_label = load_data = (
    #     data_preprocessing.load_data()
    # )
    # for i in load_data:
    #     print(i.shape)
    #     if len(i.shape) == 1:
    #         print(np.unique(i, return_counts=True))

    # print(test_label)
    # print()
