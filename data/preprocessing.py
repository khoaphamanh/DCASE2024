import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import pickle

# Get the absolute path of the current file
file_path = os.path.abspath(__file__)

# Get the directory name of the file
directory_path = os.path.dirname(file_path)

# Data directory
data_dir = "10902294"
raw_data_path = os.path.join(os.path.dirname(directory_path), data_dir)


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

        # num to label csv
        self.num_to_label_path = os.path.join(self.data_path, "num_to_label.csv")
        self.unique_labels_dict_path = os.path.join(
            self.data_path, "unique_labels_dict.pkl"
        )

    def read_data(self):
        """
        Read the feature and labels from wave.mp4 files
        """
        train_data = []
        train_label = []

        test_data = []
        test_label = []

        unique_labels_dict = {"train": [], "test": []}
        unique_labels = []

        for type in ["train", "test"]:
            for i, machine_path in enumerate(self.machines_path):
                type_path = os.path.join(machine_path, type)
                name_ts = os.listdir(type_path)

                for name in name_ts:
                    name_file = name.replace(".wav", "").split("_")
                    name_file = [
                        n for idx, n in enumerate(name_file) if idx not in [0, 1, 3, 5]
                    ]

                    # get the unique labels
                    name_file.insert(0, self.machines[i])
                    name_file = "_".join(name_file)

                    if name_file not in unique_labels:
                        unique_labels.append(name_file)

                    # get data instances and labels
                    name_path = os.path.join(type_path, name)
                    fs, ts = wavfile.read(name_path)

                    if type == "train":
                        train_data.append(ts)
                        train_label.append(name_file)

                        if name_file not in unique_labels_dict["train"]:
                            unique_labels_dict["train"].append(name_file)
                    else:
                        test_data.append(ts)
                        test_label.append(name_file)

                        if name_file not in unique_labels_dict["test"]:
                            unique_labels_dict["test"].append(name_file)

        # nummerize the labels
        label_to_num = {label: num for num, label in enumerate(unique_labels)}
        train_label = [label_to_num[label] for label in train_label]
        test_label = [label_to_num[label] for label in test_label]

        # save label_to_num as csv
        num_to_label = {str(num): label for (label, num) in label_to_num.items()}
        num_to_label = pd.DataFrame(
            list(num_to_label.items()), columns=["Number", "Label"]
        )
        num_to_label.to_csv(self.num_to_label_path, index=False)

        # numerize the unique labels dict
        unique_labels_dict["train"] = [
            label_to_num[label] for label in unique_labels_dict["train"]
        ]
        unique_labels_dict["test"] = [
            label_to_num[label] for label in unique_labels_dict["test"]
        ]

        # save it unique labels dict as pkl
        with open(self.unique_labels_dict_path, "wb") as file:
            pickle.dump(unique_labels_dict, file)

        return train_data, train_label, test_data, test_label

    def create_data(self, window_size=None, hop_size=None):
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

        # windowing
        train_windows = []
        test_windows = []
        train_label_windows = []
        test_label_windows = []

        data = [
            (train_data, train_label, train_windows, train_label_windows),
            (test_data, test_label, test_windows, test_label_windows),
        ]

        for type_data, type_label, windows, label_windows in data:
            for ts, lb in zip(type_data, type_label):
                n_samples = len(ts)
                n_windows = (n_samples - window_size) // hop_size + 1
                for n in range(n_windows):
                    ts_window = ts[n * hop_size : n * hop_size + window_size]
                    windows.append(ts_window)
                    label_windows.append(lb)

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
        name_train_data = "train_data_{}_{}.npy".format(window_size, hop_size)
        name_train_label = "train_label_{}_{}.npy".format(window_size, hop_size)
        name_test_data = "test_data_{}_{}.npy".format(window_size, hop_size)
        name_test_label = "test_label_{}_{}.npy".format(window_size, hop_size)

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
            train_data, train_label, test_data, test_label = self.create_data(
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

    def load_number_to_label(self):
        """
        load num_to_label csv
        """
        # num_to_label path
        if os.path.exists(self.num_to_label_path):

            # read num_to_label csv
            num_to_label = pd.read_csv(self.num_to_label_path)
            label_dict = num_to_label["Label"].to_dict()

            return label_dict
        else:
            return None

    def load_unique_labels_dict(self):
        """
        load unique_labels_dict pkl
        """
        # unique_labels_dict path
        if os.path.exists(self.unique_labels_dict_path):

            # read the dict
            with open(self.unique_labels_dict_path, "rb") as file:
                unique_labels_dict = pickle.load(file)

        return unique_labels_dict

    def domain_to_number(self):
        """
        get the number (label) of source and target in train data
        """
        domain_dict = {}

        # get the unique labels train and test
        unique_labels_dict = self.load_unique_labels_dict()
        unique_train_labels = unique_labels_dict["train"]
        unique_test_labels = unique_labels_dict["test"]

        # sort domain from number to label
        label_dict = self.load_number_to_label()
        if label_dict is not None:
            domain_dict["train_source"] = [
                n
                for n, l in label_dict.items()
                if "source" in l and n in unique_train_labels
            ]
            domain_dict["train_target"] = [
                n
                for n, l in label_dict.items()
                if "target" in l and n in unique_train_labels
            ]
            domain_dict["test_normal"] = [
                n
                for n, l in label_dict.items()
                if "normal" in l and n in unique_test_labels
            ]
            domain_dict["test_anomaly"] = [
                n
                for n, l in label_dict.items()
                if "anomaly" in l and n in unique_test_labels
            ]
            return domain_dict

        else:
            return None


if __name__ == "__main__":

    from timeit import default_timer

    start = default_timer()

    data_name = "develop"
    data_preprocessing = DataPreprocessing(data_name=data_name)

    # data_path = data_preprocessing.data_path
    # print("data_path:", data_path)
    # machines = data_preprocessing.machines
    # print("machines:", machines)
    # machines_path = data_preprocessing.machines_path
    # print("machines_path:", machines_path)
    # attribute_file_path = data_preprocessing.attribute_file_path
    # print("attribute_file_path:", attribute_file_path)
    # duration = data_preprocessing.duration
    # print("duration:", duration)
    # len_ts = data_preprocessing.len_ts
    # print("len_ts:", len_ts)

    # load_data = data_preprocessing.create_data()
    # print("load_data:", load_data)
    # label_to_num = data_preprocessing.read_data()

    fs = 16000
    train_data, train_label, test_data, test_label = data_preprocessing.load_data(
        window_size=None, hop_size=None
    )
    print("train_label:", train_label)

    print("train_data shape:", train_data.shape)
    unique_train = np.unique(train_label)
    print("unique_train len:", len(unique_train))
    print("unique_train:", unique_train)
    print()
    unique_test = np.unique(test_label)
    print("unique_test len:", len(unique_test))
    print("unique_test:", unique_test)

    data_name = data_preprocessing.data_name
    print("data_name:", data_name)
    out = data_preprocessing.domain_to_number()
    print("out:", out)

    train_source = out["train_source"]
    print("source_train len:", len(train_source))
    target_train = out["train_target"]
    print("target_train len:", len(target_train))

    anomaly_test = out["test_anomaly"]
    print("anomaly_test len:", len(anomaly_test))
    normal_test = out["test_normal"]
    print("normal_test len:", len(normal_test))

    end = default_timer()

    print("run time", end - start)
