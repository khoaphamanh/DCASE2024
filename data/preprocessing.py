import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import librosa
import sys
from tqdm import tqdm
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from imblearn.over_sampling import SMOTE
import random

# add path from data preprocessing in data directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# data preprocessing
class DataPreprocessing:
    def __init__(self, data_name, seed=2024):

        # set the seed
        np.random.seed(seed)
        random.seed(seed)

        # directory information
        self.seed = seed
        self.data_name = data_name
        self.path_preprocessing = os.path.abspath(__file__)
        self.path_data_directory = os.path.dirname(self.path_preprocessing)
        self.path_main_directory = os.path.dirname(self.path_data_directory)

        # data information
        self.path_data_name_directory = os.path.join(
            self.path_main_directory, "data", data_name
        )
        print(self.path_data_name_directory)

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
        self.domain_data = ["source", "target"]

        # timeseries information
        self.fs = 16000
        if data_name == "develop":
            self.machine_no_attribute = ["slider", "gearbox", "ToyTrain"]
            self.duration = {
                tuple(i for i in self.machines if i not in ["ToyCar", "ToyTrain"]): 10,
                ("ToyCar", "ToyTrain"): 12,
            }
            self.len_ts_max = self.fs * max(self.duration.values())
            self.len_ts_min = self.fs * min(self.duration.values())

        self.label_condition_number = {"normal": 0, "anomaly": 1}

        # path data information
        self.path_label_unique = os.path.join(
            self.path_data_name_directory,
            "{}_label_unique.csv".format(self.data_name),
        )
        self.path_data_timeseries_information = os.path.join(
            self.path_data_name_directory,
            "{}_data_timeseries_information.csv".format(self.data_name),
        )
        self.path_indices_timeseries_analysis = os.path.join(
            self.path_data_name_directory,
            "{}_indices_timeseries_analysis.csv".format(self.data_name),
        )

        # path raw data
        self.path_train_data = os.path.join(
            self.path_data_name_directory,
            "{}_train_data.npy".format(self.data_name),
        )

        self.path_train_label = os.path.join(
            self.path_data_name_directory,
            "{}_train_label.npy".format(self.data_name),
        )

        self.path_test_data = os.path.join(
            self.path_data_name_directory,
            "{}_test_data.npy".format(self.data_name),
        )

        self.path_test_label = os.path.join(
            self.path_data_name_directory,
            "{}_test_label.npy".format(self.data_name),
        )

        # path data smote
        self.path_train_data_smote = os.path.join(
            self.path_data_name_directory, "{}_train_data_smote.npy".format(data_name)
        )
        self.path_train_label_smote = os.path.join(
            self.path_data_name_directory, "{}_train_label_smote.npy".format(data_name)
        )

    def read_raw_data(self):
        """
        Read features, labels from .wav files
        """

        # train test data list
        train_data = []
        train_label = []
        test_data = []
        test_label = []

        # dict for data timeseries information with syntax [index,name_ts,label_attributt, label_condition ]
        timeseries_information = []

        # list of unique labels
        label_unique = []

        # loop type train test
        idx_ts = 0  # index of the each timeseries, from 0 to 8399 (total 8400)

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
                    timeseries_information.append(
                        [
                            idx_ts,
                            path_n_ts,
                            label_attribute_number,
                            label_condition_number,
                            len(ts),
                        ]
                    )

                    # pad ts to have 12 seconds
                    if len(ts) > self.len_ts_min:
                        index_start = np.random.randint(
                            0, len(ts) - self.len_ts_min + 1
                        )
                        print("index_start:", index_start)
                        ts = ts[index_start : index_start + self.len_ts_min]

                    # append to data list
                    label = [idx_ts, label_attribute_number, label_condition_number]
                    if t == "train":
                        train_data.append(ts)
                        train_label.append(label)
                    elif t == "test":
                        test_data.append(ts)
                        test_label.append(label)

                    # next index
                    idx_ts = idx_ts + 1

        # save train test data list
        np.save(self.path_train_data, train_data)
        np.save(self.path_train_label, train_label)
        np.save(self.path_test_data, test_data)
        np.save(self.path_test_label, test_label)

        # convert to csv
        label_unique = [[i, label_unique[i]] for i in range(len(label_unique))]
        column_names = ["Number", "Attribute"]
        label_unique = pd.DataFrame(label_unique, columns=column_names)

        column_names = ["Index", "Name", "Attribute", "Condition", "Length"]
        timeseries_information = pd.DataFrame(
            timeseries_information, columns=column_names
        )

        # save the data frame to data name directory
        label_unique.to_csv(self.path_label_unique, index=False)
        timeseries_information.to_csv(
            self.path_data_timeseries_information, index=False
        )

    def label_unique(self):
        """
        load label unique as csv, label number and their name
        """
        if not os.path.exists(self.path_label_unique):
            self.read_raw_data()

        label_unique = pd.read_csv(self.path_label_unique)

        return label_unique

    def timeseries_information(self):
        """
        load data_timeseries_information as csv, index and the path of each timeseries in dataset
        """
        if not os.path.exists(self.path_data_timeseries_information):
            self.read_raw_data()

        data_timeseries_information = pd.read_csv(self.path_data_timeseries_information)

        return data_timeseries_information

    def indices_timeseries_analysis(self):
        """
        create the csv that analysis type_machine_domain_condition the index of each time series
        """
        # get the information from timeseries_information
        timeseries_information = self.timeseries_information().to_numpy()
        indices_timeseries = timeseries_information[:, 0]
        name_timeseries = timeseries_information[:, 1]
        condition_timeseries = timeseries_information[:, 3]

        # dict indices_timeseries_analysis
        indices_timeseries_analysis = {}

        # check if exsist
        if not os.path.exists(self.path_indices_timeseries_analysis):
            for t in self.type_data:
                for d in self.domain_data:
                    for c in self.label_condition_number.keys():

                        # condition as number
                        c_number = self.label_condition_number[c]

                        # key of the dict indices_timeseries_analyis
                        key = "{}_{}_{}".format(t, d, c)
                        indices_timeseries_analysis[key] = [
                            i
                            for i in indices_timeseries
                            if t in name_timeseries[i]
                            and d in name_timeseries[i]
                            and c_number == condition_timeseries[i]
                        ]

                        for m in self.machines:
                            if t == "test":
                                # key of the dict indices_timeseries_analyis
                                key = "{}_{}_{}".format(t, m, d)
                                indices_timeseries_analysis[key] = [
                                    i
                                    for i in indices_timeseries
                                    if t in name_timeseries[i]
                                    and m in name_timeseries[i]
                                    and d in name_timeseries[i]
                                ]

            # delete len keys == 0
            maxlen_value = max([len(i) for i in indices_timeseries_analysis.values()])

            # filter out keys with empty lists values
            indices_timeseries_analysis = {
                k: v for k, v in indices_timeseries_analysis.items() if len(v) > 0
            }

            for k, v in indices_timeseries_analysis.items():
                while len(v) < maxlen_value:
                    v.append(None)
                indices_timeseries_analysis[k] = v

            # convert to data frame and save it as .csv
            indices_timeseries_analysis = pd.DataFrame(indices_timeseries_analysis)
            indices_timeseries_analysis.to_csv(
                self.path_indices_timeseries_analysis, index=False
            )

        else:
            indices_timeseries_analysis = pd.read_csv(
                self.path_indices_timeseries_analysis
            )
        return indices_timeseries_analysis

    def load_raw_data(self):
        """
        load data .pkl file as list
        """
        check_data_exists = [
            os.path.exists(self.path_train_data),
            os.path.exists(self.path_train_label),
            os.path.exists(self.path_test_data),
            os.path.exists(self.path_test_label),
        ]
        if not all(check_data_exists):
            self.read_raw_data()

        train_data, train_label, test_data, test_label = [
            np.load(self.path_train_data),
            np.load(self.path_train_label),
            np.load(self.path_test_data),
            np.load(self.path_test_label),
        ]

        return (
            train_data.astype(np.float32),
            train_label,
            test_data.astype(np.float32),
            test_label,
        )

    def load_data_attribute(self):
        """
        load data with attribute as label (some bearing labels in test data that not in train data will not included
        """
        # load raw data
        train_data, train_label, test_data, test_label = self.load_raw_data()

        # find the unique label in train data
        label_train_attribute = train_label[:, 1]
        label_train_attribute_unique = np.unique(label_train_attribute)
        label_test_attribute = test_label[:, 1]
        index_label_unique_attribute_in_test = [
            i
            for i in range(len(test_data))
            if label_test_attribute[i] in label_train_attribute_unique
        ]

        test_data = test_data[index_label_unique_attribute_in_test]
        test_label = test_label[index_label_unique_attribute_in_test]

        return train_data, train_label, test_data, test_label

    def num_classes_attribute(self):
        """
        number of classes attribute for training
        """
        _, train_label, _, _ = self.load_data_attribute()

        return len(np.unique(train_label[:, 1]))

    def augmentation(self, train_data_aug, train_label_aug, k_smote=5):
        """
        function to do augmentation for the instances that have fewer than k_smote
        choose a random instance from a label, do one augmentation, keep doing like this until reach the k_smote+1 instances
        """

        # augmentation method
        augmentation = Compose(
            [AddGaussianNoise(), TimeStretch(), PitchShift(), Shift()]
        )

        # each label will have k_smote + 1 instances
        train_data_aug_smote = []
        train_label_aug_smote = []

        # get the unique and counts of the train_data_aug and train_label_aug
        label_attribute_aug_unique, label_attribute_aug_counts = np.unique(
            train_label_aug, return_counts=True
        )

        for label_unique, label_counts in zip(
            label_attribute_aug_unique, label_attribute_aug_counts
        ):

            # total number of ts
            total_number_ts = label_counts

            # indices for each label
            indices = np.where(train_label_aug == label_unique)[0]
            print(label_unique)
            print("indices:", indices)

            while total_number_ts <= k_smote:

                # get the random index for each label
                index_random = np.random.choice(indices)
                print("index_random:", index_random)

                # get the ts
                ts_original = train_data_aug[index_random]

                # augmented ts
                ts_augmented = augmentation(ts_original, sample_rate=self.fs)
                # print("ts_augmented:", ts_augmented[0:10])

                # increase the total_ts_number
                total_number_ts = total_number_ts + 1

                # append to augmentation list
                train_data_aug_smote.append(ts_augmented)
                train_label_aug_smote.append(label_unique)

        print()
        # stack all of them
        train_data_aug = np.vstack((train_data_aug, train_data_aug_smote))
        train_label_aug = np.concatenate((train_label_aug, train_label_aug_smote))

        return train_data_aug, train_label_aug

    def smote(self, k_smote=5):
        """
        load data smote
        """
        # check if data smote exists
        check_exsist = [
            os.path.exists(self.path_train_data_smote),
            os.path.exists(self.path_train_label_smote),
        ]

        if not all(check_exsist):
            # load data attribute
            train_data, train_label, test_data, test_label = self.load_data_attribute()

            # find the unique labels and their counts
            label_train_attribute = train_label[:, 1]
            label_train_attribute_unique, label_train_attribute_counts = np.unique(
                label_train_attribute, return_counts=True
            )
            # print("label_train_unique:", label_train_attribute_unique)
            # print("label_train_counts:", label_train_attribute_counts)

            # sort data and labels with fewer or more than k_smote
            train_data_smote = []
            train_label_smote = []
            train_data_aug = []
            train_label_aug = []

            for i in range(len(train_data)):
                if label_train_attribute_counts[label_train_attribute[i]] > k_smote:
                    train_data_smote.append(train_data[i])
                    train_label_smote.append(label_train_attribute[i])
                else:
                    train_data_aug.append(train_data[i])
                    train_label_aug.append(label_train_attribute[i])

            # augmentation for unique label that fewer than k_smote
            train_data_aug, train_label_aug = self.augmentation(
                train_data_aug=train_data_aug,
                train_label_aug=train_label_aug,
                k_smote=k_smote,
            )

            # print("train_data_smote shape:", np.array(train_data_smote).shape)
            # print("train_label_smote shape:", np.array(train_label_smote).shape)

            # print("train_data_aug shape:", train_data_aug.shape)
            # print("train_label_aug shape:", train_label_aug.shape)

            # stack the augmentation and instance has more than k_smote instance
            train_data_smote = np.vstack((train_data_smote, train_data_aug))
            # print("train_data_smote shape:", train_data_smote.shape)
            train_label_smote = np.concatenate((train_label_smote, train_label_aug))
            # print("train_label_smote shape:", train_label_smote.shape)

            # label_train_attribute_unique, label_train_attribute_counts = np.unique(
            #     train_label_smote, return_counts=True
            # )

            # print("label_train_unique:", label_train_attribute_unique)
            # print("label_train_counts:", label_train_attribute_counts)

            # applied smote to data
            smote = SMOTE(random_state=self.seed, k_neighbors=k_smote)
            train_data_smote, train_label_smote = smote.fit_resample(
                train_data_smote,
                train_label_smote,
            )

            # save the data
            np.save(self.path_train_data_smote, train_data_smote)
            np.save(self.path_train_label_smote, train_label_smote)

        else:
            train_data_smote = np.load(self.path_train_data_smote)
            train_label_smote = np.load(self.path_train_label_smote)

        return train_data_smote, train_label_smote

    def log_melspectrogram(
        self,
        data,
        window_size=400,
        hop_size=160,
        n_mels=128,
        dB=True,
    ):
        """
        convert data to log melspectrogram
        """
        data_logmel = []
        for ts in data:
            ms = librosa.feature.melspectrogram(
                y=ts, sr=self.fs, n_fft=window_size, hop_length=hop_size, n_mels=n_mels
            )
            if dB:
                ms = librosa.power_to_db(ms)
            data_logmel.append(ms)

        data_logmel = np.array(data_logmel)
        return data_logmel


if __name__ == "__main__":

    from timeit import default_timer

    seed = 1998

    # # set the seed
    # np.random.seed(seed)
    # random.seed(seed)

    start = default_timer()
    print()
    data_name = "develop"
    data_preprocessing = DataPreprocessing(data_name=data_name, seed=seed)

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

    # label_unique = data_preprocessing.label_unique()
    # print("label_unique:", label_unique)

    # data_timeseries_information = data_preprocessing.timeseries_information()
    # print("data_timeseries_information:", data_timeseries_information)

    # path_label_unique = data_preprocessing.path_label_unique
    # print("path_label_unique:", path_label_unique)

    # path_train_data = data_preprocessing.path_train_data
    # print("path_train_data:", path_train_data)

    # train_data, train_label, test_data, test_label = data_preprocessing.load_data()
    # print("train_data:", train_data.shape)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("test_label:", test_label.shape)

    # data_logmel = data_preprocessing.log_melspectrogram(data=train_data)
    # print("data_logmel shape:", data_logmel.shape)

    # indices_timeseries_analyis = data_preprocessing.indices_timeseries_analyis()
    # print("indices_timeseries_analyis:", indices_timeseries_analyis)
    # print("indices_timeseries_analyis keys:", indices_timeseries_analyis.keys())

    # train_data, train_label, test_data, test_label = data_preprocessing.load_data_attritbute()
    # print("train_data:", train_data.shape)
    # print("train_label:", train_label.shape)
    # print("test_data:", test_data.shape)
    # print("test_label:", test_label.shape)

    # num_classes_attribute = data_preprocessing.num_classes_attribute()

    # train_data_smote, train_label_smote = data_preprocessing.smote()
    # print("train_data_smote shape:", train_data_smote.shape)
    # print("train_label_smote:", train_label_smote.shape)

    # label_train_attribute_unique, label_train_attribute_counts = np.unique(
    #     train_label_smote, return_counts=True
    # )

    # print("label_train_unique:", label_train_attribute_unique)
    # print("label_train_counts:", label_train_attribute_counts)

    num_classes_attribute = data_preprocessing.num_classes_attribute()
    print("num_classes_attribute:", num_classes_attribute)

    end = default_timer()
    print(end - start)
