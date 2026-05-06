import numpy as np
import scipy.io
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader


class LaserData:
    def __init__(self, path="data/Xtrain.mat", split_ratio = 0.8, sequence_length = 20, scaler = None, mode="train", key = "Xtrain"):
        data = scipy.io.loadmat(path)
        # '__header__', '__version__', '__globals__', 'Xtrain'
        laser_data = data[key]
        self.raw_data = laser_data.astype(np.float32)

        self.split_ratio = split_ratio
        self.sequence_length = sequence_length
        self.mode = mode

        if self.mode == "test" and scaler is None:
            raise ValueError("test mode requires a fitted scaler from training.")

        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()

        self.train_raw = None
        self.val_raw = None

        self.train_sequences = None
        self.train_labels = None
        self.val_sequences = None
        self.val_labels = None

        self._prepare()
    # ----------- Internal Functions -------------
    def _prepare(self):
        if self.mode == "train":
            self._split_train_val()
            self.train_sequences, self.train_labels = self._create_sequence(self.train_raw)
            self.val_sequences, self.val_labels = self._create_sequence(self.val_raw)
        elif self.mode == "test":
            self.test_raw = self.scaler.transform(self.raw_data)
            self.test_sequences, self.test_labels = self._create_sequence(self.test_raw)

    def _split_train_val(self):
        split_idx = int(self.split_ratio * len(self.raw_data))
        train = self.raw_data[:split_idx]
        val = self.raw_data[split_idx:]

        # fit scaler on training data then apply it to validation
        # because we only know the training data, not the validation (unseen)
        self.scaler.fit(train)

        self.train_raw = self.scaler.transform(train)
        self.val_raw = self.scaler.transform(val)

    def _create_sequence(self, data):
        X, y = [], []
        data_length = len(data)
        seq_len = self.sequence_length
        for i in range(data_length - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return (torch.tensor(np.array(X), dtype=torch.float32),
                torch.tensor(np.array(y), dtype=torch.float32))

    # -------------- APIs ----------------

    def get_data(self):
        if self.mode == "train":
            return self.train_sequences, self.train_labels, self.val_sequences, self.val_labels
        else:
            return self.test_sequences, self.test_labels

    def get_scaler(self):
        return self.scaler

    def get_loaders(self, batch_size=32):
        if self.mode == "train":
            train_loader = DataLoader(
                TensorDataset(self.train_sequences, self.train_labels),
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = DataLoader(
                TensorDataset(self.val_sequences, self.val_labels),
                batch_size=batch_size,
                shuffle=False
            )
            return train_loader, val_loader
        else:
            test_loader = DataLoader(
                TensorDataset(self.test_sequences, self.test_labels),
                batch_size=batch_size,
                shuffle=False
            )
            return test_loader





if __name__ == "__main__":
    dataset = LaserData("data/Xtrain.mat")
    train_loader, val_loader = dataset.get_loaders()
    X_train, y_train = train_loader.dataset.tensors

    print(X_train.shape)
    print(y_train.shape)
    # ouput:
    # torch.Size([780, 20, 1])
    # torch.Size([780, 1])