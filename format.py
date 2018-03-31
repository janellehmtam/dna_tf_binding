import torch, os
import numpy as np
from sklearn.model_selection import train_test_split

base2index = {
    'A': 0,
    'T': 1,
    'G': 2,
    'C': 3,
}


def load_tf_data(exp_name, batch_size=128):

    train_set = os.path.join('data/', exp_name, "train.fa")
    test_set = os.path.join('data/', exp_name, "test.fa")

    # Training File #

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    with open(train_set) as dfile:

        while True:
            tfdata = np.zeros([1, 101, 4], dtype=np.dtype('float32'))
            line1 = dfile.readline()
            line2 = dfile.readline()
            if not line2:
                break

            seq = ''.join(line2.split())

            for i, char in enumerate(seq):
                tfdata[0][i][base2index[char]] = 1.0

            train_data.append(tfdata)

            if ''.join(line1.split()) == ">1":
                train_labels.append(1)
            else:
                train_labels.append(0)

    n_train = len(train_data)
    valid_size = 0.2  # % train set for validation
    split = int(valid_size * n_train)

    val_data = train_data[:split]
    train_data = train_data[split:]

    val_labels = train_labels[:split]
    train_labels = train_labels[split:]

    train_data = torch.from_numpy(np.array(train_data)).float()
    train_labels = torch.from_numpy(np.array(train_labels)).long()

    val_data = torch.from_numpy(np.array(val_data)).float()
    val_labels = torch.from_numpy(np.array(val_labels)).long()

    train_dataset = list(zip(train_data, train_labels))
    val_dataset = list(zip(val_data, val_labels))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # Test File #

    with open(test_set) as dfile:

        while True:
            tfdata = np.zeros([1, 101, 4], dtype=np.dtype('float32'))
            line1 = dfile.readline()
            line2 = dfile.readline()
            if not line2:
                break

            seq = ''.join(line2.split())

            for i, char in enumerate(seq):
                tfdata[0][i][base2index[char]] = 1.0

            test_data.append(tfdata)

            if "".join(line1.split()) == ">1":
                test_labels.append(1)
            else:
                test_labels.append(0)

    test_data = torch.from_numpy(np.array(test_data)).float()
    test_labels = torch.from_numpy(np.array(test_labels)).long()

    test_dataset = list(zip(test_data, test_labels))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


