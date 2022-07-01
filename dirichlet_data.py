import random
import numpy as np
from PIL import Image
import os.path as osp
import pandas as pd
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from cnn_lab.autoaugment import Cutout, CIFAR10Policy

SICAPV2_PATH = "/data/BasesDeDatos/SICAP/SICAPv2/"
PANDA_PATH = "/data/BasesDeDatos/Panda/Panda_patches_resized/"

MILANOMA_PATH = ""

MINI_PANDA_PATH = "/home/jiahui/data/minipanda"
MINI_SICAPV2_PATH = "/home/jiahui/data/minisicap"
panda_stats = {"norm_mean":  (0.4914, 0.4822, 0.4465), "norm_std": (0.2023, 0.1994, 0.2010)}

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        # self.data = [self.dataset[idx] for idx in self.idxs]
        # self.targets = [self.dataset.targets[idx] for idx in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img, label = self.dataset[self.idxs[item]]
        return img, label #self.data[item], self.targets[item]

class MelanomaDataset(Dataset):
    def __init__(self, df, meta_features=None, transform=None):
        # self.csv = csv.reset_index(drop=True)
        self.data_df = df
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform
        self.data = []
        onehot_targets = self.data_df[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].values
        self.targets = np.argmax(onehot_targets, axis=1)


    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.data_df.iloc[index]

        image = Image.open(osp.join(MILANOMA_PATH, row.image+".jpg"))
        if self.transform is not None:
            image = self.transform(image)

        # target = np.array(self.data_df.loc[index, ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']])
        # target = np.argmax(target)
        target = self.targets[index]
        if self.use_meta:
            data = (image, torch.tensor(self.data_df.iloc[index][self.meta_features]).float())
        else:
            data = image
        return data, target

class PandaDatast(Dataset):
    def __init__(self, df):
        # self.csv = csv.reset_index(drop=True)
        self.data_df = df
        self.transform = transforms.Compose([transforms.Resize((32,32)),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomRotation(degrees=60),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(panda_stats["norm_mean"], panda_stats["norm_std"])
                                             ])
        self.data = []
        # onehot_targets = self.data_df[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].values
        onehot_targets = self.data_df[['NC','G3','G4','G5','unlabeled']].values
        self.targets = np.argmax(onehot_targets, axis=1)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.data_df.iloc[index]

        image = Image.open(osp.join(PANDA_PATH, "images", row["image_name"]))
        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]

        return image, target

class MiniPandaDatast(Dataset):
    def __init__(self, df):
        # self.csv = csv.reset_index(drop=True)
        self.data_df = df
        self.transform = transforms.Compose([transforms.Resize((32,32)),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomRotation(degrees=60),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(panda_stats["norm_mean"], panda_stats["norm_std"])
                                             ])
        self.data = []
        # onehot_targets = self.data_df[['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']].values
        onehot_targets = self.data_df[['NC','G3','G4','G5','unlabeled']].values
        self.targets = np.argmax(onehot_targets, axis=1)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        row = self.data_df.iloc[index]

        image = Image.open(osp.join(MINI_PANDA_PATH, "images", row["image_name"]))
        if self.transform is not None:
            image = self.transform(image)

        target = self.targets[index]

        return image, target

def get_client_alpha(train_set_group):
    client_n_sample = [len(ts.idxs) for ts in train_set_group]
    total_n_sample = sum(client_n_sample)
    client_alpha = [n_sample / total_n_sample for n_sample in client_n_sample]
    # print(f'alpha = {client_alpha}')
    return client_alpha

def dirichlet_data(data_name, num_users=10, alpha = 100):

    if data_name == 'mnist':
        dataset = datasets.MNIST('./data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        test_dataset = datasets.MNIST('./data/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


    elif data_name == 'cifar10':
        dataset = datasets.CIFAR10('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR10('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))
    elif data_name == 'cifar100':

        dataset = datasets.CIFAR100('./data/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(32, padding=4, fill=128),
                                       transforms.RandomHorizontalFlip(),
                                       CIFAR10Policy(),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=16),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ]))

        test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ]))


    elif data_name == "panda":
        train_df_raw = pd.read_csv(osp.join(PANDA_PATH, "train_patches.csv"))
        test_df_raw = pd.read_csv(osp.join(PANDA_PATH, "test_patches.csv"))
        dataset = PandaDatast(train_df_raw)
        test_dataset = PandaDatast(test_df_raw)

    elif data_name == "minipanda":
        train_df_raw = pd.read_csv(osp.join(MINI_PANDA_PATH, "mini_train_patches.csv"))
        test_df_raw = pd.read_csv(osp.join(MINI_PANDA_PATH, "mini_test_patches.csv"))

        dataset = MiniPandaDatast(train_df_raw)
        test_dataset = MiniPandaDatast(train_df_raw)

    elif data_name == "sicapv2":
        dataset = None

        test_dataset = None

    elif data_name == "melanoma":
        dataset = None

        test_dataset = None

    else:
        print ('Data name error')
        return None


    class_num = 10

    dict_users = {i: np.array([]) for i in range(num_users)}

    idxs = np.arange(len(dataset.targets))
    labels = np.asarray(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    class_lableidx = [idxs_labels[:, idxs_labels[1, :] == i][0, :] for i in range(class_num)]

    sample_matrix = np.random.dirichlet([alpha for _ in range(num_users)], class_num).T
    class_sampe_start = [0 for i in range(class_num)]

    def sample_rand(rand, class_sampe_start):
        class_sampe_end = [start + int(len(class_lableidx[sidx]) * rand[sidx]) for sidx, start in enumerate(class_sampe_start)]
        rand_set = np.array([])
        for eidx, rand_end in enumerate(class_sampe_end):
            rand_start = class_sampe_start[eidx]
            if rand_end<= len(class_lableidx[eidx]):
                rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:rand_end]], axis=0)

            else:
                if rand_start< len(class_lableidx[eidx]):
                    rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:]],axis=0)
                else:
                    rand_set=np.concatenate([rand_set,random.sample(class_lableidx[eidx] , rand_end - rand_start +1)],axis=0)
        if rand_set.shape[0] == 0:
            rand_set = np.concatenate([rand_set, class_lableidx[0][0:1]], axis=0)
        return rand_set, class_sampe_end

    for i in range(num_users):
        rand_set, class_sampe_start = sample_rand(sample_matrix[i], class_sampe_start)
        dict_users[i] = rand_set

    return [DatasetSplit(deepcopy(dataset), dict_users[i]) for i in range(num_users)], test_dataset




if __name__ == '__main__':
    pass













