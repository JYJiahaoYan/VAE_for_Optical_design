import pandas as pd
import numpy as np
import pickle as pkl
import os
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader ,random_split
import scipy.io     # used to load .mat data

def train_valid_split(data_set, valid_ratio,test_ratio, seed):
    '''Split provided training data into training set and validation set'''
    '''将数据集划分为train_data,valid_data,test_data'''
    valid_set_size = int(valid_ratio * len(data_set))
    test_set_size = int(test_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size - test_set_size

    train_set, valid_set, test_set = random_split(data_set, [train_set_size, valid_set_size, test_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set), np.array(test_set)

class SiliconColor(Dataset):
    
    def __init__(self, filepath, split='train', inverse=False):
        super(SiliconColor).__init__()
        # temp = scipy.io.loadmat(filepath)
        temp = pd.read_csv(filepath)
        # self.data = np.array(list(temp.items())[3][1])
        temp = temp[["W","L","a","H","Px","Py","R","G","B","BIC"]]
        # temp = temp[["W","L","a","H","Px","Py","min_distance",'max_distance',"R","G","B","BIC"]]
        self.data = temp.to_numpy()
        x = self.data[:,:6]
        y = self.data[:,6:]
        # x = self.data[:,:8]
        # y = self.data[:,8:]
        if inverse:
            x, y = y, x
        self.data = np.hstack((x, y))

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)

        range_, min_ = self.scaler.data_range_, self.scaler.data_min_

        train_data, valid_data, test_data = train_valid_split(self.data, 0.1, 0.1, 418)


        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = valid_data
        else:
            self.data = test_data

        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data).float()

        if inverse:
            self.x, self.y = self.data[:, :4], self.data[:, 4:]
            # self.x, self.y = self.data[:, :3], self.data[:, 3:]
        else:
            self.x, self.y = self.data[:, :6], self.data[:, 6:]
            # self.x, self.y = self.data[:, :8], self.data[:, 8:]


        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_scaler(self):
        return self.x_scaler, self.y_scaler


def get_dataloaders(model, batch_=128):
    parentpath = os.getcwd()
    #datapath = parentpath + '/Meta_learning_photonics_structure/Model/data/RCWA_xyY_all.mat'
    datapath = './data/train_0904_peak.csv'
    # datapath = './data/train_0912_peak.csv'
    if model in ['forward_model', 'tandem_net','vae', 'inn', 'vae_new', 'vae_GSNN', 'vae_Full','vae_tandem', 'vae_hybrid']:
        train_dt = SiliconColor(datapath, 'train')
        val_dt = SiliconColor(datapath, 'val')
        test_dt = SiliconColor(datapath, 'test')

    else:
        train_dt = SiliconColor(datapath, 'train', inverse = True)
        val_dt = SiliconColor(datapath, 'val', inverse = True)
        test_dt = SiliconColor(datapath, 'test', inverse = True)
        
    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader


class Color(Dataset):

    def __init__(self, filepath, split='train', inverse=False):
        super(Color).__init__()
        # temp = scipy.io.loadmat(filepath)
        temp = pd.read_csv(filepath)
        # self.data = np.array(list(temp.items())[3][1])
        temp = temp[["W", "L", "a", "H", "Px", "Py", "R", "G", "B"]]
        # temp = temp[["W","L","a","H","Px","Py","min_distance",'max_distance',"R","G","B","BIC"]]
        self.data = temp.to_numpy()
        x = self.data[:, :6]
        y = self.data[:, 6:]
        # x = self.data[:,:8]
        # y = self.data[:,8:]
        if inverse:
            x, y = y, x
        self.data = np.hstack((x, y))

        # self.x_scaler = MinMaxScaler()
        # self.y_scaler = MinMaxScaler()
        # self.x_scaler.fit(x)
        # self.y_scaler.fit(y)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)

        range_, min_ = self.scaler.data_range_, self.scaler.data_min_

        train_data, valid_data, test_data = train_valid_split(self.data, 0.1, 0.1, 418)

        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = valid_data
        else:
            self.data = test_data

        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data).float()

        if inverse:
            self.x, self.y = self.data[:, :3], self.data[:, 3:]
            # self.x, self.y = self.data[:, :3], self.data[:, 3:]
        else:
            self.x, self.y = self.data[:, :6], self.data[:, 6:]
            # self.x, self.y = self.data[:, :8], self.data[:, 8:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_scaler(self):
        return self.x_scaler, self.y_scaler


def get_color_dataloaders(model, batch_=128):
    parentpath = os.getcwd()
    # datapath = parentpath + '/Meta_learning_photonics_structure/Model/data/RCWA_xyY_all.mat'
    datapath = './data/train_0819_color.csv'
    # datapath = './data/train_0912_peak.csv'
    if model in ['forward_model', 'tandem_net', 'vae', 'inn', 'vae_new', 'vae_GSNN', 'vae_Full', 'vae_tandem',
                 'vae_hybrid']:
        train_dt = Color(datapath, 'train')
        val_dt = Color(datapath, 'val')
        test_dt = Color(datapath, 'test')

    else:
        train_dt = Color(datapath, 'train', inverse=True)
        val_dt = Color(datapath, 'val', inverse=True)
        test_dt = Color(datapath, 'test', inverse=True)

    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader

class BIC(Dataset):

    def __init__(self, filepath, split='train', inverse=False):
        super(BIC).__init__()
        # temp = scipy.io.loadmat(filepath)
        temp = pd.read_csv(filepath)
        # self.data = np.array(list(temp.items())[3][1])
        temp = temp[["W", "L", "a", "H", "Px", "Py", "BIC"]]
        # temp = temp[["W","L","a","H","Px","Py","min_distance",'max_distance',"R","G","B","BIC"]]
        self.data = temp.to_numpy()
        x = self.data[:, :6]
        y = self.data[:, 6:]
        # x = self.data[:,:8]
        # y = self.data[:,8:]
        if inverse:
            x, y = y, x
        self.data = np.hstack((x, y))

        # self.x_scaler = MinMaxScaler()
        # self.y_scaler = MinMaxScaler()
        # self.x_scaler.fit(x)
        # self.y_scaler.fit(y)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)

        range_, min_ = self.scaler.data_range_, self.scaler.data_min_

        train_data, valid_data, test_data = train_valid_split(self.data, 0.1, 0.1, 418)

        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = valid_data
        else:
            self.data = test_data

        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data).float()

        if inverse:
            self.x, self.y = self.data[:, :1], self.data[:, 1:]
            # self.x, self.y = self.data[:, :3], self.data[:, 3:]
        else:
            self.x, self.y = self.data[:, :6], self.data[:, 6:]
            # self.x, self.y = self.data[:, :8], self.data[:, 8:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_scaler(self):
        return self.x_scaler, self.y_scaler


def get_BIC_dataloaders(model, batch_=128):
    parentpath = os.getcwd()
    # datapath = parentpath + '/Meta_learning_photonics_structure/Model/data/RCWA_xyY_all.mat'
    datapath = './data/train_0904_peak.csv'
    # datapath = './data/train_0912_peak.csv'
    if model in ['forward_model', 'tandem_net', 'vae', 'inn', 'vae_new', 'vae_GSNN', 'vae_Full', 'vae_tandem',
                 'vae_hybrid']:
        train_dt = BIC(datapath, 'train')
        val_dt = BIC(datapath, 'val')
        test_dt = BIC(datapath, 'test')

    else:
        train_dt = BIC(datapath, 'train', inverse=True)
        val_dt = BIC(datapath, 'val', inverse=True)
        test_dt = BIC(datapath, 'test', inverse=True)

    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader

class Binary(Dataset):

    def __init__(self, filepath, split='train', inverse=False):
        super(Binary).__init__()
        # temp = scipy.io.loadmat(filepath)
        temp = pd.read_csv(filepath)
        # self.data = np.array(list(temp.items())[3][1])
        temp = temp[["W", "L", "a", "H", "Px", "Py", "Label"]]
        # temp = temp[["W","L","a","H","Px","Py","min_distance",'max_distance',"R","G","B","BIC"]]
        self.data = temp.to_numpy()
        x = self.data[:, :6]
        y = self.data[:, 6:]
        # x = self.data[:,:8]
        # y = self.data[:,8:]
        if inverse:
            x, y = y, x
        self.data = np.hstack((x, y))

        # self.x_scaler = MinMaxScaler()
        # self.y_scaler = MinMaxScaler()
        # self.x_scaler.fit(x)
        # self.y_scaler.fit(y)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)

        range_, min_ = self.scaler.data_range_, self.scaler.data_min_

        train_data, valid_data, test_data = train_valid_split(self.data, 0.1, 0.1, 418)

        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = valid_data
        else:
            self.data = test_data

        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data).float()

        if inverse:
            self.x, self.y = self.data[:, :1], self.data[:, 1:]
            # self.x, self.y = self.data[:, :3], self.data[:, 3:]
        else:
            self.x, self.y = self.data[:, :6], self.data[:, 6:]
            # self.x, self.y = self.data[:, :8], self.data[:, 8:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_scaler(self):
        return self.x_scaler, self.y_scaler


def get_Binary_dataloaders(model, batch_=128):
    parentpath = os.getcwd()
    # datapath = parentpath + '/Meta_learning_photonics_structure/Model/data/RCWA_xyY_all.mat'
    datapath = './data/100901_data_label.csv'
    # datapath = './data/train_0912_peak.csv'
    if model in ['forward_model', 'tandem_net', 'vae', 'inn', 'vae_new', 'vae_GSNN', 'vae_Full', 'vae_tandem',
                 'vae_hybrid']:
        train_dt = Binary(datapath, 'train')
        val_dt = Binary(datapath, 'val')
        test_dt = Binary(datapath, 'test')

    else:
        train_dt = Binary(datapath, 'train', inverse=True)
        val_dt = Binary(datapath, 'val', inverse=True)
        test_dt = Binary(datapath, 'test', inverse=True)

    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader

class WidthHeight(Dataset):

    def __init__(self, filepath, split='train', inverse=False):
        super(WidthHeight).__init__()
        # temp = scipy.io.loadmat(filepath)
        temp = pd.read_csv(filepath)
        # self.data = np.array(list(temp.items())[3][1])
        temp = temp[["W", "L", "a", "H", "Px", "Py", "Width","Height"]]
        # temp = temp[["W","L","a","H","Px","Py","min_distance",'max_distance',"R","G","B","BIC"]]
        self.data = temp.to_numpy()
        x = self.data[:, :6]
        y = self.data[:, 6:]
        # x = self.data[:,:8]
        # y = self.data[:,8:]
        if inverse:
            x, y = y, x
        self.data = np.hstack((x, y))

        # self.x_scaler = MinMaxScaler()
        # self.y_scaler = MinMaxScaler()
        # self.x_scaler.fit(x)
        # self.y_scaler.fit(y)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data)

        range_, min_ = self.scaler.data_range_, self.scaler.data_min_

        train_data, valid_data, test_data = train_valid_split(self.data, 0.1, 0.1, 418)

        if split == 'train':
            self.data = train_data
        elif split == 'val':
            self.data = valid_data
        else:
            self.data = test_data

        self.data = self.scaler.transform(self.data)
        self.data = torch.tensor(self.data).float()

        if inverse:
            self.x, self.y = self.data[:, :2], self.data[:, 2:]
            # self.x, self.y = self.data[:, :3], self.data[:, 3:]
        else:
            self.x, self.y = self.data[:, :6], self.data[:, 6:]
            # self.x, self.y = self.data[:, :8], self.data[:, 8:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_scaler(self):
        return self.x_scaler, self.y_scaler


def get_WH_dataloaders(model, batch_=128):
    parentpath = os.getcwd()
    # datapath = parentpath + '/Meta_learning_photonics_structure/Model/data/RCWA_xyY_all.mat'
    datapath = './data/101001_data_WH.csv'
    # datapath = './data/train_0912_peak.csv'
    if model in ['forward_model', 'tandem_net', 'vae', 'inn', 'vae_new', 'vae_GSNN', 'vae_Full', 'vae_tandem',
                 'vae_hybrid']:
        train_dt = WidthHeight(datapath, 'train')
        val_dt = WidthHeight(datapath, 'val')
        test_dt = WidthHeight(datapath, 'test')

    else:
        train_dt = WidthHeight(datapath, 'train', inverse=True)
        val_dt = WidthHeight(datapath, 'val', inverse=True)
        test_dt = WidthHeight(datapath, 'test', inverse=True)

    train_loader = DataLoader(train_dt, batch_size=batch_, shuffle=True)
    val_loader = DataLoader(val_dt, batch_size=batch_, shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=batch_, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__  == '__main__':
    train_loader, val_loader, test_loader = get_WH_dataloaders('forward_model', 128)
    print(train_loader.dataset.x,train_loader.dataset.y)