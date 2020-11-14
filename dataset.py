import csv
import glob
import random
from collections import Counter


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.utils import make_grid

####


class DatasetSerial(data.Dataset):

    def __init__(self, pair_list, shape_augs=None, input_augs=None, has_aux=False):
        self.has_aux = has_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        #print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        #print(input_img.shape)

        #shape must be deterministic so it can be reused
        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            input_img = shape_augs.augment_image(input_img)

        # additional augmenattion just for the input
        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        #print(input_img.shape)
        return input_img, img_label

    def __len__(self):
        return len(self.pair_list)


class DatasetSerialWSI(data.Dataset):

    def __init__(self, path_list):
        self.path_list = path_list

    def __getitem__(self, idx):

        input_img = cv2.imread(self.path_list[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        return input_img, self.path_list[idx]

    def __len__(self):
        return len(self.path_list)
#
#
# def prepare_colon_data(fold_idx=0):
#
#     """TMA is Thrombotic Microangiopathy is a pattern of damage that can occur
#     in the smallest blood vessels inside organs"""
#
#     assert fold_idx < 5, "Currently only support 5 fold, each fold is 1 TMA"
#
#     tma_list = ['1010711', '1010712', '1010713', '1010714', '1010715', '1010716']
#
#     file_list = []
#     label_list = []
#     for tma_code in tma_list:
#         tma_file_list = glob.glob('../../../train/COLON_MICCAI2019/2048x2048_tma_1_rgb/%s/*.jpg' % tma_code)
#         tma_label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in tma_file_list]
#         file_list.extend(tma_file_list)
#         label_list.extend(tma_file_list)
#     pair_list = list(zip(file_list, label_list))
#     # [(0, 139), (1, 235), (2, 645), (3, 194)] highly imbalance
#
#     train_fold = []
#     valid_fold = []
#     skf = StratifiedKFold(n_splits=5, random_state=5, shuffle=False)
#     for train_idex, valid_index in skf.split(file_list, label_list):
#         train_fold.append([pair_list[idx] for idx in list(train_idex)])
#         valid_fold.append([pair_list(idx) for idx in list(valid_index)])
#
#     return train_fold[fold_idx], valid_fold[fold_idx]



def prepare_colon_manual_data(fold_idx=0):

    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        else:
            label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    assert fold_idx < 5, "Currently only support 5 fold, each fold is 1 TMA"

    data_root_dir = '/media/vtltrinh/Data1/COLON_PATCHES_1000/'
    tma_list = ['1010711', '1010712', '1010716']

    set_1010711 = load_data_info('%s/v2/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/v2/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_data_info('%s/v2/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_data_info('%s/v2/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_data_info('%s/v2/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/v2/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_data_info('%s/v2/wsi_00016/*.jpg' % data_root_dir, parse_label=False,
                               label_value=0)  # benign exclusively
    wsi_00017 = load_data_info('%s/v2/wsi_00017/*.jpg' % data_root_dir, parse_label=False,
                               label_value=0)  # benign exclusively
    wsi_00018 = load_data_info('%s/v2/wsi_00018/*.jpg' % data_root_dir, parse_label=False,
                               label_value=0)  # benign exclusively
    # random.shuffle(wsi)
    # random.shuffle(wsi)

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010714 + set_1010716 + wsi_00017 + wsi_00018
    # train_set = wsi_00016 + set_1010711 + set_1010712 #+ set_1010713 + set_1010715
    # valid_set = set_1010713 + set_1010715 + wsi_00017#  + wsi_00018
    return train_set, valid_set


def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size+1):
            sample = ds[data_idx+j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                #TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[..., :3] #gray to RGB heatmap
                aux = (aux*255).astype('unint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size


#test


# train_list, valid_list = prepare_colon_manual_data()
# train_dataset = DatasetSerial(train_list)
# valid_dataset = DatasetSerial(valid_list)

#visualize(train_dataset, batch_size=4, nr_steps=16)
