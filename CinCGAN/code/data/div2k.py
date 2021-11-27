import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    def __init__(self, args, train=True, transform=None):
        super(DIV2K, self).__init__(args, train, transform)
        # print(args.batch_size)
        self.repeat = int(args.test_every / (args.n_train / args.batch_size))
    def _scan(self):
        list_hr = []#clean?
        list_lr = [[] for _ in self.scale]#원본 input LR
        list_lrb = [[] for _ in self.scale] #이것만 다른이미지?
        if self.train:
            idx_begin = self.args.offset_train
            idx_end = self.args.offset_train + self.args.n_train
        else:
            idx_begin = self.args.offset_val
            idx_end = self.args.offset_val + self.args.n_val
        # print(idx_begin + 1, idx_end + 1)
        for idx, i in enumerate(range(idx_begin + 1, idx_end + 1)):
            # filename_x= '{:0>4}'.format(idx+1)
            filename_yz = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename_yz + self.ext)) #401~
            # for si, s in enumerate(self.scale):
            list_lr[0].append(os.path.join(
                self.dir_lr,
                '{}x{}m{}'.format(filename_yz, 4, self.ext)#1~ #_x에서 _yz로 통일시킴
            ))
            list_lrb[0].append(os.path.join(
                self.dir_lrb,
                'X{}/{}x{}{}'.format(4, filename_yz, 4, self.ext)#401~
            ))
        return list_hr, list_lr, list_lrb

    def _set_filesystem(self, dir_data):
        # self.apath = dir_data + '/DIV2K/DS_CinCGAN'
        self.apath = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_mild')
        self.dir_lrb = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.npy'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.npy'.format(self.split, scale)
        )

    def __len__(self):
        if self.train:
            return len((self.images_hr)) * self.repeat
        else:
            return len((self.images_hr))

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

