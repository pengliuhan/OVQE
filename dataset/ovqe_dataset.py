import glob
import random
import torch
import os.path as op
import numpy as np
from cv2 import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop, augment, totensor
import torchvision

def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


class OVQEDataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.

    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """

    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['gt_path']
        )
        self.lq_root = op.join(
            'data/MFQEv2/',
            self.opts_dict['lq_path']
        )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root,
            self.opts_dict['meta_info_fp']
        )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root,
            self.gt_root
        ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        nfs = 2 * radius + 1
        self.neighbor_list = [i + 1 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]

        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        # get the neighboring HQ frames
        img_gt = []
        for neighbor in self.neighbor_list:
            img_gt_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_g = _bytes2img(img_bytes)  # (H W 1)
            img_gt.append(img_g)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
        )
        img_lqs = img_lqs + img_gt

        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
        )

        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:len(img_gt)], dim=0)
        img_gt = torch.stack(img_results[len(img_gt):2 * len(img_gt)], dim=0)

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
        }

    def __len__(self):
        return len(self.keys)


