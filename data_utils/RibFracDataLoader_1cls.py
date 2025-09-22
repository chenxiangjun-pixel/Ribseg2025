import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/pn', npoints=30000, split='train', transforms=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.transforms = transforms
        self.normal_channel = normal_channel

        self.meta = {}
        self.meta_label = {}
        train_ids = set([x for x in os.listdir(self.root + '/data_pn/train')])
        val_ids = set([x for x in os.listdir(self.root + '/data_pn/val')])
        test_ids = set([x for x in os.listdir(self.root + '/data_pn/test')])

        self.meta['rib'] = []
        self.meta_label['rib'] = []

        # print(fns[0][0:-4])
        if split == 'trainval':
            fns = ['./data/pn/data_pn/train/' + fn for fn in train_ids] + ['./data/pn/data_pn/val/' + fn for fn in val_ids]
            fns_l = ['./data/pn/label_pn/train/' + fn for fn in train_ids] + ['./data/pn/label_pn/val/' + fn for fn in val_ids]
        elif split == 'train':
            fns = ['./data/pn/data_pn/train/' + fn for fn in train_ids]
            fns_l = ['./data/pn/label_pn/train/' + fn for fn in train_ids]
        elif split == 'val':
            fns = ['./data/pn/data_pn/val/' + fn for fn in val_ids]
            fns_l = ['./data/pn/label_pn/val/' + fn for fn in val_ids]
        elif split == 'test':
            fns = ['./data/pn/data_pn/test/' + fn for fn in test_ids]
            fns_l = ['./data/pn/label_pn/test/' + fn for fn in test_ids]
        else:
            fns = []
            fns_l = []
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        # print(os.path.basename(fns))
        for fn in fns:
            self.meta['rib'].append(fn)

        for fn in fns_l:
            self.meta_label['rib'].append(fn)

        self.datapath = []
        self.labelpath = []

        for fn in self.meta['rib']:
            self.datapath.append(('rib', fn))

        for fn in self.meta_label['rib']:
            self.labelpath.append(('rib', fn))

        self.classes = {'rib': 0}
        self.seg_classes = {'rib': [0, 1]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 100

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            fn_l = self.labelpath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)

            data = np.load(fn[1]).astype(np.float32)
            point_set = data[:, 0:3]
            seg = np.load(fn_l[1]).astype(np.int32)
            seg[seg != 0] = 1
            # print(Counter(seg.flatten()))
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        choice = np.random.choice(len(seg), self.npoints, replace=False)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.transforms is not None:
            point_set = self.transforms(point_set)

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
