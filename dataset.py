import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json



class PartNormalDataset(Dataset):
    def __init__(self, root='D:/nusit_data_sampling/', npoints=2048, split='train', class_choice=None,
                 normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.split = split

        # 1. 定义区域 (根据你的需求修改)
        self.train_areas = ['Area1',  'Area4', 'Area5',
                            'Area6', 'Area7', 'Area9', 'Area10']
        self.test_areas = ['Area3',  'Area8']

        self.val_areas = ['Area2', 'Area11']



        if split == 'train':
            self.target_areas = self.train_areas
        elif split == 'test':
            self.target_areas = self.test_areas
        elif split == 'val':
            self.target_areas = self.val_areas
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        print(f"Current Split: {split}, Loading Areas: {self.target_areas}")

        # ------------------------------------------------------------------
        # 【补回来的部分】 定义 seg_classes
        # ------------------------------------------------------------------
        # 原本你是 self.seg_classes = {'AREA1': [0, 1, 2, 3, 4, 5, 6, 7]}
        # 但现在你有 Area1 - Area11，如果只写 AREA1，训练到 Area2 时可能会报错。
        # 这里自动把所有 Area 都设为包含标签 0-7。

        self.seg_classes = {}
        # 获取所有可能的区域列表（包括训练和测试），防止 key error
        all_possible_areas = set(self.train_areas + self.test_areas + self.val_areas)

        for area in all_possible_areas:
            # 假设你的所有区域标签都是 0 到 7 (共8类)
            self.seg_classes[area] = [0, 1, 2, 3, 4, 5, 6, 7]

        # ------------------------------------------------------------------
        # 2. 遍历指定文件夹
        # ------------------------------------------------------------------
        self.datapath = []
        self.data_list = []

        for area_name in self.target_areas:
            dir_point = os.path.join(self.root, area_name)
            if not os.path.exists(dir_point):
                print(f"Warning: Folder {dir_point} does not exist, skipping.")
                continue

            fns = sorted([fn for fn in os.listdir(dir_point) if fn.endswith('.txt')])

            for fn in fns:
                full_path = os.path.join(dir_point, fn)
                self.datapath.append((area_name, full_path))
                token = os.path.splitext(fn)[0]
                self.data_list.append(token)

        # 3. 简单的类别映射 (Area Name -> Int)
        self.classes = {}
        for area in all_possible_areas:
            self.classes[area] = 0

        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = fn[0]
            cls = np.array([0]).astype(np.int32)


            # data = np.loadtxt(fn[1]).astype(np.float32)
            try:
                data = np.loadtxt(fn[1]).astype(np.float32)
            except ValueError:
                print(f"!!! 数据损坏的文件路径: {fn[1]}")
                raise  # 继续抛出错误让程序停止，以便你看到上面的打印信息

            point_set = data[:, 0:8]
            # if not self.normal_channel:
            #     point_set = data[:, 0:6]
            # else:
            #     point_set = data[:, 0:8]

            seg = data[:, -1].astype(np.int32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set.copy(), cls, seg.copy())

        originxyz = point_set[:, 0:3].copy()
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # point_set[:, 3:8] = point_set[:, 3:8] / 255.0  # 加入这一行
        if len(seg) >= self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=True)

        point_set = point_set[choice, :]
        seg = seg[choice]
        originxyz = originxyz[choice, :]

        cur_name = self.data_list[index]

        return point_set, originxyz, cls, seg, cur_name

    def __len__(self):
        return len(self.datapath)



if __name__ == '__main__':
    data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,label in DataLoader:
        print(point.shape)
        print(label.shape)