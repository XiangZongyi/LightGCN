import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.sparse as sp
import random
import torch
import copy


def create_dataset(args):
    if args.dataset in ['ml-100k', 'ml-1m', 'ml-10m']:
        return ML_Dataset(args)
    elif args.dataset in ['gowalla', 'amazon-book', 'yelp2018']:
        return LightGCN_Dataset(args)
    else:
        raise ValueError('Check args.dataset if right!')


def create_dataloader(dataset, batch_size, training=False):
    if training:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class ML_Dataset(object):
    def __init__(self, args):
        self.args = args

        self.dataset = self.args.dataset
        self.data_path = self.args.data_path

        self.inter_feat = self._load_dataframe()  # user-item-interaction DataFrame: user, item, rating, timestamp

        # user_id, item_id, rating, timestamp
        self.uid_field, self.iid_field, self.rating_field, self.timestamp = self.inter_feat.columns

        self.user_num = self.inter_feat[self.uid_field].max() + 1
        self.item_num = self.inter_feat[self.iid_field].max() + 1

        self.train_inter_feat, self.test_inter_feat = self._split_inter_feat()  # DataFrame: user, pos_item_list

        # the positive item num and negative item num of each user
        self.train_items_num = [len(i) for i in self.train_inter_feat['train_items']]
        self.neg_items_num = [self.args.neg_sample_num * len(i) for i in self.train_inter_feat['train_items']]

    def _load_dataframe(self):
        # '../data/ml-100k/ml-100k.inter'
        inter_feat_path = os.path.join(self.data_path + '/' + self.dataset + '/' + f'{self.dataset}.inter')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')

        # create DataFrame
        columns = []
        usecols = []
        dtype = {}
        with open(inter_feat_path, 'r') as f:
            head = f.readline()[:-1]
        for field_type in head.split('\t'):
            field, ftype = field_type.split(':')
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == 'float' else int
        df = pd.read_csv(
            inter_feat_path, delimiter='\t', usecols=usecols, dtype=dtype
        )
        df.columns = columns
        if 'rating' in columns:
            df['rating'] = 1.0  # implicit feedback

        # reset user(item) id from 1-943(1-1682) to 0-942(0-1681)
        if df[columns[0]].min() > 0:
            df[columns[0]] = df[columns[0]].apply(lambda x: x - 1)
            df[columns[1]] = df[columns[1]].apply(lambda x: x - 1)

        return df

    def _split_inter_feat(self):
        interact_status = self.inter_feat.groupby(self.uid_field)[self.iid_field].apply(set).reset_index().rename(
            columns={self.iid_field: 'interacted_items'}
        )  # user-item_dic-interaction DataFrame: user, interacted_items

        # split train and test randomly by args.data_split_ratio
        interact_status['train_items'] = interact_status['interacted_items'].\
            apply(lambda x: set(random.sample(x, round(len(x) * self.args.data_split_ratio[0]))))
        interact_status['test_items'] = interact_status['interacted_items'] - interact_status['train_items']
        interact_status['train_items'] = interact_status['train_items'].apply(list)
        interact_status['test_items'] = interact_status['test_items'].apply(list)

        train_inter_feat = interact_status[[self.uid_field, 'train_items']]
        test_inter_feat = interact_status[[self.uid_field, 'test_items']]

        return train_inter_feat, test_inter_feat

    def _sample_negative(self, pos_item, sampling_num):
        neg_item = []
        for i in range(sampling_num):
            while True:
                negitem = random.choice(range(self.item_num))
                if negitem not in pos_item:
                    break
            neg_item.append(negitem)
        return neg_item

    def get_train_dataset(self):
        users, pos_items, neg_items = [], [], []
        mask_index = {}  # dict: used for test
        for row in self.train_inter_feat.itertuples():
            index = getattr(row, 'Index')
            user_id = getattr(row, self.uid_field)
            pos_item = getattr(row, 'train_items')
            neg_item = self._sample_negative(pos_item, self.neg_items_num[index])

            mask_index[user_id] = pos_item

            users.extend([user_id] * len(neg_item))
            pos_items.extend(pos_item * self.args.neg_sample_num)
            neg_items.extend(neg_item)

        interaction_matrix = self.inter_matrix(users, pos_items)

        train_dataset = TorchDataset(user=torch.LongTensor(users),
                                     pos_item=torch.LongTensor(pos_items),
                                     neg_item=torch.LongTensor(neg_items))
        return train_dataset, interaction_matrix, mask_index

    def get_test_data(self):
        test_users = list(self.test_inter_feat[self.uid_field])
        ground_true_items = list(self.test_inter_feat['test_items'])  # list like [[],[],...,[]] len: n_users
        return test_users, ground_true_items

    def inter_matrix(self, users, pos_items, form='coo'):
        row = users
        col = pos_items
        data = np.ones(len(row))

        mat = sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')


class LightGCN_Dataset(object):
        def __init__(self, args):
            self.args = args

            path = os.path.join(self.args.data_path, self.args.dataset)
            self.train_file = path + '/train.txt'
            self.test_file = path + '/test.txt'

            if self.args.dataset == 'gowalla':
                self.user_num = 29858
                self.item_num = 40981
            elif self.args.dataset == 'amazon-book':
                self.user_num = 52643
                self.item_num = 91599
            elif self.args.dataset == 'yelp2018':
                self.user_num = 31668
                self.item_num = 38048
            else:
                raise ValueError('Check args.dataset if right!')

        def get_train_dataset(self):
            users, pos_items, neg_items = [], [], []
            mask_index = {}
            with open(self.train_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        pos_item = [int(i) for i in l[1:]]
                        user_id = int(l[0])
                        neg_item = self._sample_negative(pos_item)

                        mask_index[user_id] = pos_item

                        users.extend([user_id] * len(neg_item))
                        pos_items.extend(pos_item * self.args.neg_sample_num)
                        neg_items.extend(neg_item)
                interaction_matrix = self._inter_matrix(users, pos_items)

                train_dataset = TorchDataset(user=torch.LongTensor(users),
                                             pos_item=torch.LongTensor(pos_items),
                                             neg_item=torch.LongTensor(neg_items))
            return train_dataset, interaction_matrix, mask_index

        def _sample_negative(self, pos_item):
            sampling_num = len(pos_item) * self.args.neg_sample_num
            neg_item = []
            for i in range(sampling_num):
                while True:
                    negitem = random.choice(range(self.item_num))
                    if negitem not in pos_item:
                        break
                neg_item.append(negitem)
            return neg_item

        def _inter_matrix(self, users, pos_items, form='coo'):
            row = users
            col = pos_items
            data = np.ones(len(row))

            mat = sp.coo_matrix((data, (row, col)), shape=(self.user_num, self.item_num))

            if form == 'coo':
                return mat
            elif form == 'csr':
                return mat.tocsr()
            else:
                raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

        def get_test_data(self):
            test_users = list(range(self.user_num))

            ground_true_items = []  # list like [[],[],...,[]] len: n_users
            with open(self.test_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        if l[1:] == ['']:
                            items = []
                            ground_true_items.append(items)
                        else:
                            items = [int(i) for i in l[1:]]
                            ground_true_items.append(items)
            return test_users, ground_true_items


class TorchDataset(Dataset):
    def __init__(self, user, pos_item, neg_item):
        super(Dataset, self).__init__()

        self.user = user
        self.pos_item = pos_item
        self.neg_item = neg_item

    def __len__(self):
        return len(self.user)

    def __getitem__(self, idx):
        return self.user[idx], self.pos_item[idx], self.neg_item[idx]

