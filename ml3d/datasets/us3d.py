import numpy as np
import os, sys, glob, pickle
from pathlib import Path
from os.path import join, exists, dirname, abspath
import random
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
import json
import pdal

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)

class US3DSplit(BaseDatasetSplit):
    """This class is used to create a custom dataset split.

    Initialize the class.

    Args:
        dataset: The dataset to split.
        split: A string identifying the dataset split that is usually one of
        'training', 'test', 'validation', or 'all'.
        **kwargs: The configuration of the model as keyword arguments.

    Returns:
        A dataset split object providing the requested subset of the data.
    """

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        print("Reading {}".format(pc_path))

        p = pdal.Pipeline(json.dumps([{"type":"readers.las","filename":pc_path,"use_eb_vlr":True}]))
        cnt = p.execute()
        print("Processed {} points".format(cnt))

        data = p.arrays[0]
        
        points = np.vstack((data['X'], data['Y'], data['Z'])).T.astype(np.float32)
        # print(points.shape)

        # dz = np.zeros((len(data['X'])))
        # marked = np.zeros((len(data['X'])))
        # print(len(data['X']))
        # for i in range(len(data['X'])):
        #     if marked[i]>0:
        #         continue
        #     t = data['GpsTime'][i]
        #     # print(i, marked[i], t)
        #     idx = np.where(data['GpsTime']==t)[0]
        #     # print(idx)
        #     # dz[i] = data['Z'][i] - data['Z'][idx].min()
        #     dz[idx] = data['Z'][idx] - data['Z'][idx].min()
        #     marked[idx] = 1
        #     print(np.sum(marked))

        print(np.max(data['ReturnNumber']), np.max(data['NumberOfReturns']))
        print(np.max(data['ReturnNumber']-1), np.max(data['NumberOfReturns']-1))
        rn = np.eye(5)[data['ReturnNumber']-1]
        nr = np.eye(5)[data['NumberOfReturns']-1]
        dz = (data['DeltaZ'] - 1.967377673) / 4.567553495
        print(np.max(dz), np.max(data['ReturnNumber']), np.max(data['NumberOfReturns']))
        print(np.min(dz), np.max(data['ReturnNumber']-1), np.max(data['NumberOfReturns']-1))

        print(rn.shape, nr.shape, dz[:, None].shape)

        # Look into why we couldn't include Verticality.
        # feat = np.vstack((data['Linearity'],data['Planarity'],data['Scattering'],data['Verticality'])).T.astype(np.float32)
        # feat = np.vstack((data['X'], data['Y'], data['Z'])).T.astype(np.float32)
        # feat = np.vstack((data['ReturnNumber'],data['Intensity'])).T.astype(np.float32)
        feat = np.concatenate((rn.T,nr.T,dz[:, None].T)).T.astype(np.float32)
        print(feat.shape)

        if (self.split != 'test'):
            labels = data['Classification'].astype(np.int32)
            labels[np.isin(labels,[2, 5, 6],invert=True)]=0
            labels = [self.dataset.label_to_idx[label] for label in labels]
        else:
            labels = np.zeros((points.shape[0],), dtype=np.int32)

        data = {'point': points, 'feat': feat, 'label': labels}

        return data

    def get_attr(self, idx):
        pc_path = Path(self.path_list[idx])
        name = pc_path.name.replace('.txt', '')

        attr = {'name': name, 'path': str(pc_path), 'split': self.split}

        return attr


class US3D(BaseDataset):
    """A template for customized dataset that you can use with a dataloader to
    feed data when training a model. This inherits all functions from the base
    dataset and can be modified by users. Initialize the function by passing the
    dataset and other details.

    Args:
        dataset_path: The path to the dataset to use.
        name: The name of the dataset.
        cache_dir: The directory where the cache is stored.
        use_cache: Indicates if the dataset should be cached.
        num_points: The maximum number of points to use when splitting the dataset.
        ignored_label_inds: A list of labels that should be ignored in the dataset.
        test_result_folder: The folder where the test results should be stored.
    """

    def __init__(self,
                 dataset_path,
                 name='US3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 ignored_label_inds=[],
                 test_result_folder='./test',
                 **kwargs):

        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.dataset_path = cfg.dataset_path
        self.label_to_names = self.get_label_to_names()

        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array(cfg.ignored_label_inds)

        self.train_dir = str(Path(cfg.dataset_path) / cfg.train_dir)
        self.val_dir = str(Path(cfg.dataset_path) / cfg.val_dir)
        self.test_dir = str(Path(cfg.dataset_path) / cfg.test_dir)

        self.train_files = [f for f in glob.glob(self.train_dir + "/*.laz")]
        self.val_files = [f for f in glob.glob(self.val_dir + "/*.laz")]
        self.test_files = [f for f in glob.glob(self.test_dir + "/*.laz")]

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictonary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        # Need to clean this up such that we can use the actual, sparse labels.
        label_to_names = {
            0: 'Unclassified',
            2: 'Ground',
            5: 'Vegetation',
            6: 'Building',
            # 9: 'Water',
            # 17: 'Bridge',
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return US3DSplit(self, split=split)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
             ValueError: Indicates that the split name passed is incorrect. The
             split name should be one of 'training', 'test', 'validation', or
             'all'.
        """
        if split in ['test', 'testing']:
            random.shuffle(self.test_files)
            return self.test_files
        elif split in ['val', 'validation']:
            random.shuffle(self.val_files)
            return self.val_files
        elif split in ['train', 'training']:
            random.shuffle(self.train_files)
            return self.train_files
        elif split in ['all']:
            files = self.val_files + self.train_files + self.test_files
            return files
        else:
            raise ValueError("Invalid split {}".format(split))

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        store_path = join(path, self.name, name + '.npy')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        path = cfg.test_result_folder
        make_dir(path)

        pred = results['predict_labels']
        pred = np.array(pred)

        store_path = join(path, name + '.npy')
        np.save(store_path, pred)

    # Need to modify such that we can either pass the desired filename or construct it from the input.
    def write_result(self, filename, data):
        p = pdal.Pipeline(json.dumps([{
            "type":"writers.las",
            "filename":filename,
            "forward":"all"
            }]), [data])
        p.validate()
        p.execute()

DATASET._register_module(US3D)
