# -*- coding: utf-8 -*-

from tools_config import *
from tools_general import rng, np

from tensorflow.contrib.learn.python.learn.datasets import base
import os


class DataSet(object):

    def __init__(self, datas):

        self._num_examples = datas.shape[0]

        self._datas = datas
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.data_shape = datas.shape 
        
        self.indexlist = np.arange(self.datas.shape[0])
        rng.shuffle(self.indexlist)

    @property
    def datas(self):
        return self._datas

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """
        Load the next data batch from the dataset.
        """
        assert batch_size <= self._num_examples
        
        idxtofetch = []
        for bIdx in np.arange(batch_size):
            if self._index_in_epoch >= self._num_examples:  # Did we finish an epoch?
                self._epochs_completed += 1
                self._index_in_epoch = 0
                rng.shuffle(self.indexlist)
                

            idxtofetch.append(self.indexlist[self._index_in_epoch])
            self._index_in_epoch += 1

        return self.datas[idxtofetch, :, :, :]
    
    
def fetch_data(data_dir, networktype, dtype=np.float32):

    Xtrain = np.load(os.path.join(data_dir, 'train.npz'))['data']
    Xtest = np.load(os.path.join(data_dir, 'test.npz'))['data']

    train_data = Xtrain.copy()
    test_data = Xtest.copy()
    # train_data = normalize_data(Xtrain, networktype).astype(dtype=dtype)
    # test_data = normalize_data(Xtest, networktype).astype(dtype=dtype)
    
    return train_data, test_data

def load_dataset(data_dir, networktype='manifold', val_size=0.0, dtype=np.float32):
    """val_size = %X of the training dataset as validation data"""

    train_data, test_data = fetch_data(data_dir, networktype, dtype)
    validation_size = int(val_size * len(train_data))
    
    val_data = train_data[:validation_size]
    train_data = train_data[validation_size:]

    train = DataSet(train_data)
    validation = DataSet(val_data)
    test = DataSet(test_data)

    #print('Total Train data size is {}. Total Validation data size is {}. Total Test data size is {}.'.format(train._num_examples, validation._num_examples, test._num_examples))
    return base.Datasets(train=train, validation=validation, test=test)

if __name__ == "__main__":
    load_dataset(data_dir, dtype=np.float32, val_size=0.1)

            
