from tensorflow.examples.tutorials.mnist import input_data
from tools_general import np, tf
from tools_config import *

def get_train_params(batch_size, epochs=20, test_in_each_epoch=1, networktype='GAN_MNIST'):

    data = input_data.read_data_sets(data_dir+'/'+networktype, one_hot=False, reshape=False)
    
    train_num = data.train.num_examples  # total number of training images
    test_num = data.test.num_examples  # total number of validation images
           
    max_iter = int(np.ceil(epochs * train_num / batch_size))
    test_iter = int(np.ceil(test_num / batch_size))
    test_interval = int(train_num / (test_in_each_epoch * batch_size))  # test 2 times in each epoch
    disp_interval = int(test_interval / 2)
    if disp_interval == 0: disp_interval = 1
    
    # snapshot_interval = test_interval * 5  # save at every epoch
    
    return data, max_iter, test_iter, test_interval, disp_interval