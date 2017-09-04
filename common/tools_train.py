from tensorflow.examples.tutorials.mnist import input_data
import custom_input_data
import matplotlib.pyplot as plt

from tools_general import np, tf
import scipy.misc

def get_train_params(data_dir, batch_size, epochs=20, test_in_each_epoch=1,one_hot=True, networktype='GAN_MNIST'):
    
    if 'img2img' in networktype:
        data_dir = data_dir + '/' + networktype.replace('_A2B','').replace('_B2A','')
        data = custom_input_data.load_dataset(data_dir, networktype=networktype)
    else:
        data = input_data.read_data_sets(data_dir + '/' + networktype, one_hot=one_hot, reshape=False)
    
    train_num = data.train.num_examples  # total number of training images
    test_num = data.test.num_examples  # total number of validation images
    
    print('Trainset size:', train_num, 'Testset_size:', test_num)           
    max_iter = int(np.ceil(epochs * train_num / batch_size))
    test_iter = int(np.ceil(test_num / batch_size))
    test_interval = int(train_num / (test_in_each_epoch * batch_size))  # test 2 times in each epoch
    disp_interval = int(test_interval * 2)
    if disp_interval == 0: disp_interval = 1
    
    # snapshot_interval = test_interval * 5  # save at every epoch
    
    return data, max_iter, test_iter, test_interval, disp_interval

def OneHot(X, n=10):
    return np.eye(n)[np.array(X).reshape(-1)].astype(np.float32)

def vis_square(X, nh_nw, save_path=None):
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))
    for n, x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = x
    if save_path:
        scipy.misc.imsave(save_path, img)
        return save_path
    else:
        return img
    
def plot_latent_variable(data, labels):
    if data.shape[1] != 2:
        pca = PCA(n_components=2)
        data = pca.fit_transform(data)
        print(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 8))
    plt.axes().set_aspect('equal')
    color = plt.cm.rainbow(np.linspace(0, 1, 10))
    for l, c in enumerate(color):
        idxs = np.where(labels==l)
        plt.scatter(data[idxs, 0], data[idxs, 1], c=c, label=l, linewidth=0, s=8)
    plt.legend()
    plt.show()
    
def count_model_params(variables=None):
    if variables == None:
        variables = tf.trainable_variables()
    total_parameters = 0
    for variable in variables:
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    return(total_parameters)