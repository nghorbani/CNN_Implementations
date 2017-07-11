# -*- coding: utf-8 -*-
'''
Generative Adversarial Networks - Goodfellow et al
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - Radford et al
'''
from tools_config import *
import os
import matplotlib.pyplot as plt
from tools_train import get_train_params, OneHot, vis_square
from datetime import datetime
from tools_general import tf, np
from tools_networks import deconv, conv, dense, clipped_crossentropy, dropout

def concat_labels(X, labels):
    if X.get_shape().ndims == 4:
        X_shape = tf.shape(X)
        labels_reshaped = tf.reshape(labels, [-1, 1, 1, 10])
        a = tf.ones([X_shape[0], X_shape[1], X_shape[2], 10])
        X = tf.concat([X, labels_reshaped * a], axis=3)
    return X
     
def create_gan_G(z, labels, is_training, Cout=1, trainable=True, reuse=False, networktype='ganG'):
    '''input : batchsize * 100 and labels to make the generator conditional
        output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        z = tf.concat(axis=-1, values=[z, labels])
        Gz = dense(z, is_training, Dout=4 * 4 * 256, act='reLu', norm='batchnorm', name='dense2')
        Gz = tf.reshape(Gz, shape=[-1, 4, 4, 256])  # 4
        Gz = deconv(Gz, is_training, kernel_w=5, stride=2, Cout=256, trainable=trainable, act='reLu', norm='batchnorm', name='deconv1')  # 11
        Gz = deconv(Gz, is_training, kernel_w=5, stride=2, Cout=128, trainable=trainable, act='reLu', norm='batchnorm', name='deconv2')  # 25
        Gz = deconv(Gz, is_training, kernel_w=4, stride=1, Cout=1, act=None, norm=None, name='deconv3')  # 28
        Gz = tf.nn.sigmoid(Gz)
    return Gz

def create_gan_D(xz, labels, is_training, trainable=True, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        xz = concat_labels(xz, labels)
        Dxz = conv(xz, is_training, kernel_w=5, stride=2, Cout=128, trainable=trainable, act='lrelu', norm=None, name='conv1')  # 12
        Dxz = conv(Dxz, is_training, kernel_w=5, stride=2, Cout=256, trainable=trainable, act='lrelu', norm='batchnorm', name='conv2')  # 4
        Dxz = conv(Dxz, is_training, kernel_w=2, stride=2, Cout=256, trainable=trainable, act='lrelu', norm='batchnorm', name='conv3')  # 2
        Dxz = conv(Dxz, is_training, kernel_w=2, stride=2, Cout=1, trainable=trainable, act='lrelu', norm='batchnorm', name='conv4')  # 2
        Dxz = tf.nn.sigmoid(Dxz)
    return Dxz

def create_dcgan_trainer(base_lr=1e-4, networktype='dcgan'):
    '''Train a Generative Adversarial Network'''
    # with tf.name_scope('train_%s' % networktype): 
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    inZ = tf.placeholder(tf.float32, [None, 100])  # tf.random_uniform(shape=[batch_size, 100], minval=-1., maxval=1., dtype=tf.float32)
    inL = tf.placeholder(tf.float32, [None, 10])  # we want to condition the generated out put on some parameters of the input
    inX = tf.placeholder(tf.float32, [None, 28, 28, 1])

    Gz = create_gan_G(inZ, inL, is_training, Cout=1, trainable=True, reuse=False, networktype=networktype + '_G') 

    DGz = create_gan_D(Gz, inL, is_training, trainable=True, reuse=False, networktype=networktype + '_D')
    Dx = create_gan_D(inX, inL, is_training, trainable=True, reuse=True, networktype=networktype + '_D')
    
    ganG_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(ganG_var_list), [var.name for var in ganG_var_list])

    ganD_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(ganD_var_list), [var.name for var in ganD_var_list])
          
    Gscore = clipped_crossentropy(DGz, tf.ones_like(DGz))
    Dscore = clipped_crossentropy(DGz, tf.zeros_like(DGz)) + clipped_crossentropy(Dx, tf.ones_like(Dx))
    
    Gtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Gscore, var_list=ganG_var_list)
    Dtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Dscore, var_list=ganD_var_list)
    
    return Gtrain, Dtrain, Gscore, Dscore, is_training, inZ, inX, inL, Gz

if __name__ == '__main__':
    networktype = 'DCGAN_MNIST'
    
    batch_size = 128
    base_lr = 0.0002  # 1e-4
    epochs = 30
    
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    
    data, max_iter, test_iter, test_int, disp_int = get_train_params(data_dir + '/' + networktype, batch_size, epochs=epochs, test_in_each_epoch=1, networktype=networktype)
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    Gtrain, Dtrain, Gscore, Dscore, is_training, inZ, inX, inL, Gz = create_dcgan_trainer(base_lr, networktype=networktype)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep = 1000)
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt')  
    
    Z_test = np.random.uniform(size=[batch_size, 100], low=-1., high=1.).astype(np.float32)
    labels_test = OneHot(np.random.randint(10, size=[batch_size]), n=10)    
    
    k = 1
     
    for it in range(1, max_iter): 
        Z = np.random.uniform(size=[batch_size, 100], low=-1., high=1.).astype(np.float32)
        X, labels = data.train.next_batch(batch_size)
         
        for itD in range(k):
            cur_Gscore, _ = sess.run([Gscore, Gtrain], feed_dict={inZ:Z, inL:labels, is_training:True})
    
        cur_Dscore, _ = sess.run([Dscore, Dtrain], feed_dict={inX:X, inZ:Z, inL:labels, is_training:True})
     
        if it % disp_int == 0:
            Gz_sample = sess.run(Gz, feed_dict={inZ: Z_test, inL: labels_test, is_training:False})
            vis_square(Gz_sample[:121], [11, 11], save_path=work_dir + 'Iter_%d.jpg' % it)
            saver.save(sess, work_dir + "%.3d_model.ckpt" % it)
            if ('cur_Dscore' in vars()) and ('cur_Gscore' in vars()):
                print("Iteration #%4d, Train Gscore = %f, Dscore=%f" % (it, cur_Gscore, cur_Dscore))
