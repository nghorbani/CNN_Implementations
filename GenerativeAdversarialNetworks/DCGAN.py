# -*- coding: utf-8 -*-
'''
Generative Adversarial Networks - Goodfellow et al. 2014
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - Radford et al. 2015

Use this code with no warranty and please respect the accompanying license.
'''

import sys
sys.path.append('../common')

from tools_config import data_dir, expr_dir
import os
import matplotlib.pyplot as plt
from tools_train import get_train_params, OneHot, vis_square
from datetime import datetime
from tools_general import tf, np
from tools_networks import deconv, conv, dense, clipped_crossentropy, dropout

from tensorflow.examples.tutorials.mnist import input_data
    
def create_gan_G(z, is_training, Cout=1, trainable=True, reuse=False, networktype='ganG'):
    '''input : batchsize * latentDim
       output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        Gout = dense(z, is_training, Cout=7 * 7 * 256, trainable=trainable, act='reLu', norm='batchnorm', name='dense1')
        Gout = tf.reshape(Gout, shape=[-1, 7, 7, 256])  # 7
        Gout = deconv(Gout, is_training, kernel_w=4, stride=2, epf=2, Cout=128, trainable=trainable, act='reLu', norm='batchnorm', name='deconv1')  # 14
        Gout = deconv(Gout, is_training, kernel_w=4, stride=2, epf=2, Cout=Cout, trainable=trainable, act=None, norm=None, name='deconv2')  # 28
        Gout = tf.nn.sigmoid(Gout)
    return Gout

def create_gan_D(xz, is_training, trainable=True, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        Dout = conv(xz, is_training, kernel_w=4, stride=2, pad=1, Cout=128, trainable=trainable, act='lrelu', norm=None, name='conv1')  # 14
        Dout = conv(Dout, is_training, kernel_w=4, stride=2, pad=1, Cout=256  , trainable=trainable, act='lrelu', norm='batchnorm', name='conv2')  # 7
        Dout = conv(Dout, is_training, kernel_w=3, stride=1, pad=None, Cout=1, trainable=trainable, act=None, norm='batchnorm', name='conv4')  # 5
        Dout = tf.nn.sigmoid(Dout)
    return Dout

def create_dcgan_trainer(base_lr=1e-4, networktype='dcgan', latentDim=100):
    '''Train a Generative Adversarial Network'''
    eps = 1e-8
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Zph = tf.placeholder(tf.float32, [None, latentDim])  # tf.random_uniform(shape=[batch_size, 100], minval=-1., maxval=1., dtype=tf.float32)
    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])

    Gout_op = create_gan_G(Zph, is_training, Cout=1, trainable=True, reuse=False, networktype=networktype + '_G') 

    fakeLogits = create_gan_D(Gout_op, is_training, trainable=True, reuse=False, networktype=networktype + '_D')
    realLogits = create_gan_D(Xph, is_training, trainable=True, reuse=True, networktype=networktype + '_D')
    
    G_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(G_varlist), [var.name for var in G_varlist])

    D_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(D_varlist), [var.name for var in D_varlist])
          
    Gloss = clipped_crossentropy(fakeLogits, tf.ones_like(fakeLogits))
    Dloss = clipped_crossentropy(fakeLogits, tf.zeros_like(fakeLogits)) + clipped_crossentropy(realLogits, tf.ones_like(realLogits))
    
#     clipvals = lambda val: tf.clip_by_value(val, eps, 1. - eps)
#     Dloss = tf.reduce_mean(tf.log(clipvals(realLogits)) + tf.log(1. - clipvals(fakeLogits)))
#     Gloss = -tf.reduce_mean(tf.log(clipvals(fakeLogits)))
    
    Gtrain_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Gloss, var_list=G_varlist)
    Dtrain_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Dloss, var_list=D_varlist)
    
    return Gtrain_op, Dtrain_op, Gloss, Dloss, is_training, Zph, Xph, Gout_op

if __name__ == '__main__':
    networktype = 'DCGAN_MNIST'
    
    batch_size = 128
    base_lr = 2e-4
    epochs = 500
    latentDim = 100
    disp_every_epoch = 5
    
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    
    data = input_data.read_data_sets(data_dir + '/' + networktype, reshape=False)
    disp_int = disp_every_epoch * int(data.train.num_examples / batch_size)  # every two epochs
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    Gtrain_op, Dtrain_op, Gloss, Dloss, is_training, Zph, Xph, Gout_op = create_dcgan_trainer(base_lr, networktype=networktype)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * 0.1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt')  
        
    k = 1
    it = 0
    disp_losses = False    

    while data.train.epochs_completed < epochs:
        dtemploss = 0 
        
        for itD in range(k):
            it += 1
            Z = np.random.uniform(size=[batch_size, latentDim], low=-1., high=1.).astype(np.float32)
            X, _ = data.train.next_batch(batch_size)
            
            cur_Dloss, _ = sess.run([Dloss, Dtrain_op], feed_dict={Xph:X, Zph:Z, is_training:True})
            dtemploss += cur_Dloss
            
            if it % disp_int == 0:disp_losses = True
             
        cur_Dloss = dtemploss / k   
        
        Z = np.random.uniform(size=[batch_size, latentDim], low=-1., high=1.).astype(np.float32)     
        cur_Gloss, _ = sess.run([Gloss, Gtrain_op], feed_dict={Zph:Z, is_training:True})
    
        if disp_losses:
            Gsample = sess.run(Gout_op, feed_dict={Zph: Z, is_training:False})
            vis_square(Gsample[:121], [11, 11], save_path=work_dir + 'Epoch%.3d.jpg' % data.train.epochs_completed)
            saver.save(sess, work_dir + "%.3d_model.ckpt" % data.train.epochs_completed)
            print("Epoch #%.3d, Train Gloss = %f, Dloss=%f" % (data.train.epochs_completed, cur_Gloss, cur_Dloss))
            disp_losses = False
