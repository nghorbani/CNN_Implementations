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
    
def create_generator(Xin, is_training, Cout=1, reuse=False, networktype='ganG'):
    '''input : batchsize * latentD
       output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        Xout = dense(Xin, is_training, Cout=7 * 7 * 256, act='reLu', norm='batchnorm', name='dense1')
        Xout = tf.reshape(Xout, shape=[-1, 7, 7, 256])  # 7
        Xout = deconv(Xout, is_training, kernel_w=4, stride=2, epf=2, Cout=128, act='reLu', norm='batchnorm', name='deconv1')  # 14
        Xout = deconv(Xout, is_training, kernel_w=4, stride=2, epf=2, Cout=Cout, act=None, norm=None, name='deconv2')  # 28
        Xout = tf.nn.sigmoid(Xout)
    return Xout

def create_Discriminator(Xin, is_training, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        Xout = conv(Xin, is_training, kernel_w=4, stride=2, pad=1, Cout=128, act='lrelu', norm=None, name='conv1')  # 14
        Xout = conv(Xout, is_training, kernel_w=4, stride=2, pad=1, Cout=256  , act='lrelu', norm='batchnorm', name='conv2')  # 7
        Xout = conv(Xout, is_training, kernel_w=3, stride=1, pad=None, Cout=1, act=None, norm='batchnorm', name='conv4')  # 5
        Xout = tf.nn.sigmoid(Xout)
    return Xout

def create_dcgan_trainer(base_lr=1e-4, latentD=100, networktype='dcgan'):
    '''Train a Generative Adversarial Network'''
    eps = 1e-8
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Zph = tf.placeholder(tf.float32, [None, latentD])  # tf.random_uniform(shape=[batch_size, 100], minval=-1., maxval=1., dtype=tf.float32)
    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])

    Gout_op = create_generator(Zph, is_training, Cout=1, reuse=False, networktype=networktype + '_G') 

    fakeLogits = create_Discriminator(Gout_op, is_training, reuse=False, networktype=networktype + '_D')
    realLogits = create_Discriminator(Xph, is_training, reuse=True, networktype=networktype + '_D')
    
    G_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(G_varlist), [var.name for var in G_varlist])

    D_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(D_varlist), [var.name for var in D_varlist])
          
    Gloss = clipped_crossentropy(fakeLogits, tf.ones_like(fakeLogits))
    Dloss = clipped_crossentropy(fakeLogits, tf.zeros_like(fakeLogits)) + clipped_crossentropy(realLogits, tf.ones_like(realLogits))
    
    Gtrain_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Gloss, var_list=G_varlist)
    Dtrain_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Dloss, var_list=D_varlist)
    
    return Gtrain_op, Dtrain_op, Gloss, Dloss, is_training, Zph, Xph, Gout_op

if __name__ == '__main__':
    networktype = 'DCGAN_MNIST'
    
    batch_size = 128
    base_lr = 2e-4
    epochs = 500
    latentD = 2
    disp_every_epoch = 5
    
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    
    data = input_data.read_data_sets(data_dir + '/' + networktype, reshape=False)
    disp_int = disp_every_epoch * int(data.train.num_examples / batch_size)  # every two epochs
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    Gtrain_op, Dtrain_op, Gloss, Dloss, is_training, Zph, Xph, Gout_op = create_dcgan_trainer(base_lr, latentD, networktype)
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
            Z = np.random.uniform(size=[batch_size, latentD], low=-1., high=1.).astype(np.float32)
            X, _ = data.train.next_batch(batch_size)
            
            cur_Dloss, _ = sess.run([Dloss, Dtrain_op], feed_dict={Xph:X, Zph:Z, is_training:True})
            dtemploss += cur_Dloss
            
            if it % disp_int == 0:disp_losses = True
             
        cur_Dloss = dtemploss / k   
        
        Z = np.random.uniform(size=[batch_size, latentD], low=-1., high=1.).astype(np.float32)     
        cur_Gloss, _ = sess.run([Gloss, Gtrain_op], feed_dict={Zph:Z, is_training:True})
    
        if disp_losses:
            Gsample = sess.run(Gout_op, feed_dict={Zph: Z, is_training:False})
            vis_square(Gsample[:121], [11, 11], save_path=work_dir + 'Epoch%.3d.jpg' % data.train.epochs_completed)
            saver.save(sess, work_dir + "%.3d_model.ckpt" % data.train.epochs_completed)
            print("Epoch #%.3d, Train Gloss = %f, Dloss=%f" % (data.train.epochs_completed, cur_Gloss, cur_Dloss))
            disp_losses = False
