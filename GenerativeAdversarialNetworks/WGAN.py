# -*- coding: utf-8 -*-
'''
Wasserstein GAN - Arjovsky et al. 2017

This work is absolutely not an effort to reproduce exact results of the cited paper, nor I confine my implementations to the suggestion of the original authors.
I have tried to implement my own limited understanding of the original paper in hope to get a better insight into their work. 
Use this code with no warranty and please respect the accompanying license.
'''
from datetime import datetime

import sys, os
sys.path.append('../common')

from tools_general import tf, np
from tools_config import data_dir, expr_dir
from tools_train import vis_square
from tools_networks import deconv, conv, dense

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
    '''Train a Wasserstein Generative Adversarial Network'''

    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Zph = tf.placeholder(tf.float32, [None, latentDim])
    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])

    Gout_op = create_gan_G(Zph, is_training, Cout=1, trainable=True, reuse=False, networktype=networktype + '_G') 

    fakeLogits = create_gan_D(Gout_op, is_training, trainable=True, reuse=False, networktype=networktype + '_D')
    realLogits = create_gan_D(Xph, is_training, trainable=True, reuse=True, networktype=networktype + '_D')
    
    G_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(G_varlist), [var.name for var in G_varlist])

    D_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(D_varlist), [var.name for var in D_varlist])

    Dloss = tf.reduce_mean(realLogits) - tf.reduce_mean(fakeLogits)
    Gloss = tf.reduce_mean(tf.abs(fakeLogits))
    
    D_weights = [var for var in D_varlist if '_W' in var.name]
    D_weights_clip_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in D_weights]
                
    # Gtrain_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Gloss, var_list=G_varlist)
    # Dtrain_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Dloss, var_list=D_varlist)
    
    Gtrain_op = tf.train.RMSPropOptimizer(learning_rate=base_lr, decay=0.9).minimize(Gloss, var_list=G_varlist)
    Dtrain_op = tf.train.RMSPropOptimizer(learning_rate=base_lr, decay=0.9).minimize(Dloss, var_list=D_varlist)

    return Gtrain_op, Dtrain_op, D_weights_clip_op, Gloss, Dloss, is_training, Zph, Xph, Gout_op

if __name__ == '__main__':
    networktype = 'WGAN_MNIST'
    
    batch_size = 128
    base_lr = 5e-5  
    epochs = 1000
    latentDim = 100
    
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
        
    data = input_data.read_data_sets(data_dir + '/' + networktype, reshape=False)
    disp_int = 2 * int(data.train.num_examples / batch_size) #every two epochs
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    Gtrain_op, Dtrain_op, D_weights_clip_op, Gloss, Dloss, is_training, Zph, Xph, Gout_op = create_dcgan_trainer(base_lr, networktype, latentDim)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * 0.1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt')  
        
    it = 0       
    while data.train.epochs_completed < epochs:
        k = 100 if it < 25 or it % 500 == 0 else 5  # from the original pytorch implementation
        dtemploss = 0
        for itD in range(k):
            it += 1
            Z = np.random.uniform(size=[batch_size, latentDim], low=-1., high=1.).astype(np.float32)
            X, _ = data.train.next_batch(batch_size)
        
            cur_Dloss, _ = sess.run([Dloss, Dtrain_op], feed_dict={Xph:X, Zph:Z, is_training:True})
            sess.run(D_weights_clip_op)
            dtemploss += cur_Dloss
            
            if it % disp_int == 0:
                Gsample = sess.run(Gout_op, feed_dict={Zph: Z, is_training:False})
                vis_square(Gsample[:121], [11, 11], save_path=work_dir + 'Epoch%.3d.jpg' % data.train.epochs_completed)
                saver.save(sess, work_dir + "%.3d_model.ckpt" % data.train.epochs_completed)
                if ('cur_Dloss' in vars()) and ('cur_Gloss' in vars()):
                    print("Epoch #%4d, Train Gloss = %f, Dloss=%f" % (data.train.epochs_completed, cur_Gloss, cur_Dloss))
            
        cur_Dloss = dtemploss / k
            
        Z = np.random.uniform(size=[batch_size, latentDim], low=-1., high=1.).astype(np.float32)
        cur_Gloss, _ = sess.run([Gloss, Gtrain_op], feed_dict={Zph:Z, is_training:True})