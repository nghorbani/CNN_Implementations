# -*- coding: utf-8 -*-
'''
Generative Adversarial Networks - Goodfellow et al. 2014
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - Radford et al. 2015

Use this code with no warranty and please respect the accompanying license.
'''

import sys
sys.path.append('../common')

from tools_config import data_dir, expr_dir
import os, sys, shutil
import matplotlib.pyplot as plt
from tools_train import get_train_params, OneHot, vis_square, count_model_params
from datetime import datetime
from tools_general import tf, np
from tools_networks import deconv, conv, dense, clipped_crossentropy, dropout

import logging

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

def create_discriminator(Xin, is_training, reuse=False, networktype='ganD'):
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

    fakeLogits = create_discriminator(Gout_op, is_training, reuse=False, networktype=networktype + '_D')
    realLogits = create_discriminator(Xph, is_training, reuse=True, networktype=networktype + '_D')
          
    gen_loss_op = clipped_crossentropy(fakeLogits, tf.ones_like(fakeLogits))
    dis_loss_op = clipped_crossentropy(fakeLogits, tf.zeros_like(fakeLogits)) + clipped_crossentropy(realLogits, tf.ones_like(realLogits))
    
    gen_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    logging.info('# of Trainable vars in Generator:%d -- %s' % (len(gen_varlist), '; '.join([var.name.split('/')[1] for var in gen_varlist])))

    dis_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    logging.info('# of Trainable vars in Discriminator:%d -- %s' % (len(dis_varlist), '; '.join([var.name.split('/')[1] for var in dis_varlist])))
    
    gen_train_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(gen_loss_op, var_list=gen_varlist)
    dis_train_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(dis_loss_op, var_list=dis_varlist)

    logging.info('Total Trainable Variables Count in Generator %2.3f M and in Discriminator: %2.3f M.' % (count_model_params(gen_varlist) * 1e-6, count_model_params(dis_varlist) * 1e-6,))

    return gen_train_op, dis_train_op, gen_loss_op, dis_loss_op, is_training, Zph, Xph, Gout_op

if __name__ == '__main__':
    networktype = 'DCGAN_MNIST'
    
    batch_size = 128
    
    base_lr = 2e-4
    epochs = 100
    
    latentD = 2
        
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    
    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
    
    logging.basicConfig(filename=work_dir + '%s.log' % log_name, level=logging.DEBUG, format='%(asctime)s :: %(message)s', datefmt='%Y%m%d-%H%M%S')
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info('Started Training of %s at %s' % (networktype, datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S')))
    logging.info('\nTraining Hyperparamters: batch_size= %d, base_lr= %1.1e, epochs= %d, latentD= %d\n' % (batch_size, base_lr, epochs, latentD))

    shutil.copy2(os.path.basename(sys.argv[0]), work_dir)   

    data, max_iter, test_iter, test_int, disp_int = get_train_params(data_dir, batch_size, epochs=epochs, test_in_each_epoch=1, networktype=networktype)
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    gen_train_op, dis_train_op, gen_loss_op, dis_loss_op, is_training, Zph, Xph, Gout_op = create_dcgan_trainer(base_lr, latentD, networktype)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * 0.1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt')  
        
    k = 1
    
    it = 0
    for it in range(max_iter): 
        X, _ = data.train.next_batch(batch_size)
        
        dtemploss = 0         
        for itD in range(k):
            Z = np.random.uniform(size=[batch_size, latentD], low=-1., high=1.).astype(np.float32)
            Dloss, _ = sess.run([dis_loss_op, dis_train_op], feed_dict={Xph:X, Zph:Z, is_training:True})
            dtemploss += Dloss             
        Dloss = dtemploss / k   
        
        cur_Gloss, _ = sess.run([gen_loss_op, gen_train_op], feed_dict={Zph:Z, is_training:True})
    
        if it % disp_int == 0:
            Gsample = sess.run(Gout_op, feed_dict={Zph: Z, is_training:False})
            vis_square(Gsample[:121], [11, 11], save_path=work_dir + 'Gen_Iter_%d.jpg' % it)
            saver.save(sess, work_dir + "Model_Iter_%.3d.ckpt" % it)
            logging.info("Epoch #%.3d, Train Generator Loss = %2.5f, Discriminator Loss=%2.5f" % (data.train.epochs_completed, cur_Gloss, Dloss))
    
    endtime = datetime.now().replace(microsecond=0)
    logging.info('Finished Training of %s at %s' % (networktype, datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
    logging.info('Training done in %s !' % (endtime - starttime))
