# -*- coding: utf-8 -*-
'''
Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion. Vincent et al. 2010

Use this code with no warranty and please respect the accompanying license.
'''

import sys
sys.path.append('../common')

from tools_config import data_dir, expr_dir
import os
import matplotlib.pyplot as plt
from tools_train import get_train_params, OneHot, vis_square, count_model_params
from datetime import datetime
from tools_general import tf, np
from tools_networks import deconv, conv, dense, clipped_crossentropy, dropout

import logging

def create_encoder(Xin, is_training, latentD, reuse=False, networktype='cdaeE'):
    '''Xin: batchsize * H * W * Cin
       output1-2: batchsize * Cout'''
    with tf.variable_scope(networktype, reuse=reuse):
        Xout = conv(Xin, is_training, kernel_w=4, stride=2, Cout=64, pad=1, act='reLu', norm='batchnorm', name='conv1')  # 14*14
        Xout = conv(Xout, is_training, kernel_w=4, stride=2, Cout=128, pad=1, act='reLu', norm='batchnorm', name='conv2')  # 7*7
        Xout = dense(Xout, is_training, Cout=latentD, act=None, norm=None, name='dense_mean')
    return Xout 

def create_decoder(Xin, is_training, latentD, Cout=1, reuse=False, networktype='cdaeD'):
    with tf.variable_scope(networktype, reuse=reuse):
        Xout = dense(Xin, is_training, Cout=7 * 7 * 256, act='reLu', norm='batchnorm', name='dense1')
        Xout = tf.reshape(Xout, shape=[-1, 7, 7, 256])  # 7
        Xout = deconv(Xout, is_training, kernel_w=4, stride=2, Cout=256, epf=2, act='reLu', norm='batchnorm', name='deconv1')  # 14
        Xout = deconv(Xout, is_training, kernel_w=4, stride=2, Cout=Cout, epf=2, act=None, norm=None, name='deconv2')  # 28
        Xout = tf.nn.sigmoid(Xout)
    return Xout
    
def create_cdae_trainer(base_lr=1e-4, latentD=2, networktype='CDAE'):
    '''Train a Variational AutoEncoder'''
    eps = 1e-5
    
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
    Xc_op = tf.cond(is_training, lambda: tf.nn.dropout(Xph, keep_prob=0.75), lambda: tf.identity(Xph))
    Xenc_op = create_encoder(Xc_op, is_training, latentD, reuse=False, networktype=networktype + '_Enc') 
    Xrec_op = create_decoder(Xenc_op, is_training, latentD, reuse=False, networktype=networktype + '_Dec')
    
    # reconstruction loss
    rec_loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(Xph, Xrec_op)), reduction_indices=[1, 2, 3]))

    Enc_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_Enc')    
    Dec_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_Dec')
    
    total_loss_op = rec_loss_op 
    train_step_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.9).minimize(total_loss_op, var_list=Enc_varlist + Dec_varlist)
 
    print('Total Trainable Variables Count in Encoder %2.3f M and in Decoder: %2.3f M.' % (count_model_params(Enc_varlist) * 1e-6, count_model_params(Dec_varlist) * 1e-6,))

    return train_step_op, rec_loss_op, is_training, Xph, Xrec_op

if __name__ == '__main__':
    networktype = 'CDAE_MNIST'
    
    batch_size = 128
    
    base_lr = 1e-3
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
    
    data, max_iter, test_iter, test_int, disp_int = get_train_params(data_dir, batch_size, epochs=epochs, test_in_each_epoch=1, networktype=networktype)
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    train_step_op, rec_loss_op, is_training, Xph, Xrec_op = create_cdae_trainer(base_lr, latentD, networktype)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * .1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt') 
     
    best_test_rec_loss = np.inf 
 
    train_loss = np.zeros([max_iter, 1])
    test_loss = np.zeros([int(np.ceil(max_iter / test_int)), 1])
         
    for it in range(max_iter): 
          
        if it % test_int == 0:  # Record summaries and test-set accuracy
            acc_loss = np.zeros([1, 1])
            for i_test in range(test_iter):
                X, _ = data.test.next_batch(batch_size)
                resloss = sess.run([rec_loss_op, ], feed_dict={Xph:X, is_training:False})
                acc_loss = np.add(acc_loss, resloss)
                 
            test_loss[it // test_int] = np.divide(acc_loss, test_iter)
     
            logging.info("Epoch %4d, Iteration #%4d, Test Reconstruction Loss = %2.5f" % (data.train.epochs_completed, it, test_loss[it // test_int, 0]))
            if test_loss[it // test_int, 0] < best_test_rec_loss:
                best_test_rec_loss = test_loss[it // test_int, 0]
                logging.info('### Best Test Reconstruction Loss yet.[%2.5f]' % best_test_rec_loss)
                cdae_sample = sess.run(Xrec_op, feed_dict={Xph:X, is_training:False})
                vis_square(cdae_sample[:121], [11, 11], save_path=work_dir + 'Rec_Iter_%d.jpg' % it)
                saver.save(sess, work_dir + "Model_Iter_%.3d.ckpt" % it)
         
        X, _ = data.train.next_batch(batch_size)
        recloss, _ = sess.run([rec_loss_op, train_step_op], feed_dict={Xph:X, is_training:True})
         
        train_loss[it] = recloss
        if it % disp_int == 0:logging.info("Iteration #%4d, Train Loss = %f" % (it, train_loss[it]))
    
    endtime = datetime.now().replace(microsecond=0)
    logging.info('Finished Training of %s at %s' % (networktype, datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
    logging.info('Training done in %s ! Best Test Reconstruction Loss = %2.5f' % (endtime - starttime, best_test_rec_loss))
