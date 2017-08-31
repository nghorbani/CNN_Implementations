# -*- coding: utf-8 -*-
'''
Auto-Encoding Variational Bayes - Kingma and Welling 2013

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
  
def create_VAE_E(Xin, is_training, latentW, latentC, reuse=False, networktype='vaeE'):
    '''Xin: batchsize * H * W * Cin
       output1-2: batchsize * Cout'''
    with tf.variable_scope(networktype, reuse=reuse):
        Eout = conv(Xin, is_training, kernel_w=4, stride=2, Cout=64, pad=1, act='reLu', norm='batchnorm', name='conv1')  # 14*14
        Eout = conv(Eout, is_training, kernel_w=4, stride=2, Cout=128, pad=1, act='reLu', norm='batchnorm', name='conv2')  # 7*7
        
        posteriorMu = conv(Eout, is_training, kernel_w=3, stride=1, Cout=latentC, pad=1, act=None, norm=None, name='conv_mu')
        posteriorSigma = conv(Eout, is_training, kernel_w=3, stride=1, Cout=latentC, pad=1, act=None, norm=None, name='conv_sig')

        posteriorMu = tf.reshape(posteriorMu, shape=[-1, latentW * latentW * latentC])
        posteriorSigma = tf.reshape(posteriorSigma, shape=[-1, latentW * latentW * latentC])
        
    return posteriorMu, posteriorSigma

def create_VAE_D(z, is_training, Cout, latentW, latentC, reuse=False, networktype='vaeD'):
    '''input : batchsize * latentDim
       output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        print("Latent Space Dim = H=%d, W=%d, C=%d" % (latentW, latentW, latentC))
        Gout = tf.reshape(z, shape=[-1, latentW, latentW, latentC])
        Gout = deconv(Gout, is_training, kernel_w=4, stride=2, epf=2, Cout=128, act='reLu', norm='batchnorm', name='deconv1')  # 14
        Gout = deconv(Gout, is_training, kernel_w=4, stride=2, epf=2, Cout=Cout, act=None, norm=None, name='deconv2')  # 28
        Gout = tf.nn.sigmoid(Gout)
    return Gout

def create_vae_trainer(base_lr=1e-4, networktype='VAE', Cout=1, latentW=7, latentC=2):
    '''Train a Variational AutoEncoder'''
    eps = 1e-5
    
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Zph = tf.placeholder(tf.float32, [None, latentW * latentW * latentC])
    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])

    posteriorMu, posteriorSigma = create_VAE_E(Xph, is_training, latentW, latentC, reuse=False, networktype=networktype + '_E') 
    Z_op = posteriorSigma * Zph + posteriorMu
    Xrec_op = create_VAE_D(Z_op, is_training, Cout, latentW, latentC, reuse=False, networktype=networktype + '_D')
    
    # E[log P(X|z)]
    # rec_loss_op = tf.reduce_mean(tf.reduce_sum((Xph - 1.0) * tf.log(1.0 - Xrec_op + eps) - Xph * tf.log(Xrec_op + eps), reduction_indices=[1, 2, 3]))
    rec_loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(Xph, Xrec_op)), reduction_indices=[1, 2, 3]))

    # D_KL(Q(z|X) || P(z))
    KL_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(posteriorSigma) + tf.square(posteriorMu) - 1 - posteriorSigma, reduction_indices=[1, ]))
    
    total_loss_op = tf.add(rec_loss_op , KL_loss)  
    train_step_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.9).minimize(total_loss_op)
 
    E_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_E')    
    D_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_D')
    print('Total Trainable Variables Count in Encoder %2.3f M and in Decoder: %2.3f M.' % (count_model_params(E_varlist) / 1000000, count_model_params(D_varlist) / 1000000,))

    return train_step_op, total_loss_op, rec_loss_op, KL_loss, is_training, Zph, Xph, Xrec_op

if __name__ == '__main__':
    networktype = 'VAE_MNIST'
    
    batch_size = 128
    base_lr = 1e-5
    epochs = 200
    
    Cout = 1
    
    latentW = 7
    latentC = 2
    latendDim = latentW * latentW * latentC
    
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    
    data, max_iter, test_iter, test_int, disp_int = get_train_params(data_dir + '/' + networktype, batch_size, epochs=epochs, test_in_each_epoch=1, networktype=networktype)
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    train_step_op, total_loss_op, rec_loss_op, KL_loss, is_training, Zph, Xph, Xrec_op = create_vae_trainer(base_lr, networktype, Cout, latentW, latentC)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * .1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt') 
     
    best_test_total_loss = np.inf 
 
    train_loss = np.zeros([max_iter,3])
    test_loss = np.zeros([int(np.ceil(max_iter / test_int)),3])
         
    for it in range(max_iter): 
        Z = np.random.normal(size=[batch_size, latendDim], loc=0.0, scale=1.).astype(np.float32)
  
        if it % test_int == 0:  # Record summaries and test-set accuracy
            acc_loss = np.zeros([1,3])
            for i_test in range(test_iter):
                X, _ = data.test.next_batch(batch_size)
                resloss = sess.run([total_loss_op, rec_loss_op, KL_loss], feed_dict={Xph:X, Zph: Z, is_training:False})
                acc_loss = np.add(acc_loss, resloss)
                 
            test_loss[it // test_int] = np.divide(acc_loss, test_iter)
     
            print("Iteration #%4d, testing .... Test Loss [total| rec| KL] = " % it, test_loss[it // test_int])
            if test_loss[it // test_int,0] < best_test_total_loss:
                best_test_total_loss = test_loss[it // test_int,0]
                print('################ Best Results yet.[loss = %2.5f] saving results...' % best_test_total_loss)
                vaeD_sample = sess.run(Xrec_op, feed_dict={Xph:X, Zph: Z , is_training:False})
                vis_square(vaeD_sample[:121], [11, 11], save_path=work_dir + 'Epoch_%.3d_Iter_%d.jpg' % (data.train.epochs_completed, it))
                saver.save(sess, work_dir + "%.3d_%.3d_model.ckpt" % (data.train.epochs_completed, it))
         
        X, _ = data.train.next_batch(batch_size)
        recloss, _ = sess.run([total_loss_op, train_step_op], feed_dict={Xph:X, Zph: Z, is_training:True})
         
        train_loss[it] = recloss
        if it % disp_int == 0:print("Iteration #%4d, Train Loss = %f" % (it, recloss))
