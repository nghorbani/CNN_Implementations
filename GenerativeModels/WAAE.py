# -*- coding: utf-8 -*-
'''
Adversarial Autoencoder. Makhzani et al. 2015
Improved Triaing of Wasserstein GANs - Gulrajani et al. 2017

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

def create_encoder(Xin, is_training, latentD, reuse=False, networktype='cdaeE'):
    '''Xin: batchsize * H * W * Cin
       output1-2: batchsize * Cout'''
    with tf.variable_scope(networktype, reuse=reuse):
        Xout = conv(Xin, is_training, kernel_w=4, stride=2, Cout=64, pad=1, act='reLu', norm='batchnorm', name='conv1')  # 14*14
        Xout = conv(Xout, is_training, kernel_w=4, stride=2, Cout=128, pad=1, act='reLu', norm='batchnorm', name='conv2')  # 7*7
        Xout = dense(Xout, is_training, Cout=latentD, act=None, norm=None, name='dense_mean')
    return Xout 

def create_decoder(Xin, is_training, latentD, Cout=1, reuse=False, networktype='vaeD'):
    with tf.variable_scope(networktype, reuse=reuse):
        Xout = dense(Xin, is_training, Cout=7 * 7 * 256, act='reLu', norm='batchnorm', name='dense1')
        Xout = tf.reshape(Xout, shape=[-1, 7, 7, 256])  # 7
        Xout = deconv(Xout, is_training, kernel_w=4, stride=2, Cout=256, epf=2, act='reLu', norm='batchnorm', name='deconv1')  # 14
        Xout = deconv(Xout, is_training, kernel_w=4, stride=2, Cout=Cout, epf=2, act=None, norm=None, name='deconv2')  # 28
        Xout = tf.nn.sigmoid(Xout)
    return Xout

def create_discriminator(Xin, is_training, reuse=False, networktype='ganD'):
   with tf.variable_scope(networktype, reuse=reuse):
        Xout = dense(Xin, is_training, Cout=7 * 7 * 256, act='reLu', norm=None, name='dense1')
        Xout = tf.reshape(Xout, shape=[-1, 7, 7, 256])  # 7
        Xout = conv(Xout, is_training, kernel_w=3, stride=1, pad=1, Cout=128, act='lrelu', norm=None, name='conv1')  # 7
        Xout = conv(Xout, is_training, kernel_w=3, stride=1, pad=1, Cout=256  , act='lrelu', norm=None, name='conv2')  # 7
        Xout = conv(Xout, is_training, kernel_w=3, stride=1, pad=None, Cout=1, act=None, norm=None, name='conv3')  # 5
        Xout = tf.nn.sigmoid(Xout)
   return Xout
   
def create_aae_trainer(base_lr=1e-4, latentD=2, networktype='AAE'):
    '''Train an Adversarial Autoencoder'''
    gp_lambda = 10.

    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Zph = tf.placeholder(tf.float32, [None, latentD])
    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])
    
    Xc_op = tf.cond(is_training, lambda: tf.nn.dropout(Xph, keep_prob=0.75), lambda: tf.identity(Xph))
    Z_op = create_encoder(Xc_op, is_training, latentD, reuse=False, networktype=networktype + '_Enc') 
    Xrec_op = create_decoder(Z_op, is_training, latentD, reuse=False, networktype=networktype + '_Dec')
    Xgen_op = create_decoder(Zph, is_training, latentD, reuse=True, networktype=networktype + '_Dec')
    
    fakeLogits = create_discriminator(Z_op, is_training, reuse=False, networktype=networktype + '_Dis')
    realLogits = create_discriminator(Zph, is_training, reuse=True, networktype=networktype + '_Dis')
    
    # reconstruction loss
    rec_loss_op = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(Xph, Xrec_op)), reduction_indices=[1, 2, 3]))

    # regularization loss
    batch_size = tf.shape(fakeLogits)[0]
    epsilon = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)

    Zhat = epsilon * Zph + (1 - epsilon) * Z_op
    D_Zhat = create_discriminator(Zhat, is_training, reuse=True, networktype=networktype + '_Dis')
    
    ddz = tf.gradients(D_Zhat, [Zhat])[0]
    ddz_norm = tf.sqrt(tf.reduce_sum(tf.square(ddz), axis=1))
    gradient_penalty = tf.reduce_mean(tf.square(ddz_norm - 1.0) * gp_lambda)
    
    dec_loss_op = rec_loss_op
    dis_loss_op = tf.reduce_mean(fakeLogits) - tf.reduce_mean(realLogits) + gradient_penalty   
    enc_gen_loss_op = -tf.reduce_mean(tf.abs(fakeLogits)) + 0.1 * rec_loss_op
    enc_rec_loss_op = -tf.reduce_mean(tf.abs(fakeLogits)) + 10 * rec_loss_op
    
    enc_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_Enc')    
    dec_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_Dec')
    dis_varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=networktype + '_Dis')
    
    train_dec_op = tf.train.AdamOptimizer(learning_rate=1.0 * base_lr, beta1=0.5).minimize(dec_loss_op, var_list=dec_varlist)    
    train_enc_rec_op = tf.train.AdamOptimizer(learning_rate=1.0 * base_lr, beta1=0.5).minimize(enc_rec_loss_op, var_list=enc_varlist)
    train_enc_gen_op = tf.train.AdamOptimizer(learning_rate=1.0 * base_lr, beta1=0.5).minimize(enc_gen_loss_op, var_list=enc_varlist)
    train_dis_op = tf.train.AdamOptimizer(learning_rate=1.0 * base_lr, beta1=0.5).minimize(dis_loss_op, var_list=dis_varlist)
 
    logging.info('Total Trainable Variables Count in Encoder %2.3f M, Decoder: %2.3f M, and Discriminator: %2.3f' 
          % (count_model_params(enc_varlist) * 1e-6, count_model_params(dec_varlist) * 1e-6, count_model_params(dis_varlist) * 1e-6))

    return train_dec_op, train_dis_op, train_enc_gen_op, train_enc_rec_op, rec_loss_op, dis_loss_op, enc_gen_loss_op, is_training, Zph, Xph, Xrec_op, Xgen_op

if __name__ == '__main__':
    exp_id = 1
    networktype = 'WAAE_MNIST'
    
    batch_size = 128
    
    base_lr = 1e-4

    epochs = 400
        
    latentD = 2
    
    work_dir = expr_dir + '%s/%.2d/' % (networktype, exp_id)
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    else: raise ValueError('Experiment folder already exists. You probably wnt to change the experiment ID.')
    
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
    test_int = test_int * 3
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    train_dec_op, train_dis_op, train_enc_gen_op, train_enc_rec_op, rec_loss_op, dis_loss_op, enc_gen_loss_op, is_training, Zph, Xph, Xrec_op, Xgen_op = \
                                                                                    create_aae_trainer(base_lr, latentD, networktype)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * .1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt') 
     
    best_test_loss = np.ones([3, ]) * np.inf 
 
    train_loss = np.zeros([max_iter, 3])
    test_loss = np.zeros([int(np.ceil(max_iter / test_int)), 3])
    k = 5
    for it in range(max_iter):
        
        X, _ = data.train.next_batch(batch_size)
        Z = np.random.normal(size=[batch_size, latentD], loc=0.0, scale=1.).astype(np.float32)
        # 1- Train the Encoder and the Decoder for reconstructing the input
        sess.run(train_dec_op, feed_dict={Xph:X, is_training:True})        
        # 2- Train the Discriminator 
        for _ in range(k):
            dis_loss, _ = sess.run([dis_loss_op, train_dis_op], feed_dict={Xph:X, Zph:Z, is_training:True})
        # 3 - Train the Generator (Encoder)
        sess.run(train_enc_rec_op, feed_dict={Xph:X, is_training:True})
        enc_loss, rec_loss, _ = sess.run([enc_gen_loss_op, rec_loss_op, train_enc_gen_op], feed_dict={Xph:X, is_training:True})
        
        if it % test_int == 0:  # Record summaries and test-set accuracy
            acc_loss = np.zeros([1, 3])
            for i_test in range(test_iter):
                X, _ = data.test.next_batch(batch_size)
                resloss = sess.run([rec_loss_op, dis_loss_op, enc_gen_loss_op], feed_dict={Xph:X, Zph: Z, is_training:False})
                acc_loss = np.add(acc_loss, resloss)
                 
            test_loss[it // test_int] = np.divide(acc_loss, test_iter)
     
            logging.info("Epoch %4d, Iteration #%4d, Test Loss [rec| dis| enc] = [%s]" % (data.train.epochs_completed, it, ' | '.join(['%2.5f' % a for a in test_loss[it // test_int]])))
            if test_loss[it // test_int, 0] < best_test_loss[0]:
                best_test_loss = test_loss[it // test_int]
                logging.info("### Best Test Results Yet. Test Loss [rec| dis| enc] = [%s]" % (' | '.join(['%2.5f' % a for a in test_loss[it // test_int]])))
                rec_sample = sess.run(Xrec_op, feed_dict={Xph:X, is_training:False})
                vis_square(rec_sample[:121], [11, 11], save_path=work_dir + 'Rec_Iter_%d.jpg' % it)
                
                gen_sample = sess.run(Xgen_op, feed_dict={Zph:Z, is_training:False})
                vis_square(gen_sample[:121], [11, 11], save_path=work_dir + 'Gen_Iter_%d.jpg' % it)

                saver.save(sess, work_dir + "Model_Iter_%.3d.ckpt" % it)
                 
        train_loss[it] = [rec_loss, dis_loss, enc_loss]
#         if it % disp_int == 0: 
#             logging.info("Epoch %4d, Iteration #%4d, Train Loss [rec| dis| enc] = [%s]" % (data.train.epochs_completed, it, ' | '.join(['%2.5f' % a for a in train_loss[it]])))

    endtime = datetime.now().replace(microsecond=0)
    logging.info('Finished Training of %s at %s' % (networktype, datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
    logging.info('Training done in %s ! Best Test Loss [rec| dis| enc] = [%s]' % (endtime - starttime, ' | '.join(['%2.5f' % a for a in best_test_loss])))
