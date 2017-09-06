# -*- coding: utf-8 -*-
'''
Improved Triaing of Wasserstein GANs - Gulrajani et al. 2017
 
Use this code with no warranty and please respect the accompanying license.
'''
from datetime import datetime

import sys, os
sys.path.append('../common')

from tools_general import tf, np
from tools_config import data_dir, expr_dir
from tools_train import get_train_params, OneHot, vis_square, count_model_params
from tools_networks import deconv, conv, dense

from tensorflow.examples.tutorials.mnist import input_data

import logging
  
def create_generator(z, is_training, Cout=1, reuse=False, networktype='ganG'):
    '''input : batchsize * latentD
       output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        Gout = dense(z, is_training, Cout=7 * 7 * 256, act='reLu', norm='batchnorm', name='dense1')
        Gout = tf.reshape(Gout, shape=[-1, 7, 7, 256])  # 7
        Gout = deconv(Gout, is_training, kernel_w=4, stride=2, epf=2, Cout=128, act='reLu', norm='batchnorm', name='deconv1')  # 14
        Gout = deconv(Gout, is_training, kernel_w=4, stride=2, epf=2, Cout=Cout, act=None, norm=None, name='deconv2')  # 28
        Gout = tf.nn.sigmoid(Gout)
    return Gout

def create_discriminator(xz, is_training, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        Dout = conv(xz, is_training, kernel_w=4, stride=2, pad=1, Cout=128, act='lrelu', norm=None, name='conv1')  # 14
        Dout = conv(Dout, is_training, kernel_w=4, stride=2, pad=1, Cout=256  , act='lrelu', norm=None, name='conv2')  # 7
        Dout = conv(Dout, is_training, kernel_w=3, stride=1, pad=None, Cout=1, act=None, norm=None, name='conv4')  # 5
        Dout = tf.nn.sigmoid(Dout)
    return Dout

def create_wgan2_trainer(base_lr=1e-4, networktype='dcgan', latentD=100):
    '''Train a Wasserstein Generative Adversarial Network with Gradient Penalty'''
    gp_lambda = 10.
    
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    Zph = tf.placeholder(tf.float32, [None, latentD])
    Xph = tf.placeholder(tf.float32, [None, 28, 28, 1])

    Xgen_op = create_generator(Zph, is_training, Cout=1, reuse=False, networktype=networktype + '_G') 

    fakeLogits = create_discriminator(Xgen_op, is_training, reuse=False, networktype=networktype + '_D')
    realLogits = create_discriminator(Xph, is_training, reuse=True, networktype=networktype + '_D')
    
    gen_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    logging.info('# of Trainable vars in Generator:%d -- %s' % (len(gen_varlist), '; '.join([var.name.split('/')[1] for var in gen_varlist])))

    dis_varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    logging.info('# of Trainable vars in Discriminator:%d -- %s' % (len(dis_varlist), '; '.join([var.name.split('/')[1] for var in dis_varlist])))
    
    batch_size = tf.shape(fakeLogits)[0]
    epsilon = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)

    Xhat = epsilon * Xph + (1 - epsilon) * Xgen_op
    D_Xhat = create_discriminator(Xhat, is_training, reuse=True, networktype=networktype + '_D')
    
    ddx = tf.gradients(D_Xhat, [Xhat])[0]
    ddx_norm = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
    gradient_penalty = tf.reduce_mean(tf.square(ddx_norm - 1.0) * gp_lambda)
    
    dis_loss_op = tf.reduce_mean(fakeLogits) - tf.reduce_mean(realLogits) + gradient_penalty   
    gen_loss_op = -tf.reduce_mean(tf.abs(fakeLogits))
           
    gen_train_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(gen_loss_op, var_list=gen_varlist)
    dis_train_op = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(dis_loss_op, var_list=dis_varlist)
    
    logging.info('Total Trainable Variables Count in Generator %2.3f M and in Discriminator: %2.3f M.' % (count_model_params(gen_varlist) * 1e-6, count_model_params(dis_varlist) * 1e-6,))

    return gen_train_op, dis_train_op, gen_loss_op, dis_loss_op, is_training, Zph, Xph, Xgen_op

if __name__ == '__main__':
    ''' Since discriminator in this gan is not trained to classifiy 
    it is called a critic in the paper but i will stick to D, as i like it.'''
    
    networktype = 'WGAN2_MNIST'
    
    batch_size = 128
    base_lr = 1e-4  
    epochs = 500
    latentD = 2
    disp_every_epoch = 5
    
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
    
    data = input_data.read_data_sets(data_dir + '/' + networktype, reshape=False)
    disp_int = disp_every_epoch * int(np.ceil(data.train.num_examples / batch_size))  # every two epochs
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    gen_train_op, dis_train_op, gen_loss_op, dis_loss_op, is_training, Zph, Xph, Xgen_op = create_wgan2_trainer(base_lr, networktype, latentD)
    tf.global_variables_initializer().run()
    
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(epochs * 0.1))
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt')  
    
    k = 5 
    it = 0   
    disp_losses = False    
    while data.train.epochs_completed < epochs:
        cur_Dloss = 0
        for itD in range(k):
            it += 1
            Z = np.random.uniform(size=[batch_size, latentD], low=-1., high=1.).astype(np.float32)
            X, _ = data.train.next_batch(batch_size)
            dtemploss, _ = sess.run([dis_loss_op, dis_train_op], feed_dict={Xph:X, Zph:Z, is_training:True})
            cur_Dloss += dtemploss
            
            if it % disp_int == 0:disp_losses = True
                            
        cur_Dloss = dtemploss / k
            
        Z = np.random.uniform(size=[batch_size, latentD], low=-1., high=1.).astype(np.float32)
        cur_Gloss, _ = sess.run([gen_loss_op, gen_train_op], feed_dict={Zph:Z, is_training:True})
        
        if disp_losses:
            Gsample = sess.run(Xgen_op, feed_dict={Zph: Z, is_training:False})
            vis_square(Gsample[:121], [11, 11], save_path=work_dir + 'Gen_Iter_%d.jpg' % it)
            saver.save(sess, work_dir + "Model_Iter_%.3d.ckpt" % it)
            logging.info("Epoch #%.3d, Train Generator Loss = %2.5f, Discriminator Loss=%2.5f" % (data.train.epochs_completed, cur_Gloss, Dloss))
            disp_losses = False
    
    endtime = datetime.now().replace(microsecond=0)
    logging.info('Finished Training of %s at %s' % (networktype, datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
    logging.info('Training done in %s ! Best Test Loss [rec| dis| enc] = [%s]' % (endtime - starttime, ' | '.join(['%2.5f' % a for a in best_test_loss])))

