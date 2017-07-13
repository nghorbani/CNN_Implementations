# -*- coding: utf-8 -*-
'''
Image-to-Image Translation with Conditional Adversarial Networks - Isola et al
'''
from tools_config import *
import os
from tools_train import get_train_params, OneHot, vis_square
from datetime import datetime
from tools_general import tf, np
from tools_networks import clipped_crossentropy, dropout, conv, deconv
from tools_data import retransform

def conch(A, B):
    '''Concatenate channelwise'''
    with tf.variable_scope("deconv"):
        X = tf.concat([A, B], axis=3)
        return X
      
def create_gan_G(GE0, is_training, Cout=3, trainable=True, reuse=False, networktype='ganG'):

    with tf.variable_scope(networktype, reuse=reuse):
        GE1 = conv(GE0, is_training, kernel_w=4, stride=2, Cout=64 , pad=1, trainable=trainable, act='lreLu', norm=None, name='ENconv1')  # 128
        GE2 = conv(GE1, is_training, kernel_w=4, stride=2, Cout=128, pad=1, trainable=trainable, act='lreLu', norm='instance', name='ENconv2')  # 64
        GE3 = conv(GE2, is_training, kernel_w=4, stride=2, Cout=256, pad=1, trainable=trainable, act='lreLu', norm='instance', name='ENconv3')  # 32
        GE4 = conv(GE3, is_training, kernel_w=4, stride=2, Cout=512, pad=1, trainable=trainable, act='lreLu', norm='instance', name='ENconv4')  # 16
        GE5 = conv(GE4, is_training, kernel_w=4, stride=2, Cout=512, pad=1, trainable=trainable, act='lreLu', norm='instance', name='ENconv5')  # 8
        GE6 = conv(GE5, is_training, kernel_w=4, stride=2, Cout=512, pad=1, trainable=trainable, act='lreLu', norm='instance', name='ENconv6')  # 4
        GE7 = conv(GE6, is_training, kernel_w=4, stride=2, Cout=512, pad=1, trainable=trainable, act='lreLu', norm='instance', name='ENconv7')  # 2
        
        GBNeck = conv(GE7, is_training, kernel_w=4, stride=2, Cout=512, pad=1, trainable=trainable, act='lreLu', norm='instance', name='GBNeck')  # 1 - Bottleneck
        
        GD7 = deconv(GBNeck, is_training, kernel_w=4, stride=2, Cout=512, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv1');GD7 = dropout(GD7, is_training, p=0.5);  # 2
        GD6 = deconv(conch(GD7, GE7), is_training, kernel_w=4, stride=2, Cout=512, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv2');GD6 = dropout(GD6, is_training, p=0.5)  # 4
        GD5 = deconv(conch(GD6, GE6), is_training, kernel_w=4, stride=2, Cout=512, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv3');GD5 = dropout(GD5, is_training, p=0.5)  # 8
        GD4 = deconv(conch(GD5, GE5), is_training, kernel_w=4, stride=2, Cout=512, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv4')  # 16
        GD3 = deconv(conch(GD4, GE4), is_training, kernel_w=4, stride=2, Cout=512, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv5')  # 32
        GD2 = deconv(conch(GD3, GE3), is_training, kernel_w=4, stride=2, Cout=256, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv6')  # 64
        GD1 = deconv(conch(GD2, GE2), is_training, kernel_w=4, stride=2, Cout=128, epf=2, trainable=trainable, act='reLu', norm='instance', name='DEdeconv7')  # 128
        GD0 = deconv(conch(GD1, GE1), is_training, kernel_w=4, stride=2, Cout=Cout, epf=2, trainable=trainable, act=None, norm='instance', name='DEdeconv8')  # 256
        
        Xout = tf.nn.tanh(GD0)
        
    return Xout

def create_gan_D(inSource, inTarget, is_training, trainable=True, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        inSource = conch(inSource, inTarget)
        Dxz = conv(inSource, is_training, kernel_w=4, stride=2, Cout=64,  trainable=trainable, act='lrelu', norm=None, name='conv1')  # 128
        Dxz = conv(Dxz, is_training, kernel_w=4, stride=2, Cout=128, trainable=trainable, act='lrelu', norm='instance', name='conv2')  # 64
        Dxz = conv(Dxz, is_training, kernel_w=4, stride=2, Cout=256, trainable=trainable, act='lrelu', norm='instance', name='conv3')  # 32
        Dxz = conv(Dxz, is_training, kernel_w=1, stride=1, Cout=1,   trainable=trainable, act='lrelu', norm='instance', name='conv4')  # 32
        Dxz = tf.nn.sigmoid(Dxz)
    return Dxz

def create_pix2pix_trainer(base_lr=1e-4, networktype='pix2pix'):
    Cout = 3
    lambda_weight = 100
    
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    inSource = tf.placeholder(tf.float32, [None, 256, 256, Cout])
    inTarget = tf.placeholder(tf.float32, [None, 256, 256, Cout])

    GX = create_gan_G(inSource, is_training, Cout=Cout, trainable=True, reuse=False, networktype=networktype + '_G') 

    DGX = create_gan_D(GX, inTarget, is_training, trainable=True, reuse=False, networktype=networktype + '_D')
    DX = create_gan_D(inSource, inTarget, is_training, trainable=True, reuse=True, networktype=networktype + '_D')
    
    ganG_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(ganG_var_list), [var.name for var in ganG_var_list])

    ganD_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(ganD_var_list), [var.name for var in ganD_var_list])
              
    Gscore_L1 = tf.reduce_mean(tf.abs(inTarget - GX))
    Gscore = clipped_crossentropy(DGX, tf.ones_like(DGX)) + lambda_weight * Gscore_L1
    
    Dscore = clipped_crossentropy(DGX, tf.zeros_like(DGX)) + clipped_crossentropy(DX, tf.ones_like(DX))
    
    Gtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Gscore, var_list=ganG_var_list)
    Dtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.5).minimize(Dscore, var_list=ganD_var_list)
    
    return Gtrain, Dtrain, Gscore, Dscore, is_training, inSource, inTarget, GX

if __name__ == '__main__':
    direction = 'B2A'
    networktype = 'img2imgGAN_CMP_%s'%direction
    
    batch_size = 1
    base_lr = 0.0002  # 1e-4
    epochs = 200
        
    work_dir = expr_dir + '%s/%s/' % (networktype, datetime.strftime(datetime.today(), '%Y%m%d'))
    if not os.path.exists(work_dir): os.makedirs(work_dir)
    
    data, max_iter, test_iter, test_int, disp_int = get_train_params(data_dir, batch_size, epochs=epochs, test_in_each_epoch=1, networktype=networktype)
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()

    Gtrain, Dtrain, Gscore, Dscore, is_training, inSource, inTarget, GX = create_pix2pix_trainer(base_lr, networktype=networktype)
    tf.global_variables_initializer().run()
     
    var_list = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if (networktype.lower() in var.name.lower()) and ('adam' not in var.name.lower())]  
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100)
    # saver.restore(sess, expr_dir + 'ganMNIST/20170707/214_model.ckpt')  
     
    Xeval = np.load(data_dir + '%s/eval.npz' % networktype.replace('_A2B','').replace('_B2A',''))['data']    
    if direction == 'A2B': # from natural image to labels
            A_test = Xeval[:4, :, :, :3]
            B_test = Xeval[:4, :, :, 3:] 
    else: # from label to natural image            
            A_test = Xeval[:4, :, :, 3:]
            B_test = Xeval[:4, :, :, :3]
 
    taskImg = retransform(np.concatenate([A_test, B_test]))
    vis_square(taskImg, [4,2], save_path=work_dir + 'task.jpg')
       
    k = 1
      
    for it in range(1, max_iter): 
        X = data.train.next_batch(batch_size)
        if direction == 'A2B':# from natural image to labels
            A = X[:, :, :, :3]
            B = X[:, :, :, 3:] 
        else: # from label to natural image
            A = X[:, :, :, 3:]
            B = X[:, :, :, :3]
          
        for itD in range(k):
            cur_Dscore, _ = sess.run([Dscore, Dtrain], feed_dict={inSource: A, inTarget: B, is_training:True})
            
        cur_Gscore, _ = sess.run([Gscore, Gtrain], feed_dict={inSource: A, inTarget: B, is_training:True})

        if it % disp_int == 0:
            GX_sample = sess.run(GX, feed_dict={inSource:A_test, is_training:True})
            
            testImg = retransform(np.concatenate([A_test, GX_sample, B_test]))

            vis_square(testImg, [3,4], save_path=work_dir + 'Iter_%d.jpg' % it)
            saver.save(sess, work_dir + "%.3d_model.ckpt" % it)
            if ('cur_Dscore' in vars()) and ('cur_Gscore' in vars()):
                print("Iteration #%4d, Train Gscore = %f, Dscore=%f" % (it, cur_Gscore, cur_Dscore))
