from tools_general import np, tf
from tools_train import get_train_params
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
        Gz = dense(z, is_training, Dout=4 * 4 * 256, act='reLu', useBN=True, name='dense2')
        Gz = tf.reshape(Gz, shape=[-1, 4, 4, 256])  # 4
        Gz = deconv(Gz, is_training, kernel_w=5, stride=2, Cout=256, trainable=trainable, act='reLu', useBN=True, name='deconv1')  # 11
        Gz = deconv(Gz, is_training, kernel_w=5, stride=2, Cout=128, trainable=trainable, act='reLu', useBN=True, name='deconv2')  # 25
        Gz = deconv(Gz, is_training, kernel_w=4, stride=1, Cout=1, act=None, useBN=False, name='deconv3')  # 28
        Gz = tf.nn.sigmoid(Gz)
    return Gz

def create_gan_D(xz, labels, is_training, trainable=True, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        xz = concat_labels(xz, labels)
        Dxz = conv(xz, is_training, kernel_w=5, stride=2, Cout=128, trainable=trainable, act='lrelu', useBN=False, name='conv1')  # 12
        Dxz = conv(Dxz, is_training, kernel_w=5, stride=2, Cout=256, trainable=trainable, act='lrelu', useBN=True, name='conv2')  # 4
        Dxz = conv(Dxz, is_training, kernel_w=2, stride=2, Cout=256, trainable=trainable, act='lrelu', useBN=True, name='conv3')  # 2
        Dxz = conv(Dxz, is_training, kernel_w=2, stride=2, Cout=1, trainable=trainable, act='lrelu', useBN=True, name='conv4')  # 2
        Dxz = tf.nn.sigmoid(Dxz)
    return Dxz

def create_gan_trainer(base_lr=1e-4, networktype='gan'):
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
    from tools_config import *
    
    import matplotlib.pyplot as plt
    import scipy.misc
    from tools_train import OneHot
   
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    is_training = tf.placeholder(tf.bool, [], 'is_training')
    
    batchsize = 15
    
    z = tf.constant(0.0, shape=[batchsize, 100])
    xz = tf.constant(0.0, shape=[batchsize, 28, 28, 1])
    labels = OneHot(np.random.randint(10, size=[batchsize]), n=10)
    # labels = OneHot(tf.random_uniform(minval = 0, maxval = 10, shape=[batchsize], dtype = tf.int32), n=10)        
    
    Gz = create_gan_G(z, labels, is_training, Cout=1, trainable=True, reuse=False, networktype='ganG')
    Dxz = create_gan_D(xz, labels, is_training, trainable=True, reuse=False, networktype='ganD')
    
    tf.global_variables_initializer().run()

    # out = sess.run(Gz, feed_dict = {is_training: False})
    out = sess.run(Gz, feed_dict={is_training: False})

    print(out.shape)
