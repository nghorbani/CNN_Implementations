from tools_general import np, tf
from utils import get_train_params
from tools_tensorflow import deconv, conv, dense, clipped_crossentropy
     
def create_gan_G(z, labels, is_training, Cout=1, trainable=True, reuse=False, networktype='ganG'):
    '''input : batchsize * 100
        output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        batch_size = labels.get_shape().as_list()[0]
        
        labels_reshaped = tf.reshape(labels, [batch_size, 1, 1, 10])
        z = tf.concat(axis=-1, values=[z, labels_reshaped * tf.ones([batch_size, 8, 8, 10])])
        Gz = conv(z, is_training, kernel=2, stride=2, Cout=1024, trainable=trainable, act='ReLu', useBN=True, name='conv1')  # 12
        Gz = tf.concat(axis=-1, values=[Gz, labels_reshaped * tf.ones([batch_size, 4, 4, 10])])
        Gz = deconv(Gz, is_training, kernel=5, stride=2, Cout=512, trainable=trainable, act='ReLu', useBN=True, name='deconv1')  # 11
        Gz = tf.concat(axis=-1, values=[Gz, labels_reshaped * tf.ones([batch_size, 11, 11, 10])])
        Gz = deconv(Gz, is_training, kernel=5, stride=2, Cout=256, trainable=trainable, act='ReLu', useBN=True, name='deconv2')  # 25
        Gz = tf.concat(axis=-1, values=[Gz, labels_reshaped * tf.ones([batch_size, 25, 25, 10])])
        Gz = deconv(Gz, is_training, kernel=4, stride=1, Cout=1, act=None, useBN=False, name='deconv3')  # 28
        Gz = tf.nn.sigmoid(Gz)
    return Gz

def create_gan_D(xz, labels, is_training, trainable=True, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        batch_size = labels.get_shape().as_list()[0]
        
        labels_reshaped = tf.reshape(labels, [batch_size, 1, 1, 10])
        xz = tf.concat(axis=-1, values=[xz, labels_reshaped * tf.ones([batch_size, 28, 28, 10])])
        Dxz = conv(xz, is_training, kernel=5, stride=2, Cout=512, trainable=trainable, act='lrelu', useBN=False, name='conv1')  # 12
        Dxz = tf.concat(axis=-1, values=[Dxz, labels_reshaped * tf.ones([batch_size, 12, 12, 10])])
        Dxz = conv(Dxz, is_training, kernel=5, stride=2, Cout=1024, trainable=trainable, act='lrelu', useBN=True, name='conv2')  # 4
        Dxz = tf.concat(axis=-1, values=[Dxz, labels_reshaped * tf.ones([batch_size, 4, 4, 10])])
        Dxz = conv(Dxz, is_training, kernel=4, stride=4, Cout=1, trainable=trainable, act='lrelu', useBN=True, name='conv3')  # 2
        #Dxz = conv(Dxz, is_training, kernel=2, stride=2, Cout=1, trainable=trainable, act='lrelu', useBN=True, name='conv4')  # 1
        Dxz = tf.nn.sigmoid(Dxz)
    return Dxz

def create_gan_trainer(base_lr=1e-4, batch_size=128, networktype='gan'):
    '''Train Generative Adversarial Network'''
    # with tf.name_scope('train_%s' % networktype): 
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    inZ = tf.placeholder(tf.float32, [batch_size, 8, 8, 1])  # tf.random_uniform(shape=[batch_size, 100], minval=-1., maxval=1., dtype=tf.float32)
    inL = tf.placeholder(tf.float32, [batch_size, 10])  # we want to condition the generated out put on some parameters of the input
    inX = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])

    Gz = create_gan_G(inZ, inL, is_training, Cout=1, trainable=True, reuse=False, networktype=networktype + '_G') 

    DGz = create_gan_D(Gz, inL, is_training, trainable=True, reuse=False, networktype=networktype + '_D')
    Dx = create_gan_D(inX, inL, is_training, trainable=True, reuse=True, networktype=networktype + '_D')
    
    ganG_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(ganG_var_list), [var.name for var in ganG_var_list])

    ganD_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(ganD_var_list), [var.name for var in ganD_var_list])
          
    Gscore = clipped_crossentropy(DGz, tf.ones_like(DGz))
    Dscore = clipped_crossentropy(DGz, tf.zeros_like(DGz)) + clipped_crossentropy(Dx, tf.ones_like(Dx))
    
    Gtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.45).minimize(Gscore, var_list=ganG_var_list)
    Dtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.4 5).minimize(Dscore, var_list=ganD_var_list)
    
    return Gtrain, Dtrain, Gscore, Dscore, is_training, inZ, inX, inL, Gz
       
if __name__ == '__main__':
    from tools_config import *
    
    import matplotlib.pyplot as plt
    import scipy.misc
    from utils import OneHot
   
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
    
    is_training = tf.placeholder(tf.bool, [], 'is_training')
    
    batchsize = 15
    
    z = tf.constant(0.0, shape=[batchsize, 8, 8, 1])
    xz = tf.constant(0.0, shape=[batchsize, 28, 28, 1])
    labels = OneHot(np.random.randint(10, size=[batchsize]), n=10)
    # labels = OneHot(tf.random_uniform(minval = 0, maxval = 10, shape=[batchsize], dtype = tf.int32), n=10)        
    
    Gz = create_gan_G(z, labels, is_training, Cout=1, trainable=True, reuse=False, networktype='ganG')
    Dxz = create_gan_D(xz, labels, is_training, trainable=True, reuse=False, networktype='ganD')
    
    tf.global_variables_initializer().run()

    # out = sess.run(Gz, feed_dict = {is_training: False})
    out = sess.run(Dxz, feed_dict={is_training: False})

    print(out.shape)
