from tools_general import np, tf
from utils import get_train_params

def xavier(shape, dtype, partition_info=None):
    fan_in = np.prod(shape[:3])
    fan_out = np.prod([shape[0], shape[1], shape[3]])
    w_bound = np.sqrt(6.0 / (fan_in + fan_out))
    initial_value = np.random.uniform(size=shape, low=-w_bound, high=w_bound)
    return initial_value

def weight_variable(shape, name=None, trainable=True):
    """return an initialized weight variable of a given shape
    shape: HxWxCinxCout
    """
    with tf.device('/gpu:0'):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=xavier)

def bias_variable(shape, name=None, trainable=True):
    """return an initialized bias variable of a given shape"""
    with tf.device('/gpu:0'):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    
def deconv(X, is_training, kernel, Cout, name, stride=1, act='ReLu', useBN=False, trainable=True, summary=1):
    '''use collectable prefix for accessing weights and biases throughout the code'''
    in_shape = X.get_shape().as_list()
    
    W = weight_variable([kernel, kernel, Cout, in_shape[3]], trainable=trainable, name='%s_W' % (name))  # HWCinCout
    b = bias_variable([Cout, ], trainable=trainable, name='%s_b' % (name))
    
    out_w = tf.cast(((tf.shape(X)[1] - 1) * stride) + kernel, tf.int32)
    output_shape = [tf.shape(X)[0], out_w, out_w, Cout]
    
    Y = tf.nn.bias_add(tf.nn.conv2d_transpose(X, W, strides=[1, stride, stride, 1], output_shape=output_shape, padding='VALID'), b)
    
    if useBN == True:
        Y = batch_norm(Y, is_training=is_training, trainable=trainable, name='%s_BN' % name)
    
    if act != None:
        if act.lower() == 'relu':
            Y = tf.nn.relu(Y)
        elif act.lower() == 'lrelu':
            Y = tf.maximum(Y, 0.2 * Y)
        elif act.lower() == 'tanh':
            Y = tf.nn.tanh(Y)
        elif act.lower() == 'sigmoid':
            Y = tf.nn.sigmoid(Y)
        else:
            print('Unknown activation function')     
        
    if summary:
        tf.summary.histogram('%s_W_histogram' % (name), W)
        tf.summary.histogram('%s_bias_histogram' % (name), b)
    return Y

def conv(X, is_training, kernel, Cout, name, stride=1, act='ReLu', useBN=False, trainable=True, summary=1):
    '''use collectable prefix for accessing weights and biases throughout the code'''
    in_shape = X.get_shape().as_list()
    W = weight_variable([kernel, kernel, in_shape[3], Cout], trainable=trainable, name='%s_W' % name)  # HWCinCout
    b = bias_variable([Cout, ], trainable=trainable, name='%s_b' % name)
    Y = tf.nn.bias_add(tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID'), b)
    if useBN == True:
        Y = batch_norm(Y, is_training=is_training, trainable=trainable, name='%s_BN' % name)
    if act != None:
        if act.lower() == 'relu':
            Y = tf.nn.relu(Y)
        elif act.lower() == 'lrelu':
            Y = tf.maximum(Y, 0.2 * Y)
        elif act.lower() == 'tanh':
            Y = tf.nn.tanh(Y)
        elif act.lower() == 'sigmoid':
            Y = tf.nn.sigmoid(Y)
        else:
            print('Unknown activation function')
            
    if summary:
        tf.summary.histogram('%s_W_histogram' % name, W)
        tf.summary.histogram('%s_bias_histogram' % name, b)
    return Y

def dense(X, Dout, name, trainable=True, summary=1):
    shapeIn = X.get_shape().as_list()
    
    with tf.device('/gpu:0'):
        W = tf.get_variable(name='%s_W' % name, shape=[shapeIn[1], Dout], trainable=trainable, initializer=tf.random_normal_initializer())
        b = bias_variable([Dout, ], name='%s_b' % name, trainable=trainable)
        
        out = tf.nn.bias_add(tf.matmul(X, W), b)
    
    if summary:
        tf.summary.histogram('%s_W_histogram' % (name), W)
        tf.summary.histogram('%s_bias_histogram' % (name), b)    
    return out  


def batch_norm(inputs, is_training, trainable=True, name='BN', decay=0.999, epsilon=1e-5, summary=1):
    '''
    correct way to use:
    Y = Activation(BN(conv(W,X)+bias))
    BN(Xbatch) = gamma((Xbatch-Meanbatch)/sqrt(Varbatch+epsilon)) + beta
    '''
   
    beta = tf.get_variable(name='%s_beta' % name, shape=[inputs.get_shape()[-1]], trainable=trainable, initializer=tf.constant_initializer(0.0))
    gamma = tf.get_variable(name='%s_gamma' % name, shape=[inputs.get_shape()[-1]], trainable=trainable, initializer=tf.constant_initializer(1.0))
       
    pop_mean = tf.get_variable(name='%s_pop_mean' % name, shape=[inputs.get_shape()[-1]], trainable=False, initializer=tf.constant_initializer(0.0))
    pop_var = tf.get_variable(name='%s_pop_var' % name, shape=[inputs.get_shape()[-1]], trainable=False, initializer=tf.constant_initializer(1.0))
     
    def BN_Train():
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='%s_moments' % name)
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=beta, scale=gamma, variance_epsilon=epsilon, name=name)
         
    def BN_Test():
        return tf.nn.batch_normalization(inputs, mean=pop_mean, variance=pop_var, offset=beta, scale=gamma, variance_epsilon=epsilon, name=name)
    if summary:
        tf.summary.histogram('Population mean', pop_mean)
        tf.summary.histogram('Population Var', pop_var)
    return tf.cond(is_training, lambda: BN_Train(), lambda: BN_Test())
    
def create_gan_G(z, is_training, Cout=1, trainable=True, reuse=False, networktype='ganG'):
    '''input : batchsize * 100
        output: batchsize * 28 * 28 * 1'''
    with tf.variable_scope(networktype, reuse=reuse):
        Gz = dense(z, Dout=4 * 4 * 1024, name='dense1')
        Gz = tf.reshape(Gz, shape=[-1, 4, 4, 1024])  # 4
        Gz = deconv(Gz, is_training=is_training, kernel=5, stride=2, Cout=512, act='relu', useBN=False, trainable=trainable, summary=1, name='deconv1')  # 11
        Gz = deconv(Gz, is_training=is_training, kernel=5, stride=2, Cout=256, act='relu', useBN=False, trainable=trainable, summary=1, name='deconv2')  # 25
        Gz = deconv(Gz, is_training=is_training, kernel=4, stride=1, Cout=1, act=None, useBN=False, trainable=trainable, summary=1, name='deconv3')  # 28
        Gz = tf.nn.sigmoid(Gz)
    return Gz

def create_gan_D(Gz, is_training, trainable=True, reuse=False, networktype='ganD'):
    with tf.variable_scope(networktype, reuse=reuse):
        DGz = conv(Gz, is_training=is_training, kernel=5, stride=2, Cout=512, act='lrelu', useBN=False, trainable=trainable, summary=1, name='conv1')  # 12
        DGz = conv(DGz, is_training=is_training, kernel=5, stride=2, Cout=1024, act='lrelu', useBN=False, trainable=trainable, summary=1, name='conv2')  # 4
        DGz = tf.reshape(DGz, shape=[-1, 4 * 4 * 1024])
        DGz = dense(DGz, Dout=1, name='dense1')
        DGz = tf.nn.sigmoid(DGz)
    return DGz

def create_gan_trainer(base_lr=1e-4, batch_size=128, networktype='gan'):
    '''Train Generative Adversarial Network'''
    # with tf.name_scope('train_%s' % networktype): 
    is_training = tf.placeholder(tf.bool, [], 'is_training')

    inZ = tf.random_uniform(shape=[batch_size, 100], minval=0., maxval=1., dtype=tf.float32)  # tf.placeholder(tf.float32, [None, 100])
    inX = tf.placeholder(tf.float32, [None, 28, 28, 1])

    Gz = create_gan_G(inZ, is_training, Cout=1, trainable=True, reuse=False, networktype=networktype + '_G') 

    DGz = create_gan_D(Gz, is_training, trainable=True, reuse=False, networktype=networktype + '_D')
    Dx = create_gan_D(inX, is_training, trainable=True, reuse=True, networktype=networktype + '_D')
    
    ganG_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_G')
    print(len(ganG_var_list), [var.name for var in ganG_var_list])

    ganD_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=networktype + '_D')
    print(len(ganD_var_list), [var.name for var in ganD_var_list])
    
    # DGz = tf.clip_by_value(DGz, 1e-7, 1. - 1e-7)
    # Dx = tf.clip_by_value(Dx, 1e-7, 1. - 1e-7) 
       
    #Gscore = tf.reduce_mean(tf.log(DGz))
    #Dscore = tf.reduce_mean(tf.log(Dx) + tf.log(1. - DGz))
    Gscore = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones(shape=[batch_size,])))
    
    Gtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(Gscore, var_list=ganG_var_list)
    Dtrain = tf.train.AdamOptimizer(learning_rate=base_lr, beta1=0.45, beta2=0.999, epsilon=1e-08).minimize(-Dscore, var_list=ganD_var_list)
    
    return Gtrain, Dtrain, Gscore, Dscore, is_training, inZ, inX, Gz

def vis_square(data, filename=None):
    
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
       source: https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb"""
           
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don'ad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure(figsize=(50, 50))
    plt.imshow(data, cmap='hot'); plt.axis('off')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show() 
        
if __name__ == '__main__':
    from tools_config import *
    
    import matplotlib.pyplot as plt

    networktype = 'ganMNIST'
    batch_size = 128
    base_lr = 0.0001  # 1e-4
    epochs = 100
     
    data, max_iter, test_iter, test_int, disp_int = get_train_params(batch_size, epochs=10, test_in_each_epoch=1, networktype=networktype)
    # disp_int = 10
    
    tf.reset_default_graph() 
    sess = tf.InteractiveSession()
     
    Gtrain, Dtrain, Gscore, Dscore, is_training, inZ, inX, Gz = create_gan_trainer(base_lr, batch_size, networktype=networktype)
    tf.global_variables_initializer().run()
    
    for it in range(max_iter): 
        # Z = np.random.uniform(size=[batch_size, 100], low=0., high=1.).astype(np.float32)
        X, _ = data.train.next_batch(batch_size)
        
        cur_Dscore, _ = sess.run([Dscore, Dtrain], feed_dict={inX: X, is_training:True})
        cur_Gscore, _ = sess.run([Gscore, Gtrain], feed_dict={is_training:True})
        
        if it % disp_int == 0:
            Gz_sample = sess.run(Gz, feed_dict={is_training:False})
            vis_square(Gz_sample[0:50].squeeze(), expr_dir + '%d.png' % it)
            print("Iteration #%4d, Train Gscore = %f, Dscore=%f" % (it, cur_Gscore, cur_Dscore))
            
