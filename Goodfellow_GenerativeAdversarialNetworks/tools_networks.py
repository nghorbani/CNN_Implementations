from tools_general import np, tf

def deconv(X, is_training, kernel_w, Cout, name, stride=1, act='ReLu', useBN=False, trainable=True, summary=1):
    '''use collectable prefix for accessing weights and biases throughout the code'''
    in_shape = X.get_shape().as_list()
    
    W = weight_variable([kernel_w, kernel_w, Cout, in_shape[3]], trainable=trainable, name='%s_W' % name)  # HWCinCout
    b = bias_variable([Cout, ], trainable=trainable, name='%s_b' % name)
    
    out_w = tf.cast(((in_shape[1] - 1) * stride) + kernel_w, tf.int32)
    output_shape = [in_shape[0], out_w, out_w, Cout]
    with tf.device('/gpu:0'):
        Y = tf.nn.bias_add(tf.nn.conv2d_transpose(X, W, strides=[1, stride, stride, 1], output_shape=output_shape, padding='VALID'), b)
        if useBN == True:
            Y = batch_norm(Y, is_training, trainable,name='%s_BN' % name)
        if act != None:
            if act.lower() == 'relu':
                Y = tf.nn.relu(Y)
            elif act.lower() == 'lrelu':
                Y = lrelu(Y, leak=0.2)
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

def conv(X, is_training, kernel_w, Cout, name, stride=1, act='ReLu', useBN=False, trainable=True, summary=1):
    '''use collectable prefix for accessing weights and biases throughout the code'''
    in_shape = X.get_shape().as_list()
    W = weight_variable([kernel_w, kernel_w, in_shape[3], Cout], trainable=trainable, name='%s_W' % name)  # HWCinCout
    b = bias_variable([Cout, ], trainable=trainable, name='%s_b' % name)
    with tf.device('/gpu:0'):
        Y = tf.nn.bias_add(tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID'), b)
        if useBN == True:
            Y = batch_norm(Y, is_training,trainable, name='%s_BN' % name)
        if act != None:
            if act.lower() == 'relu':
                Y = tf.nn.relu(Y)
            elif act.lower() == 'lrelu':
                Y = lrelu(Y, leak=0.2)
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

def dense(X, is_training, Dout, name, trainable=True, act='ReLu', useBN=False, summary=1):
    shapeIn = X.get_shape().as_list()
    
    with tf.device('/gpu:0'):
        W = tf.get_variable(name='%s_W' % name, shape=[shapeIn[1], Dout], trainable=trainable, initializer=tf.random_normal_initializer(stddev=0.02))
        b = bias_variable([Dout, ], name='%s_b' % name, trainable=trainable)
        
        Y = tf.nn.bias_add(tf.matmul(X, W), b)
        
        if useBN == True:
            Y = batch_norm(Y, is_training, trainable,name='%s_BN' % name)
   
        if act != None:
            if act.lower() == 'relu':
                Y = tf.nn.relu(Y)
            elif act.lower() == 'lrelu':
                Y = lrelu(Y, leak=0.2)
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

def batch_norm(X, is_training,trainable, name='BN', eps=1e-8):
#     Y = tf.contrib.layers.batch_norm(inputs=X,
#                                    decay=.999,
#                                    center=True,
#                                    scale=True,
#                                    updates_collections=None,
#                                    is_training=is_training,
#                                    scope=name,
#                                    fused=False,
#                                    trainable=trainable)

    if X.get_shape().ndims == 4:
        batch_mean, batch_var = tf.nn.moments(X, [0, 1, 2])
        Y = (X-batch_mean) / tf.sqrt(batch_var+eps)
 
    elif X.get_shape().ndims == 2:
        batch_mean, batch_var = tf.nn.moments(X, [0])
        Y = (X-batch_mean) / tf.sqrt(batch_var+eps)

    return Y

def xavier(shape, dtype, partition_info=None):
    fan_in = np.prod(shape[:3])
    fan_out = np.prod([shape[0], shape[1], shape[3]])
    w_bound = np.sqrt(6.0 / (fan_in + fan_out))
    initial_value = np.random.uniform(size=shape, low=-w_bound, high=w_bound)
    return initial_value

def lrelu(X, leak=0.2):
    Y = tf.maximum(X, leak * X)
    return Y

def clipped_crossentropy(X, L):
    Y = tf.clip_by_value(X, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y, labels=L))

def bias_variable(shape, name=None, trainable=True):
    """return an initialized bias variable of a given shape"""
    with tf.device('/gpu:0'):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    
def weight_variable(shape, name=None, trainable=True):
    """return an initialized weight variable of a given shape
    shape: HxWxCinxCout
    """
    with tf.device('/gpu:0'):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=xavier)
    
