from tools_general import np, tf

def deconv(X, is_training, kernel_w, stride, Cout, epf=None, trainable=True, act='ReLu', norm=None, name='deconv'):
    ''' epf is the expansion factor. when used padding will be SAME
    '''
    in_shape = X.get_shape().as_list()
    in_shape2 = tf.shape(X)
        
    W = weight_variable([kernel_w, kernel_w, Cout, in_shape[3]], trainable=trainable, name='%s_W' % name)  # HWCinCout
    b = tf.zeros([Cout,], tf.float32)#it seems there is a bug in tensorflow and adding zero as bias fixes the problem
    with tf.device('/gpu:0'):
        if epf == None:
            out_w = tf.cast(((in_shape2[1] - 1) * stride) + kernel_w, tf.int32)
            output_shape = [in_shape2[0], out_w, out_w, Cout]
            Y = tf.nn.bias_add(tf.nn.conv2d_transpose(X, W, strides=[1, stride, stride, 1], output_shape=output_shape, padding='VALID'), b)
        else:
            Y = tf.nn.bias_add(tf.nn.conv2d_transpose(X, W, strides=[1, stride, stride, 1], output_shape=[in_shape2[0], in_shape2[1] * epf, in_shape2[2] * epf, Cout], padding='SAME'), b)
        if norm != None:
            if norm.lower() == 'batchnorm':
                Y = batch_norm(Y, is_training, trainable, name='%s_BN' % name)
            elif norm.lower() == 'instance':
                Y = instance_norm(Y, trainable, name='%s_IN' % name) 
            else:
                print('Unknown normalization procedure',print(norm.lower()))             
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
        return Y

def conv(X, is_training, kernel_w, stride, Cout, pad=None, trainable=True, act='ReLu', norm=None, name='conv'):
    with tf.device('/gpu:0'):
        X = tf.identity(X)
        in_shape = X.get_shape().as_list()
        
        W = weight_variable([kernel_w, kernel_w, in_shape[3], Cout], trainable=trainable, name='%s_W' % name)  # HWCinCout
    
        if pad != None: 
            X = tf.pad(X, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode="CONSTANT")
        Y = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID')
        if norm != None:
            if norm.lower() == 'batchnorm':
                Y = batch_norm(Y, is_training, trainable, name='%s_BN' % name)
            elif norm.lower() == 'instance':
                Y = instance_norm(Y, trainable, name='%s_IN' % name)
            else:
                print('Unknown normalization procedure')
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
                print('Unknown activation function',print(norm.lower()))
        return Y

def dense(X, is_training, Cout, trainable=True, act='ReLu', norm=None, name='dense'):
    '''output = batchsize * Cout'''
    with tf.device('/gpu:0'):
        if X.get_shape().ndims == 4:
            shapeIn = X.get_shape().as_list()
            X = tf.reshape(X,shape = [-1, shapeIn[1]*shapeIn[2]*shapeIn[3]])
            
        X = tf.identity(X)
        shapeIn = X.get_shape().as_list()
    
        W = tf.get_variable(name='%s_W' % name, shape=[shapeIn[1], Cout], trainable=trainable, initializer=tf.random_normal_initializer(stddev=0.02))

        Y = tf.matmul(X, W)
        if norm != None:
            if norm.lower() == 'batchnorm':
                Y = batch_norm(Y, is_training, trainable, name='%s_BN' % name)
            elif norm.lower() == 'instance':
                Y = instance_norm(Y, trainable, name='%s_IN' % name)
            else:
                print('Unknown normalization procedure',print(norm.lower()))
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
    return Y 
    
def batch_norm(inputs, is_training, trainable, name, decay=0.9, epsilon=1e-5):
    '''
    correct way to use:
    Y = Activation(BN(conv(W,X)+bias))
    BN(Xbatch) = gamma((Xbatch-Meanbatch)/sqrt(Varbatch+epsilon)) + beta
    '''
    with tf.device('/gpu:0'):
        beta = tf.get_variable(name='%s_beta' % name, shape=inputs.get_shape()[-1], trainable=trainable, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(name='%s_scale' % name, shape=inputs.get_shape()[-1], trainable=trainable, dtype=tf.float32, initializer=tf.constant_initializer(1.0))  # scale each channel
        #gamma = tf.get_variable(name='%s_scale' % name, shape=inputs.get_shape()[-1], trainable=trainable, dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))  # scale each channel
        
        pop_mean = tf.get_variable(name='%s_pop_mean' % name, shape=inputs.get_shape()[-1], trainable=False, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pop_var = tf.get_variable(name='%s_pop_var' % name, shape=inputs.get_shape()[-1], trainable=False, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
     
        def BN_Train():
            if inputs.get_shape().ndims == 4:
                batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
            elif inputs.get_shape().ndims == 2:
                batch_mean, batch_var = tf.nn.moments(inputs, [0], name='moments')
    
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
             
        def BN_Test():
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, epsilon)
        
        return tf.cond(is_training, lambda: BN_Train(), lambda: BN_Test())

def instance_norm(inputs, trainable, name, decay=0.9, epsilon=1e-5):
    with tf.device('/gpu:0'):
        beta = tf.get_variable(name='%s_beta' % name, shape=inputs.get_shape()[-1], trainable=trainable, dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable(name='%s_scale' % name, shape=inputs.get_shape()[-1], trainable=trainable, dtype=tf.float32, initializer=tf.constant_initializer(1.0))  # scale each channel
        
        if inputs.get_shape().ndims == 4:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            #batch_mean, batch_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
        elif inputs.get_shape().ndims == 2:
            batch_mean, batch_var = tf.nn.moments(inputs, [])

        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)

def dropout(Xin, is_training, p=0.8):
    with tf.device('/gpu:0'):
        Xout = tf.cond(is_training, lambda: tf.nn.dropout(Xin, keep_prob=p), lambda: tf.identity(Xin))
        return Xout

def lrelu(X, leak=0.2):
    with tf.device('/gpu:0'):
        Y = tf.maximum(X, leak * X)
        return Y

def clipped_crossentropy(X, L):
    with tf.device('/gpu:0'):
        Y = tf.clip_by_value(X, 1e-7, 1. - 1e-7)
        return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y, labels=L), [1,2,3]))

def bias_variable(shape, name=None, trainable=True):
    """return an initialized bias variable of a given shape"""
    with tf.device('/gpu:0'):
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, trainable=trainable, initializer=tf.constant_initializer(0.0))
    
def weight_variable(shape, name=None, trainable=True):
    """return an initialized weight variable of a given shape
    shape: HxWxCinxCout
    """
    with tf.device('/gpu:0'):
        #return tf.get_variable(name=name, shape=shape, dtype=tf.float32, trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
        return tf.get_variable(name=name, shape=shape, dtype=tf.float32, trainable=trainable, initializer=tf.truncated_normal_initializer(stddev=0.02))
def regularization(variables, regtype='L1', regcoef=0.1):
    regs = tf.constant(0.0)
    for var in variables:
        if regtype.upper() == 'L2':
                regs = tf.add(regs, tf.nn.l2_loss(var))
        elif regtype.upper() == 'L1':
            regs = tf.add(regs, tf.reduce_mean(tf.abs(var)))
        else:
            raise('regularization type not detected!')
    print("Regularizing with type %s, coef %s for %d variables!" % (regtype, regcoef, len(variables)))
    return tf.multiply(regcoef, regs)
