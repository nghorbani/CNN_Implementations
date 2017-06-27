from tools_config import *

import matplotlib.pyplot as plt
from networks import create_gan_trainer
from utils import get_train_params, OneHot, vis_square

from tools_general import tf, np

networktype = 'ganMNIST'
batch_size = 128
base_lr = 0.0003  # 1e-4
epochs = 100
 
data, max_iter, test_iter, test_int, disp_int = get_train_params(batch_size, epochs=epochs, test_in_each_epoch=1, networktype=networktype)

tf.reset_default_graph() 
sess = tf.InteractiveSession()

Gtrain, Dtrain, Gscore, Dscore, is_training, inZ, inX, inL, Gz = create_gan_trainer(base_lr, batch_size, networktype=networktype)
tf.global_variables_initializer().run()
 
Z_test = np.random.uniform(size=[batch_size, 100], low=-1., high=1.).astype(np.float32)
labels_test = OneHot(np.random.randint(10, size=[batch_size]), n=10)    

k = 2
 
for it in range(max_iter): 
    Z = np.random.uniform(size=[batch_size, 100], low=-1., high=1.).astype(np.float32)
    X, labels = data.train.next_batch(batch_size)
     
    if it%k == 0:
        cur_Gscore, _ = sess.run([Gscore, Gtrain], feed_dict={inZ:Z, inL: labels,  is_training:True})
    else:
        cur_Dscore, _ = sess.run([Dscore, Dtrain], feed_dict={inX: X, inZ:Z,inL: labels, is_training:True})
 
    if it % disp_int == 0:
        Gz_sample = sess.run(Gz, feed_dict={inZ: Z_test, inL: labels_test, is_training:False})
        vis_square(Gz_sample[:121], [11,11], save_path=expr_dir+'Iter_%d.jpg'%it)
        if it != 0: print("Iteration #%4d, Train Gscore = %f, Dscore=%f" % (it, cur_Gscore, cur_Dscore))