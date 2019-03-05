from layers import *

class Discriminator(object):
    def __init__(self,batch_norm=True,name='disc'):
        self.x_dim = [32,32,3]
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            w = tf.random_normal_initializer(stddev=0.02)
            c0_0 = tcl.conv2d(x,    64, [3, 3], [1, 1],weights_initializer=w,activation_fn=leaky_relu)
            c1_0 = tcl.conv2d(c0_0, 128, [4, 4], [2, 2],weights_initializer=w,activation_fn=leaky_relu)
            c1_1 = tcl.conv2d(c1_0, 128, [3, 3], [1, 1],weights_initializer=w,activation_fn=leaky_relu)
            c2_0 = tcl.conv2d(c1_1, 256, [4, 4], [2, 2],weights_initializer=w,activation_fn=leaky_relu)
            c2_1 = tcl.conv2d(c2_0, 256, [3, 3], [1, 1],weights_initializer=w,activation_fn=leaky_relu)
            c3_0 = tcl.conv2d(c2_1, 512, [4, 4], [2, 2],weights_initializer=w,activation_fn=leaky_relu)
            c3_1 = tcl.conv2d(c3_0, 512, [3, 3], [1, 1],weights_initializer=w,activation_fn=leaky_relu)
            fc = tcl.fully_connected(tcl.flatten(c3_1), 1,weights_initializer=w,activation_fn=tf.identity)
        return  fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]       

class Generator(object):
    def __init__(self,batch_norm=True,name='g_woa'):
        self.z_dim = 128
        self.x_dim = [32,32,3]
        self.activity = relu_batch_norm if batch_norm else tf.nn.relu
        self.name = name

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            w = tf.random_normal_initializer(stddev=0.02)
            bs = tf.shape(z)[0]
            fc = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 512]))
            conv1 = self.activity(conv1)
            conv2 = tcl.conv2d_transpose(conv1, 256, [4, 4], [2, 2], weights_initializer=w, activation_fn=self.activity)
            conv3 = tcl.conv2d_transpose(conv2, 128, [4, 4], [2, 2], weights_initializer=w, activation_fn=self.activity)
            conv4 = tcl.conv2d_transpose(conv3, 64 , [4, 4], [2, 2], weights_initializer=w, activation_fn=self.activity)
            conv5 = tcl.conv2d_transpose(conv4, 3  , [3, 3], [1, 1], weights_initializer=w, activation_fn=tf.tanh)
            return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Generator_EMA(object):
    def __init__(self,batch_norm=True,name='g_ema'):
        self.z_dim = 128
        self.x_dim = [32,32,3]
        self.activity = relu_batch_norm if batch_norm else tf.nn.relu
        self.name = name

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            w = tf.random_normal_initializer(stddev=0.02)
            bs = tf.shape(z)[0]
            fc = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 512]))
            conv1 = self.activity(conv1)
            conv2 = tcl.conv2d_transpose(conv1, 256, [4, 4], [2, 2], weights_initializer=w, activation_fn=self.activity)
            conv3 = tcl.conv2d_transpose(conv2, 128, [4, 4], [2, 2], weights_initializer=w, activation_fn=self.activity)
            conv4 = tcl.conv2d_transpose(conv3, 64 , [4, 4], [2, 2], weights_initializer=w, activation_fn=self.activity)
            conv5 = tcl.conv2d_transpose(conv4, 3  , [3, 3], [1, 1], weights_initializer=w, activation_fn=tf.tanh)
            return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]