import tensorflow as tf
import tensorflow.contrib.layers as tcl


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)

def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x,decay=0.9))
    
def leaky_relu_layer_norm(x, alpha=0.2):
    return leaky_relu(tcl.layer_norm(x), alpha)

def relu_layer_norm(x):
    return tf.nn.relu(tcl.layer_norm(x))
