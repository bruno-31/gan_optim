import numpy as np
import tensorflow.contrib as tc


class DataSampler(object):
    def __init__(self,dim):
        self.shape = [32,32,3]
        self.name = "cifar10"
        self.train, self.test = tc.keras.datasets.cifar10.load_data()
        self.train_x, self.train_y = self.train
        self.train_x = self.train_x.astype('float32')
        self.train_x /= 255.0
        self.train_x -= 0.5
        self.cur_batch = 0
        self.train_size = len(self.train_x)

    def __call__(self, batch_size):
        if not (self.cur_batch+1)*batch_size <= self.train_size:
            self.cur_batch = 0
        x = self.train_x[self.cur_batch*batch_size:(self.cur_batch+1)*batch_size,:,:,:]
        self.cur_batch +=1
        return x

    def data2img(self, data):
        rescaled = np.multiply(data + 0.5, 255.0)
        return np.reshape(np.clip(rescaled, 0.0, 255.0), [data.shape[0]] + self.shape)


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, z_dim])