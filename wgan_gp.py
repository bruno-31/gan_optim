import os
import time
import argparse
import importlib
import tensorflow as tf
from scipy.misc import imsave
import fid

from visualize import *

class GAN(object):
    def __init__(self, d_net, g_net, x_sampler, z_sampler, args,inception, log_dir, scale=10.0):
        self.model = args.model
        self.data = args.data
        self.log_dir = log_dir
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = d_net.x_dim
        self.z_dim = g_net.z_dim
        self.beta = 0.9999
        self.d_iters = 1
        self.batch_size=64
        self.inception=inception
        self.inception_path = fid.check_or_download_inception('./data/imagenet_model')
        
        if self.data == 'cifar10':
            self.stats_path = './data/fid_stats_cifar10_train.npz'
        elif self.data == 'stl10':
            self.stats_path = './data/fid_stats_stl10.npz'

        self.x = tf.placeholder(tf.float32, [None] + self.x_dim, name='x')
        self.z = tf.placeholder(tf.float32, [None] + [self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x)
        self.d_ = self.d_net(self.x_, reuse=True)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.d_net(x_hat, reuse=True)

        ddx = tf.gradients(d_hat, x_hat)[0]
        print(ddx.get_shape().as_list())
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        self.gp_loss = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss_reg = self.d_loss + self.gp_loss

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.0, beta2=0.9)\
                .minimize(self.d_loss_reg, var_list=self.d_net.vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.0, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

            
        for var_ in tf.model_variables('g_woa'):
            print(var_.name)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        with self.sess:
            fid.create_inception_graph(self.inception_path)  # load the graph into the current TF graph


    def train(self, num_batches=200000):
        self.sess.run(tf.global_variables_initializer())
        mean_list = []
        fid_list = []
        start_time = time.time()
        for t in range(0, num_batches):
            for _ in range(0, self.d_iters):
                bx = self.x_sampler(self.batch_size)
                bz = self.z_sampler(self.batch_size, self.z_dim)
                self.sess.run(self.d_adam, feed_dict={self.x: bx, self.z: bz})

            bx = self.x_sampler(self.batch_size)
            bz = self.z_sampler(self.batch_size, self.z_dim)
            self.sess.run([self.g_adam], feed_dict={self.z: bz, self.x: bx})

            if t % 1000 == 0:
                bx = self.x_sampler(self.batch_size)
                bz = self.z_sampler(self.batch_size,self.z_dim)
                dl,gl,gp,x_ = self.sess.run([self.d_loss,self.g_loss,self.gp_loss, self.x_],
                                                  feed_dict={self.x: bx,self.z:bz})

                print('Iter [%8d] Time [%.4f] dl [%.4f] gl [%.4f] gp [%.4f]'% (t, time.time() - start_time,dl,gl,gp))
                
                
                x_ = self.x_sampler.data2img(x_)
                x_ = grid_transform(x_, self.x_sampler.shape)
                imsave(self.log_dir+'/wos/{}.png'.format(int(t)), x_)


            if t%10000 == 0 and t>0:
                in_list = []
                for _ in range(int(50000/self.batch_size)):
                    bz = self.z_sampler(self.batch_size,self.z_dim)
                    x_ = self.sess.run(self.x_,feed_dict={self.z:bz})
                    x_ = self.x_sampler.data2img(x_)
                    bx_list = np.split(x_,self.batch_size)
                    in_list = in_list + [np.squeeze(x) for x in bx_list]
                mean, std = self.inception.get_inception_score(in_list,splits=10)
                mean_list.append(mean)
                np.save(self.log_dir+'/inception_score_wgan_gp.npy',np.asarray(mean_list))
                print('inception score [%.4f]'%(mean))


            if t %10000 == 0 and t>0 and args.fid:
                f = np.load(self.stats_path)
                mu_real, sigma_real = f['mu'][:], f['sigma'][:]
                f.close()

                mu_gen, sigma_gen = fid.calculate_activation_statistics(np.array(in_list[:10000]), self.sess, batch_size=100)
                fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                print("FID: %s" % fid_value)
                fid_list.append(fid_value)
                np.save(self.log_dir+'/fid_score_wgan_gp.npy',np.asarray(fid_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--model', type=str, default='wdcgan')
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--log', type=str, default='./logs')
    parser.add_argument('--fid', type=bool, default=True)


    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module('cifar10'+ '.' + args.model)

    inception = importlib.import_module('inception_score')
    log_dir = args.log + '/' + args.data

    xs = data.DataSampler(args.size)
    print('wgan')
    print('data size:', xs.train_size)
    zs = data.NoiseSampler()

    d_net = model.Discriminator(batch_norm=False)
    g_net = model.Generator()

    os.makedirs(os.path.abspath(log_dir + '/wos/'.format(args.data)))
    gan = GAN(d_net,g_net, xs, zs, args,inception,log_dir)
    gan.train()
