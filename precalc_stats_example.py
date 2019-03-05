
import numpy as np
import fid
import tensorflow as tf

########
# PATHS
########
data_path = '/home/blecouat//data/stl10/' # set path to training set images
output_path = './data/fid_stats_stl10.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = './data/imagenet_model'
print("check for inception model..")
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..")
# image_list = glob.glob(os.path.join(data_path, '*.jpg'))
# images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
train, _ = tf.keras.datasets.cifar10.load_data()
images, _ = train

print("%d images found and loaded" % len(images))

print("create inception graph..")
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")