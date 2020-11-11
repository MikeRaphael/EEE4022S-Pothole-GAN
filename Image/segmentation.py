# --------------------------
# Source and reference
#   https://github.com/udacity/CarND-Semantic-Segmentation
#   https://gist.github.com/lianyi/a5ba8d84f5b68401c2313b05e020b062
#   https://medium.com/nanonets/how-to-do-image-segmentation-using-deep-learning-c673cc5862ef
# --------------------------


# --------------------------
# DATA PREPARATION
# Download the data http://www.cvlibs.net/datasets/kitti/eval_road.php from http://www.cvlibs.net/download.php?file=data_road.zip
# extract the data to ./data directory
# --------------------------
#

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import warnings
import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
# --------------------------
# USER-SPECIFIED DATA
# --------------------------

# Tune these parameters

NUMBER_OF_CLASSES = 2
IMAGE_SHAPE = (160, 576)
EPOCHS = 40
BATCH_SIZE = 16
DROPOUT = 0.75

# Specify these directory paths

data_dir = './data'
runs_dir = './runs'
training_dir = './data/data_road/training'
vgg_path = './data/vgg'

# --------------------------
# Check for a GPU
# --------------------------

#
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# --------------------------
# PLACEHOLDER TENSORS
# --------------------------

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

class DLProgress(tqdm):
	"""
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)  # Updates progress
		self.last_block = block_num



# --------------------------
# FUNCTIONS
# --------------------------
def maybe_download_pretrained_vgg(data_dir):
	"""
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
	vgg_filename = 'vgg.zip'
	vgg_path = os.path.join(data_dir, 'vgg')
	vgg_files = [
		os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
		os.path.join(vgg_path, 'variables/variables.index'),
		os.path.join(vgg_path, 'saved_model.pb')]

	missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
	if missing_vgg_files:
		# Clean vgg dir
		if os.path.exists(vgg_path):
			shutil.rmtree(vgg_path)
		os.makedirs(vgg_path)

		# Download vgg
		print('Downloading pre-trained vgg model...')
		with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
			urlretrieve(
				'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
				os.path.join(vgg_path, vgg_filename),
				pbar.hook)

		# Extract vgg
		print('Extracting model...')
		zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
		zip_ref.extractall(data_dir)
		zip_ref.close()

		# Remove zip file to save space
		os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
	"""
	Generate function to create batches of training data
	:param data_folder: Path to folder that contains all the datasets
	:param image_shape: Tuple - Shape of image
	:return:
	"""
	def get_batches_fn(batch_size):
		"""
		Create batches of training data
		:param batch_size: Batch Size
		:return: Batches of training data
		"""
		# Grab image and label paths
		image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
		label_paths = {
			re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
			for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
		background_color = np.array([255, 0, 0])

		# Shuffle training data
		random.shuffle(image_paths)
		# Loop through batches and grab images, yielding each batch
		for batch_i in range(0, len(image_paths), batch_size):
			images = []
			gt_images = []
			for image_file in image_paths[batch_i:batch_i+batch_size]:
				gt_image_file = label_paths[os.path.basename(image_file)]
				# Re-size to image_shape
				# image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
				image = np.array(image_file.fromarray(image_shape).resize())
				# gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
				gt_image = np.array(gt_image_file.fromarray(image_shape).resize())

				# Create "one-hot-like" labels by class
				gt_bg = np.all(gt_image == background_color, axis=2)
				gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
				gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

				images.append(image)
				gt_images.append(gt_image)

			yield np.array(images), np.array(gt_images)
	return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
		image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_pl: [image]})
		# Splice out second column (road), reshape output back to image_shape
		im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
		# If road softmax > 0.5, prediction is road
		segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
		# Create mask based on segmentation to apply to original image
		mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
		mask = scipy.misc.toimage(mask, mode="RGBA")
		street_im = scipy.misc.toimage(image)
		street_im.paste(mask, box=None, mask=mask)

		yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(
		sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
	for name, image in image_outputs:
		scipy.misc.imsave(os.path.join(output_dir, name), image)

def load_vgg(sess, vgg_path):
    # load the model and weights
    model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    # Get Tensors to be returned from graph
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')

    return image_input, keep_prob, layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUMBER_OF_CLASSES):
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer
    fcn8 = tf.layers.conv2d(layer7, filters=NUMBER_OF_CLASSES, kernel_size=1, name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
                                      kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
                                       kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=NUMBER_OF_CLASSES,
                                       kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11


def optimize(nn_last_layer, correct_label, learning_rate, NUMBER_OF_CLASSES):
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, NUMBER_OF_CLASSES), name="fcn_logits")
    correct_label_reshaped = tf.reshape(correct_label, (-1, NUMBER_OF_CLASSES))

    # Calculate distance from actual labels using cross entropy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
    # Take mean for total loss
    loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

    return logits, train_op, loss_op


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    keep_prob_value = 0.5
    learning_rate_value = 0.001
    for epoch in range(epochs):
        # Create function to get batches
        total_loss = 0
        for X_batch, gt_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op],
                               feed_dict={input_image: X_batch, correct_label: gt_batch,
                                          keep_prob: keep_prob_value, learning_rate: learning_rate_value})

            total_loss += loss

        print("EPOCH {} ...".format(epoch + 1))
        print("Loss = {:.3f}".format(total_loss))
        print()


def run():
    # Download pretrained vgg model
    maybe_download_pretrained_vgg(data_dir)

    # A function to get batches
    get_batches_fn = gen_batch_function(training_dir, IMAGE_SHAPE)

    with tf.Session() as session:
        # Returns the three layers, keep probability and input layer from the vgg architecture
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

        # The resulting network architecture from adding a decoder on top of the given vgg model
        model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

        # Returns the output logits, training operation and cost operation to be used
        # - logits: each row represents a pixel, each column a class
        # - train_op: function used to get the right parameters to the model to correctly label the pixels
        # - cross_entropy_loss: function outputting the cost which we are minimizing, lower cost should yield higher accuracy
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)

        # Initialize all variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model build successful, starting training")

        # Train the neural network
        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn,
                 train_op, cross_entropy_loss, image_input,
                 correct_label, keep_prob, learning_rate)

        # Run the model with the test images and save each painted output image (roads painted green)
        save_inference_samples(runs_dir, data_dir, session, IMAGE_SHAPE, logits, keep_prob, image_input)

        print("All done!")


# --------------------------
# MAIN
# --------------------------
if __name__ == '__main__':
    run()