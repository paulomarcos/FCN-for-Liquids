import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import scipy.io
import glob
from glob import glob
import cv2
import scipy.misc as misc
import warnings
from distutils.version import LooseVersion
import helper
import project_tests as tests
import skimage
import time, datetime

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

NUMBER_OF_CLASSES = 2
DATA_DIRECTORY = './data'
DATA_DIR_TTV = './data/data_liquid_ttv'
RUNS_DIRECTORY = './runs'
VGG_PATH = './data/vgg'
TRAINING_DATA_DIRECTORY = DATA_DIR_TTV + "/train"
TESTING_DATA_DIRECTORY = DATA_DIR_TTV + "/test"
VALIDATION_DIRECTORY = DATA_DIR_TTV + "/val"

NUMBER_OF_IMAGES = len(glob(TRAINING_DATA_DIRECTORY+"/*color*"))

IMAGE_SHAPE = (192, 192)


# HYPERPARAMETERS
EPOCHS = 80
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
DROPOUT = 0.75

correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUMBER_OF_CLASSES])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)

# Used for plotting to visualize if our training is going well given parameters
all_training_losses = []
val_acc_list, val_loss_list = [], []
test_acc_list, test_loss_list = [], []

model_name = "./vgg_mm_80e_csc_nir.ckpt"

# Check for a GPU
if not tf.test.gpu_device_name():
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
  """
  Load Pretrained VGG Model into TensorFlow.
  sess: TensorFlow Session
  vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
  return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3, layer4, layer7)
  """
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

def new_vgg_model(sess):
  """
  Load Pretrained VGG Model into TensorFlow.
  sess: TensorFlow Session
  vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
  return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3, layer4, layer7)
  """
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

def conv_1x1(layer, layer_name):
  """ Return the output of a 1x1 convolution of a layer """
  return tf.layers.conv2d(inputs = layer,
                          filters =  NUMBER_OF_CLASSES,
                          kernel_size = (1, 1),
                          strides = (1, 1),
                          name = layer_name)

def upsample(layer, k, s, layer_name):
  """ Return the output of transpose convolution given kernel_size k and strides s """
  # See: http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic
  return tf.layers.conv2d_transpose(inputs = layer,
                                    filters = NUMBER_OF_CLASSES,
                                    kernel_size = (k, k),
                                    strides = (s, s),
                                    padding = 'same',
                                    name = layer_name)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes = NUMBER_OF_CLASSES):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  vgg_layerX_out: TF Tensor for VGG Layer X output
  num_classes: Number of classes to classify
  return: The Tensor for the last layer of output
  """

  # Use a shorter variable name for simplicity
  layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
  layer3_nir, layer4_nir, layer7_nir = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

  # Apply a 1x1 convolution to encoder layers
  layer3x = conv_1x1(layer = layer3, layer_name = "layer3conv1x1")
  layer4x = conv_1x1(layer = layer4, layer_name = "layer4conv1x1")
  layer7x = conv_1x1(layer = layer7, layer_name = "layer7conv1x1")
  layer3x_nir = conv_1x1(layer = layer3_nir, layer_name = "layer3conv1x1_nir")
  layer4x_nir = conv_1x1(layer = layer4_nir, layer_name = "layer4conv1x1_nir")
  layer7x_nir = conv_1x1(layer = layer7_nir, layer_name = "layer7conv1x1_nir")

  # Add decoder layers to the network with skip connections and upsampling
  # Note: the kernel size and strides are the same as the example in Udacity Lectures
  #       Semantic Segmentation Scene Understanding Lesson 10-9: FCN-8 - Decoder
  decoderlayer1 = upsample(layer = layer7x, k = 4, s = 2, layer_name = "decoderlayer1")
  decoderlayer2 = tf.add(decoderlayer1, layer4x, name = "decoderlayer2")
  decoderlayer3 = upsample(layer = decoderlayer2, k = 4, s = 2, layer_name = "decoderlayer3")
  decoderlayer4 = tf.add(decoderlayer3, layer3x, name = "decoderlayer4")
  decoderlayer_output = upsample(layer = decoderlayer4, k = 16, s = 8, layer_name = "decoderlayer_output")

  decoderlayer1_nir = upsample(layer = layer7x_nir, k = 4, s = 2, layer_name = "decoderlayer1_nir")
  decoderlayer2_nir = tf.add(decoderlayer1_nir, layer4x_nir, name = "decoderlayer2_nir")
  decoderlayer3_nir = upsample(layer = decoderlayer2_nir, k = 4, s = 2, layer_name = "decoderlayer3_nir")
  decoderlayer4_nir = tf.add(decoderlayer3_nir, layer3x_nir, name = "decoderlayer4_nir")
  decoderlayer_nir_output = upsample(layer = decoderlayer4_nir, k = 16, s = 8, layer_name = "decoderlayer_nir_output")

  # Merge networks
  network = tf.add(decoderlayer_output, decoderlayer_nir_output)

  network = conv_1x1(layer = network, layer_name = "network_merged")
  network = tf.add(network, decoderlayer_nir_output)
  network = conv_1x1(layer = network, layer_name = "network_merged2")
  network = conv_1x1(layer = network, layer_name = "network_merged3")

  return network # decoderlayer_output

def optimize(nn_last_layer, correct_label, learning_rate, num_classes = NUMBER_OF_CLASSES):
  """
  Build the TensorFLow loss and optimizer operations.
  nn_last_layer: TF Tensor of the last layer in the neural network
  correct_label: TF Placeholder for the correct label image
  learning_rate: TF Placeholder for the learning rate
  num_classes: Number of classes to classify
  return: Tuple of (logits, train_op, cross_entropy_loss)
  """
  # reshape 4D tensors to 2D
  # Each row represents a pixel, each column a class
  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  class_labels = tf.reshape(correct_label, (-1, num_classes))
  # The cross_entropy_loss is the cost which we are trying to minimize to yield higher accuracy
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels)
  cross_entropy_loss = tf.reduce_mean(cross_entropy)

  # The model implements this operation to find the weights/parameters that would yield correct pixel labels
  train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

  return logits, train_op, cross_entropy_loss

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)

def train_nn(sess, epochs, batch_size, get_batches_fn_nir, logits, train_op,
             cross_entropy_loss, input_image, net_input_nir,
             correct_label, keep_prob, learning_rate, saver):
  """
  Train neural network and print out the loss during training.
  sess: TF Session
  epochs: Number of epochs
  batch_size: Batch size
  get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  train_op: TF Operation to train the neural network
  cross_entropy_loss: TF Tensor for the amount of loss
  input_image: TF Placeholder for input images
  correct_label: TF Placeholder for label images
  keep_prob: TF Placeholder for dropout keep probability
  learning_rate: TF Placeholder for learning rate
  """

  for epoch in range(EPOCHS):

    losses, i = [], 0
    st = time.time()
    epoch_st=time.time()

    for images, labels, nir_images in get_batches_fn_nir(BATCH_SIZE):
      i += 1

      feed = { input_image: images,
               net_input_nir: nir_images,
               correct_label: labels,
               keep_prob: DROPOUT,
               learning_rate: LEARNING_RATE }

      _, partial_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)

      if i % 20 == 0: #(i == 1) or (i % 20 == 0):
          print("---> iteration: ", i, " partial loss: ", partial_loss, "time: %.2f"%(time.time()-st))
          st = time.time()
      losses.append(partial_loss)


    val_acc, val_loss = helper.evaluate(val_dirs, sess, IMAGE_SHAPE, logits, keep_prob,
                         input_image, net_input_nir, train_op, cross_entropy_loss, correct_label,
                         DROPOUT, learning_rate, LEARNING_RATE, 20)

    test_acc, test_loss = helper.evaluate(test_dirs, sess, IMAGE_SHAPE, logits, keep_prob,
                         input_image, net_input_nir, train_op, cross_entropy_loss, correct_label,
                         DROPOUT, learning_rate, LEARNING_RATE, 20)

    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

    training_loss = sum(losses) / len(losses)
    all_training_losses.append(training_loss)
    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(EPOCHS-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Saving model as ", model_name)
    saver.save(sess, model_name)
    print("------------------")
    print("epoch: ", epoch + 1, " of ", EPOCHS)
    print("TRAIN LOSS: ", training_loss)
    print("\tVAL ACC: ", val_acc, " VAL LOSS ", val_loss)
    print("\tTEST ACC: ", test_acc, " TEST LOSS ", test_loss)
    print("------------------")
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    LOG(train_time)


# main:
print("NUMBER OF IMAGES:", NUMBER_OF_IMAGES)

# download vgg model
helper.maybe_download_pretrained_vgg(DATA_DIRECTORY)

train_dirs, test_dirs, val_dirs = [TRAINING_DATA_DIRECTORY], [TESTING_DATA_DIRECTORY], [VALIDATION_DIRECTORY]
# A function to get batches
get_batches_fn_nir = helper.gen_batch_function_nir_ttv(train_dirs, IMAGE_SHAPE)

with tf.Session() as session:

    # Returns the three layers, keep probability and input layer from the vgg architecture
    image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, VGG_PATH)

    net_input_nir = tf.placeholder(tf.float32, shape=[None,None, None, 3], name="net_input_nir")

    # The resulting network architecture, adding a decoder on top of the given vgg model
    model_output = layers(layer3, layer4, layer7, NUMBER_OF_CLASSES)

    # Returns the output logits, training operation and cost operation to be used
    # For the logits: each row represents a pixel, each column a class
    # training operation is what is used to get the right parameters to the model to correctly label the pixels
    # the cross entropy loss is the cost which we are minimizing, lower cost should yield higher accuracy
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, NUMBER_OF_CLASSES)

    # Initilize all variables
    session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    # Create saver
    saver = tf.train.Saver()

    # train the neural network
    train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn_nir, logits,
             train_op, cross_entropy_loss, image_input, net_input_nir,
             correct_label, keep_prob, learning_rate, saver)

    print("\n\nTRAIN LOSSES: ", all_training_losses)
    print("\n\nVAL ACC: ", val_acc_list)
    print("\n\nVAL LOSS: ", val_loss_list)
    print("\n\nTEST ACC: ", test_acc_list)
    print("\n\nTEST LOSS: ", test_loss_list)

    # Save inference data
    helper.save_inference_samples_nir_ttv(RUNS_DIRECTORY, test_dirs, session, IMAGE_SHAPE, logits, keep_prob, image_input, net_input_nir)
