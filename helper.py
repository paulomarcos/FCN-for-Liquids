import re
import random
from random import Random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import sys
import cv2
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


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
        for folder in data_folder:
            image_paths = glob(os.path.join(folder, 'color*.png')) # previously 'data*.png'
            label_paths = {
                re.sub(r'ground_truth', 'color', os.path.basename(path)): path # previously 'ground_truth', 'data'
                for path in glob(os.path.join(folder, 'ground_truth*.png'))}
            background_color = np.array([0, 0, 0, 0])

            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                    gt_bg = np.all(gt_image == background_color, axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                    images.append(image)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_batch_function_nir(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn_nir(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        for folder in data_folder:
            image_paths = glob(os.path.join(folder, 'color*.png')) # previously 'data*.png'
            label_paths = {
                re.sub(r'ground_truth', 'color', os.path.basename(path)): path # previously 'ground_truth', 'data'
                for path in glob(os.path.join(folder, 'ground_truth*.png'))}
            background_color = np.array([0, 0, 0, 0])

            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                    img_id = image_file[-8:]
                    nir = cv2.imread(folder+"/nir_"+img_id)
                    #print(folder+"/nir_"+img_id)
                    nir = scipy.misc.imresize(nir, image_shape)
                    overlay = cv2.addWeighted(image,0.5,nir,0.5,0)

                    gt_bg = np.all(gt_image == background_color, axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                    images.append(overlay)
                    gt_images.append(gt_image)

                yield np.array(images), np.array(gt_images)
    return get_batches_fn_nir

def gen_batch_function_nir_ttv(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn_nir_ttv(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        print("data_folder: ", data_folder)
        for folder in data_folder:
            image_paths = glob(os.path.join(folder, '*color*.png')) # previously 'data*.png'
            label_paths = {
                re.sub(r'ground_truth', 'color', os.path.basename(path)): path # previously 'ground_truth', 'data'
                for path in glob(os.path.join(folder, '*ground_truth*.png'))}
            background_color = np.array([0, 0, 0, 0])

            random.shuffle(image_paths)
            for batch_i in range(0, len(image_paths), batch_size):
                images = []
                gt_images = []
                nir_images = []
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
                    _, filename = os.path.split(image_file)
                    fd_id = filename[0]
                    img_id = image_file[-8:]
                    #print(folder+"/"+fd_id+"_nir_"+img_id)
                    nir = cv2.imread(folder+"/"+fd_id+"_nir_"+img_id)
                    #print(folder+"/nir_"+img_id)
                    nir = scipy.misc.imresize(nir, image_shape)

                    gt_bg = np.all(gt_image == background_color, axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                    images.append(image)
                    gt_images.append(gt_image)
                    nir_images.append(nir)

                yield np.array(images), np.array(gt_images), np.array(nir_images)
    return get_batches_fn_nir_ttv


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    i = 0
    for folder in data_folder:
        print(folder)
        for image_file in glob(os.path.join(folder, 'color*.png')): # previously 'data*.png'
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 0, 255, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")

            result = np.dot(segmentation, np.array([[0, 0, 255, 255]]))
            result = scipy.misc.toimage(result, mode="RGBA")

            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            base_name = os.path.basename(image_file)
            base_name = str(i)+"_"+base_name
            yield base_name, np.array(street_im), result
        i += 1


def gen_test_output_nir(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    i = 0
    for folder in data_folder:
        print(folder)
        for image_file in glob(os.path.join(folder, '*color*.png')): # previously 'data*.png'
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            _, filename = os.path.split(image_file)
            fd_id = filename[0]
            img_id = image_file[-8:]
            nir = cv2.imread(folder+"/"+fd_id+"_nir_"+img_id)
            nir = scipy.misc.imresize(nir, image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 0, 255, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")

            result = np.dot(segmentation, np.array([[0, 0, 255, 255]]))
            result = scipy.misc.toimage(result, mode="RGBA")

            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            base_name = os.path.basename(image_file)
            base_name = str(i)+"_"+base_name
            yield base_name, np.array(street_im), result
        i += 1

def gen_test_output_nir_ttv(sess, logits, keep_prob, image_pl, image_input_nir, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    i = 0
    for folder in data_folder:
        print(folder)
        j = 0
        image_files = glob(os.path.join(folder, '*color*.png'))
        max_iter = len(image_files)
        for image_file in image_files: # previously 'data*.png'
            sys.stdout.write("\rRunning test image %d / %d"%(j+1, max_iter))
            sys.stdout.flush()

            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            _, filename = os.path.split(image_file)
            fd_id = filename[0]
            img_id = image_file[-8:]
            nir = cv2.imread(folder+"/"+fd_id+"_nir_"+img_id)
            nir = scipy.misc.imresize(nir, image_shape)

            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image], image_input_nir: [nir]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 0, 255, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")

            result = np.dot(segmentation, np.array([[0, 0, 255, 255]]))
            result = scipy.misc.toimage(result, mode="RGBA")

            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            base_name = os.path.basename(image_file)
            base_name = str(i)+"_"+base_name
            j += 1
            yield base_name, np.array(street_im), result
        print("")
        i += 1

def evaluate(data_dir, sess, image_shape, logits, keep_prob, input_image, input_image_nir, train_op, cross_entropy_loss, correct_label, dropout, lr_tensor, learning_rate, batch_test):
    i = 0
    losses = []
    iou_scores = []
    background_color = np.array([0, 0, 0, 0])
    for folder in data_dir:
        j = 0
        image_files = glob(os.path.join(folder, '*color*.png'))
        if (batch_test != None) and (batch_test <= len(image_files)):
            Random(4).shuffle(image_files)
            image_files = image_files[:batch_test]
        else:
            raise "batch_test is None or greater than the test set"
        max_iter = len(image_files)
        for image_file in image_files: # previously 'data*.png'
            sys.stdout.write("\rRunning test image %d / %d"%(j+1, max_iter))
            sys.stdout.flush()

            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            _, filename = os.path.split(image_file)
            fd_id = filename[0]
            img_id = image_file[-8:]
            nir = cv2.imread(folder+"/"+fd_id+"_nir_"+img_id)
            nir = scipy.misc.imresize(nir, image_shape)
            gt_img = cv2.imread(folder+"/"+fd_id+"_ground_truth_"+img_id)
            gt_img = scipy.misc.imresize(gt_img, image_shape)

            gt_image = scipy.misc.imresize(scipy.misc.imread(folder+"/"+fd_id+"_ground_truth_"+img_id), image_shape)
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

            # Calculate Loss
            feed = { input_image: [image],
                     input_image_nir: [nir],
                     correct_label: [gt_image],
                     keep_prob: dropout,
                     lr_tensor: learning_rate}
            _, partial_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
            losses.append(partial_loss)

            # Calculate accuracy
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, input_image: [image], input_image_nir: [nir]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([[0, 0, 255, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")

            result = np.dot(segmentation, np.array([[0, 0, 255]]))#, 255]]))
            #result = scipy.misc.toimage(result, mode="RGB")
            #Calculate IoU
            intersection = np.logical_and(gt_img, segmentation)
            union = np.logical_or(gt_img, result)
            iou_score = np.sum(intersection) / np.sum(union)
            if np.isnan(iou_score):
                iou_score = 1
            else:
                iou_score = round(iou_score, 5)
            iou_scores.append(round(iou_score, 5))


            j += 1
        print("")
        i += 1
    return np.mean(iou_scores), np.mean(losses)

def save_inference_samples_nir(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output_nir(
        sess, logits, keep_prob, input_image, data_dir, image_shape)
    print("@@@@@@@@@@IMAGE OUTPUTS@@@@@@@")
    for name, image, result in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(output_dir, "result_"+name), result)


def save_inference_samples_nir_ttv(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, input_image_nir):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output_nir_ttv(
        sess, logits, keep_prob, input_image, input_image_nir, data_dir, image_shape)
    print("@@@@@@@@@@IMAGE OUTPUTS@@@@@@@")
    for name, image, result in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(output_dir, "result_"+name), result)
    print("Done.")


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)
    print("@@@@@@@@@@IMAGE OUTPUTS@@@@@@@")
    for name, image, result in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        scipy.misc.imsave(os.path.join(output_dir, "result_"+name), result)
