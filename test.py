import argparse
import numpy as np
import tensorflow as tf
import os
import cv2

from scipy.misc import imread
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from inpaint_model import DetInpaint
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', type=str,
                    help='The directory of tensorflow checkpoint.')

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    args = parser.parse_args()

    config_path = os.path.join('config.yml')
    config = Config(config_path)
    model = DetInpaint(config)
    image = imread(args.image)
    mask = imread(args.mask)
    mask = (mask > 173).astype(np.uint8) * 255
    assert image.shape == mask.shape

    image = tf.constant(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    mask = tf.constant(mask, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=0)
    mask = tf.expand_dims(mask, axis=-1)

    image /= 255
    mask /= 255

    images_masked = (image * (1 - mask)) + mask
    # input of the model
    inputs = tf.concat([images_masked, mask], axis=3)

    # process outputs
    output = model.inpaint_generator(inputs, 8, 64, 2)

    outputs_merged = (output * mask) + (image * (1 - mask))
    output *= 255
    outputs_merged *= 255
    images_masked *= 255
    image *= 255

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))

        sess.run(assign_ops)
        print('Model loaded.')
        gt, result, outputs_merged, masked = sess.run([image, output, outputs_merged, images_masked])
        cv2.imwrite('./merged.png', outputs_merged[0][:, :, ::-1])
        cv2.imwrite('./input.png', masked[0][:, :, ::-1])
        cv2.imwrite('./output.png', result[0][:, :, ::-1])
        m = psnr(outputs_merged[0][:, :, ::-1].astype(np.uint8), gt[0][:, :, ::-1].astype(np.uint8))
        p = ssim(outputs_merged[0][:, :, ::-1].astype(np.uint8), gt[0][:, :, ::-1].astype(np.uint8), multichannel=True, win_size=51)
        
        print(m ,p)

