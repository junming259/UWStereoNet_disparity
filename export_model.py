import os
import argparse
import tensorflow as tf
from model import create_costVolume, modual3D, refinement, predict
from res_bone import Res_bone
from image_reader import preprocess


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", default='checkpoint/dropuwstereo_disp_cityscapes/', help="path to folder containing checkpoint")
parser.add_argument("--export_dir", default='export/dropuwstereo_disp_cityscapes/', help="path to folder to save export files")
parser.add_argument("--max_num_disparity", type=int, default=192, help="maximum value for disparity")
parser.add_argument("--gpu", type=str, default='2')

a = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu


def read_image(path):
    '''
    path: input string, dtype=tf.string
    return: normalized image, with shape [1,h,w,3], range [-1,1]
    '''
    with tf.variable_scope('read_image'):
        content = tf.read_file(path)
        image = tf.image.decode_png(content)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = preprocess(image)                 # normalize image to [-1,1]
        image.set_shape([None, None, 3])
        image = tf.expand_dims(image, axis=0)
    return image


def main():
    """Create the model and start inference."""

    '''
    1.read image
    '''

    left_input_path = tf.placeholder(tf.string, name='left_input_path')
    right_input_path = tf.placeholder(tf.string, name='right_input_path')
    h_img = tf.placeholder(tf.int32, name='height_image')
    w_img = tf.placeholder(tf.int32, name='width_image')

    left = read_image(left_input_path)
    right = read_image(right_input_path)

    '''
    2. build model
    '''
    with tf.name_scope('build_graph'):
        with tf.variable_scope('model'):

            left_res = Res_bone(left, is_train=True, reuse=False)
            right_res = Res_bone(right, is_train=True, reuse=True)
            left_cost_volume, right_cost_volume = create_costVolume(left_res.disp_feature, right_res.disp_feature, a.max_num_disparity)

            with tf.variable_scope('Initial'):
                # initial disparity estimation
                left_initial_disp_logits = modual3D(left_cost_volume, is_train=True, reuse=False)
                right_initial_disp_logits = modual3D(right_cost_volume, is_train=True, reuse=True)

            with tf.variable_scope('Refined'):
                # refinement
                left_refined_disp_logits = refinement(left_initial_disp_logits, left_res.seg_embedding, is_train=True, reuse=False)
                right_refined_disp_logits = refinement(right_initial_disp_logits, right_res.seg_embedding, is_train=True, reuse=True)
                # disparity estimation, same size of original stereo images
                left_disp = predict(left_refined_disp_logits, [h_img, w_img, a.max_num_disparity+1], name='predict_left_disp')
                right_disp = predict(right_refined_disp_logits, [h_img, w_img, a.max_num_disparity+1], name='predict_right_disp')

    left_disp_pred = tf.cast(left_disp*256, dtype=tf.uint16)
    right_disp_pred = tf.cast(right_disp*256, dtype=tf.uint16)
    left_disp_pred = tf.identity(left_disp_pred, name='left_disp_pred')
    right_disp_pred = tf.identity(right_disp_pred, name='right_disp_pred')

    '''
    3. saver setting
    '''
    with tf.name_scope('saver'):
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

    '''
    4. exporting model
    '''
    with tf.Session() as sess:
        print('loading model from: {}'.format(a.checkpoint_dir))
        checkpoint = tf.train.latest_checkpoint(a.checkpoint_dir)
        restore_saver.restore(sess, checkpoint)
        print('export model to: {}'.format(a.export_dir))
        export_saver.export_meta_graph(filename=os.path.join(a.export_dir, "export.meta"))
        export_saver.save(sess, os.path.join(a.export_dir, "export"), write_meta_graph=False)


if __name__ == "__main__":
    main()
