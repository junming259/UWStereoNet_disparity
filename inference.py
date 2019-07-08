import os
import tensorflow as tf
import argparse
import numpy as np
import png
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--export_dir", default='export/dropuwstereo_disp_cityscapes/', help="path to folder containing export files")
parser.add_argument("--output_dir", default='prediction/dropuwstereo_disp_cityscapes/', help="path to folder to save reults")
parser.add_argument("--left_dir", default='data/test/left/', help="path to folder containing left-view images")
parser.add_argument("--right_dir", default='data/test/right/', help="path to folder containing right-view images")
parser.add_argument("--gpu", type=str, default='1')


a = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu


def main():
    """Inference."""

    left_dir = os.path.join(a.output_dir, 'left/')
    right_dir = os.path.join(a.output_dir, 'right/')

    if not os.path.exists(left_dir):
        os.makedirs(left_dir)
    if not os.path.exists(right_dir):
        os.makedirs(right_dir)

    if a.export_dir is None:
        raise Exception("checkpoint required for test mode")

    saver = tf.train.import_meta_graph(os.path.join(a.export_dir, 'export.meta'))
    graph = tf.get_default_graph()
    left_path = graph.get_tensor_by_name("left_input_path:0")
    right_path = graph.get_tensor_by_name("right_input_path:0")
    height = graph.get_tensor_by_name("height_image:0")
    width = graph.get_tensor_by_name("width_image:0")

    left_disp_pred = graph.get_tensor_by_name("left_disp_pred:0")
    right_disp_pred = graph.get_tensor_by_name("right_disp_pred:0")

    with tf.Session(graph=graph) as sess:
        print("loading exported model from: {}".format(a.export_dir))
        checkpoint = tf.train.latest_checkpoint(a.export_dir)
        saver.restore(sess, checkpoint)

        print('running...')

        filenames = os.listdir(a.left_dir)
        filenames.sort()
        for item in filenames:
            l_path = a.left_dir + item
            r_path = a.right_dir + item
            r_path = r_path.replace('left', 'right')
            shape = Image.open(l_path).size

            # processing
            feed_dict = {left_path: l_path, right_path: r_path, width: shape[0], height: shape[1]}
            left_disp, right_disp = sess.run([left_disp_pred, right_disp_pred], feed_dict)

            # squeeze extra dimension
            left_disp = np.squeeze(left_disp)
            right_disp = np.squeeze(right_disp)

            # path of output
            left_disp_output_path = os.path.join(left_dir, item)
            right_disp_output_path = os.path.join(right_dir, item)

            # save output as uint16
            with open(left_disp_output_path, 'wb') as f:
                writer = png.Writer(width=left_disp.shape[1], height=left_disp.shape[0], greyscale=True, bitdepth=16)
                z2list = left_disp.tolist()
                writer.write(f, z2list)

            with open(right_disp_output_path, 'wb') as f:
                writer = png.Writer(width=right_disp.shape[1], height=right_disp.shape[0], greyscale=True, bitdepth=16)
                z2list = right_disp.tolist()
                writer.write(f, z2list)

            print('writing: {}'.format(item))


if __name__ == "__main__":
    main()
