import tensorflow as tf
import glob
import collections
import os


def load_examples(dirs, size, dataset, batch_size):
    '''
    directory: directory for disparity map
    size: size of cropped image, [h, w]

    return:
        lefts: [bz, HEIGHT, WIDTH, 3]           # float32 [-1,1]
        rights: [bz, HEIGHT, WIDTH, 3]          # float32 [-1,1]

    note: default, batch_size is 1. If dataset==cityscapes, resize images to
        512x1024 before cropping.

    '''

    HEIGHT = size[0]
    WIDTH = size[1]
    '''
    Needs to be modified if input input directory is changed
    '''
    if dirs[1] is not None:
        Examples = collections.namedtuple("Examples", "lefts, rights, count, batch_size")

        input_left_dir = glob.glob(os.path.join(dirs[0], '*.png'))
        input_right_dir = glob.glob(os.path.join(dirs[1], '*.png'))
        input_left_dir.sort()
        input_right_dir.sort()

        with tf.name_scope("load_images"):
            path_queue = tf.train.slice_input_producer([input_left_dir,
                                                        input_right_dir], shuffle=True)
            with tf.name_scope('read_left_image'):
                contents = tf.read_file(path_queue[0])
                left = tf.image.decode_png(contents, dtype=tf.uint8)
                left = tf.image.convert_image_dtype(left, dtype=tf.float32)
                # normalize image to [-1,1]
                left = preprocess(left)

            with tf.name_scope('read_right_image'):
                contents = tf.read_file(path_queue[1])
                right = tf.image.decode_png(contents, dtype=tf.uint8)
                right = tf.image.convert_image_dtype(right, dtype=tf.float32)
                right = preprocess(right)

            with tf.name_scope('random_crop_images'):
                total = tf.concat([left, right], axis=-1)
                if dataset == 'cityscapes':
                    total = tf.image.resize_images(total, [512, 1024])
                cropped_total = tf.random_crop(total, [HEIGHT, WIDTH, 6])
                left, right = tf.split(cropped_total, [3, 3], axis=-1)
            left_batch, right_batch = tf.train.batch([left, right], batch_size=batch_size)

        return Examples(
                    lefts=left_batch,
                    rights=right_batch,
                    count=len(input_left_dir),
                    batch_size=batch_size)


def preprocess(img):
    '''
    normalize image with float32 [0,1] to [-1,1]
    '''
    img = img * 2 - 1
    return img


if __name__ == "__main__":

    example = load_examples()
    lefts = example.lefts
    c = example.count

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        s = sess.run(lefts)
        print(s.shape)
