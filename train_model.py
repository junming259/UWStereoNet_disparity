import argparse
import os
import tensorflow as tf
import logging
from res_bone import Res_bone
from image_reader import load_examples
from utils import deprocess
from model import create_costVolume, modual3D, predict, compute_loss, refinement

parser = argparse.ArgumentParser()
parser.add_argument("--left_dir", default='data/cityscapes/train/left/', help="path to folder containing left-view training images")
parser.add_argument("--right_dir", default='data/cityscapes/train/right/', help="path to folder containing right-view training images")
parser.add_argument("--left_val_dir", default='data/cityscapes/val/left/', help="path to folder containing left-view validation images")
parser.add_argument("--right_val_dir", default='data/cityscapes/val/right/', help="path to folder containing right-view validation images")
parser.add_argument("--checkpoint_dir", default='checkpoint/dropuwstereo_disp_cityscapes/', help="where to put checkpoints files")
parser.add_argument("--summary_dir", default='summary/dropuwstereo_disp_cityscapes/', help="where to put summary files")
parser.add_argument("--resume_dir", default=None, help="directory with checkpoint to resume training from")

parser.add_argument("--num_steps", type=int, default=100000, help="number of training steps")
parser.add_argument("--summary_freq", type=int, default=15, help="frequency to update summaries")
parser.add_argument("--schedule_freq", type=int, default=50000, help="frequency to half learning rate")
parser.add_argument("--print_summary_freq", type=int, default=50, help="frequency to print summary")
parser.add_argument("--save_freq", type=int, default=10000, help="frequency to save model")

parser.add_argument("--w1", type=float, default=0.3, help="weight for initial disparity loss")
parser.add_argument("--w2", type=float, default=0.7, help="weight for refined disparity loss")
parser.add_argument("--beta1", type=float, default=0.8, help="initial disparity loss")
parser.add_argument("--beta2", type=float, default=0.01, help="initial disparity loss")
parser.add_argument("--beta3", type=float, default=0.001, help="initial disparity loss")
parser.add_argument("--gamma1", type=float, default=0.8, help="refined disparity loss")
parser.add_argument("--gamma2", type=float, default=0.02, help="refined disparity loss")
parser.add_argument("--gamma3", type=float, default=0.002, help="refined disparity loss")
parser.add_argument("--height", type=int, default=256, help="crop images to this height")
parser.add_argument("--width", type=int, default=512, help="crop images to this width")
parser.add_argument("--max_num_disparity", type=int, default=192, help="maximum value for disparity")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate for adam")
parser.add_argument("--dataset", type=str, default='cityscapes', choices=["kitti", "cityscapes"])
parser.add_argument("--gpu", type=str, default='0', help="which gpu to use")
parser.add_argument('--is_val', dest='is_val', action='store_true', help="show validation loss")

a = parser.parse_args()

if not os.path.exists(a.summary_dir):
    os.makedirs(a.summary_dir)
if not os.path.exists(a.checkpoint_dir):
    os.makedirs(a.checkpoint_dir)

# save logging parameters
logging.basicConfig(filename=a.summary_dir+'parameters.log', level=logging.DEBUG)
adict = vars(parser.parse_args())
keys = list(adict.keys())
keys.sort()
for item in keys:
    logging.info('{0}:{1}'.format(item, adict[item]))


TARGET_SHAPE = [a.height, a.width, a.max_num_disparity+1]
WEIGHTS_LIST = [a.beta1, a.beta2, a.beta3, a.gamma1, a.gamma2, a.gamma3]
os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu


def load_data(dirs, size, name='load_data'):save_
    '''
    load left image, right image and disparity map
    '''
    with tf.variable_scope(name):
        examples = load_examples(dirs, size, a.dataset, a.batch_size)
        left = examples.lefts
        right = examples.rights

        return left, right, examples.count, examples.batch_size


def build_model(input, is_train=True, reuse=False):
    with tf.variable_scope('model'):
        left_res = Res_bone(input[0], is_train=is_train, reuse=reuse)
        right_res = Res_bone(input[1], is_train=is_train, reuse=True)
        left_cost_volume, right_cost_volume = create_costVolume(left_res.disp_feature, right_res.disp_feature, a.max_num_disparity)
        with tf.variable_scope('Initial'):
            # initial disparity estimation
            left_initial_disp_logits = modual3D(left_cost_volume, is_train=is_train, reuse=reuse)
            right_initial_disp_logits = modual3D(right_cost_volume, is_train=is_train, reuse=True)
            # disparity estimation, same size of original stereo images
            left_initial_disp = predict(left_initial_disp_logits, TARGET_SHAPE, name='left_disp')
            right_initial_disp = predict(right_initial_disp_logits, TARGET_SHAPE, name='right_disp')
            L1 = compute_loss(
                input[0], input[1],
                left_initial_disp,
                right_initial_disp,
                left_res.seg_embedding,
                right_res.seg_embedding,
                WEIGHTS_LIST,
                name='Initial_loss'
            )

        with tf.variable_scope('Refined'):
            # refinement
            left_refined_disp_logits = refinement(left_initial_disp_logits, left_res.seg_embedding, is_train=is_train, reuse=reuse)
            right_refined_disp_logits = refinement(right_initial_disp_logits, right_res.seg_embedding, is_train=is_train, reuse=True)
            # disparity estimation, same size of original stereo images
            left_refined_disp = predict(left_refined_disp_logits, TARGET_SHAPE, name='left_disp')
            right_refined_disp = predict(right_refined_disp_logits, TARGET_SHAPE, name='right_disp')
            L2 = compute_loss(
                input[0], input[1],
                left_refined_disp,
                right_refined_disp,
                left_res.seg_embedding,
                right_res.seg_embedding,
                WEIGHTS_LIST,
                name='Refined_loss'
            )
        loss = a.w1*L1 + a.w2*L2

    return loss, L1, L2, left_initial_disp, right_initial_disp, left_refined_disp, right_refined_disp


def main():
    """Create the model and start the training."""

    '''
    1. create image reader
    '''

    with tf.device('/cpu:0'):

        left, right, count, batch_size = load_data([a.left_dir, a.right_dir], [a.height, a.width], name='load_data')
        if a.is_val:
            left_val, right_val, val_count, val_batch_size = load_data([a.left_val_dir, a.right_val_dir], [a.height, a.width], name='load_val_data')

        print('Num_data: {}'.format(count))
        if a.is_val:
            print('Num_val: {}'.format(val_count))

        '''
        2. build model, the prediction
        '''

        with tf.device('/gpu:0'):
            with tf.name_scope('build_graph'):

                loss, l_init, l_ref, left_initial_disp, right_initial_disp, left_refined_disp, right_refined_disp = build_model([left, right], is_train=True, reuse=False)
                if a.is_val:
                    val_loss, _, _, _, _, _, _ = build_model([left_val, right_val], is_train=False, reuse=True)

            '''
            3. do updating
            '''
            with tf.name_scope('train'):
                global_step = tf.Variable(0, trainable=False, name='global_step')
                # lr = tf.train.exponential_decay(LEARNING_RATE, global_step, 10, 0.96, staircase=True, name='learning_rate')
                rate = tf.pow(0.5, tf.cast(tf.cast(global_step/a.schedule_freq, tf.int32), tf.float32))
                lr = a.lr * rate
                tf.summary.scalar('learning_rate', lr, collections=['train_summary'])
                tf.summary.scalar('step', global_step, collections=['train_summary'])
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(lr)
                    optimize = optimizer.minimize(loss, global_step)

            with tf.name_scope('loss'):
                tf.summary.scalar('loss', loss, collections=['train_summary'])
                tf.summary.scalar('loss_init', l_init, collections=['train_summary'])
                tf.summary.scalar('loss_ref', l_ref, collections=['train_summary'])
                loss_val = tf.placeholder(tf.float32, [])
                loss_val_sum = tf.summary.scalar('loss_val', loss_val)
            with tf.name_scope('input_images'):
                tf.summary.image("left", deprocess(left), max_outputs=1, collections=['train_summary'])
                tf.summary.image("right", deprocess(right), max_outputs=1, collections=['train_summary'])
            with tf.name_scope('disp_images'):
                tf.summary.image("left_disp_refined", left_refined_disp, max_outputs=1, collections=['train_summary'])
                tf.summary.image("right_disp_refined", right_refined_disp, max_outputs=1, collections=['train_summary'])
                tf.summary.image("left_disp_init", left_initial_disp, max_outputs=1, collections=['train_summary'])
                tf.summary.image("right_disp_init", right_initial_disp, max_outputs=1, collections=['train_summary'])

        '''
        3. training setting
        '''
        with tf.name_scope('save'):
            saver = tf.train.Saver(max_to_keep=8)
            summary_writer = tf.summary.FileWriter(a.summary_dir)
            # summary_op = tf.summary.merge([loss_sum,left_sum,right_sum,left_disp_sum,right_disp_sum, step_sum, lr_sum])
            summary_op = tf.summary.merge_all(key='train_summary')
            init = tf.global_variables_initializer()

        '''
        4. begin to train
        '''
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            if a.resume_dir is not None:
                # restoring from the checkpoint file
                ckpt = tf.train.get_checkpoint_state(a.resume_dir)
                if ckpt is not None:
                    tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
                    sess.run(global_step.assign(0))     # reset global_step to zero
                    print('Reload from: {}'.format(a.resume_dir))
                else:
                    sess.run(init)
            else:
                sess.run(init)
                # sess.run(load_pretrained_parameters)

            summary_writer.add_graph(sess.graph)

            tf.train.start_queue_runners(sess=sess)

            for step in range(a.num_steps):

                _, l, step = sess.run([optimize, loss, global_step])
                train_epoch = step * a.batch_size // count

                if step % a.summary_freq == 0:
                    s = sess.run(summary_op)
                    summary_writer.add_summary(s, step)
                    summary_writer.flush()
                    print('-------- summary saved --------')

                if a.is_val:
                    if step % count == 0:
                        print('Running Validation')
                        # iterate through validation set
                        total_vl = 0
                        for i in range(0, val_count):
                            vl = sess.run(val_loss)
                            total_vl = total_vl + vl
                        vl_avg = 1.0*total_vl/val_count

                        s = sess.run(loss_val_sum, {loss_val: vl_avg})
                        summary_writer.add_summary(s, step)
                        summary_writer.flush()
                        print('-------- training_loss:{0:.4f}    validation_loss:{1:.4f}'.format(l, vl_avg))

                if step % a.save_freq == 0 and step != 0:
                    saver.save(sess, a.checkpoint_dir + 'model.ckpt', global_step=step)
                    print('-------- checkpoint saved:{} --------'.format(step))

                if step % a.print_summary_freq == 0:
                    print('epoch:{0}    step:{1}   loss:{2:.4f}'.format(train_epoch, step, l))

            # after loop
            saver.save(sess, a.checkpoint_dir + 'model.ckpt', global_step=step)
            print('-------- checkpoint saved:{} --------'.format(step))


if __name__ == "__main__":
    main()
