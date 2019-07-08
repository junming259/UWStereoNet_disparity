import tensorflow as tf
from utils import conv3d_norm_lrelu, deconv3d_norm_lrelu, conv_norm_lrelu, convBlock, identityBlock, deconv_norm_lrelu

SCALE = 1
THRESHOLD = 10


def create_costVolume(left_res, right_res, max_num_disparity):
    '''
    Create cost volume for both side
    '''
    with tf.variable_scope('cost_volume'):
        # do feature concatenation
        left_costVolume = left_concatenate_features(left_res, right_res, max_num_disparity)
        right_costVolume = right_concatenate_features(right_res, left_res, max_num_disparity)
    return left_costVolume, right_costVolume


def modual3D(inputs, is_train, reuse):
    '''
    inputs: [n,d,h,w,f*2]
    output: [n,h,w,d]
    '''
    with tf.variable_scope('modual3D', reuse=reuse):
        inputs = tf.identity(inputs, name='inputs')
        num_features = inputs.get_shape().as_list()[-1]//2
        # pad inputs to fix dimension error
        padded_inputs = tf.pad(inputs, [[0, 0], [1, 2], [2, 2], [2, 2], [0, 0]])
        # padded_inputs = pad(inputs)
        conv = conv3d_norm_lrelu(padded_inputs, num_features, 5, strides=(1, 1, 1), is_train=is_train, padding='valid', name='3Dconv')
        conv0 = conv3d_norm_lrelu(conv, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Dconv0')
        conv1 = conv3d_norm_lrelu(conv0, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Dconv1')
        conv2 = conv3d_norm_lrelu(conv1, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Dconv2')
        conv3 = conv3d_norm_lrelu(conv2, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Dconv3')

        conv4 = deconv3d_norm_lrelu(conv3, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Ddeconv4')
        conv4 = res_shortcut(conv4, conv2, is_train=is_train, name='shortcut1')

        conv5 = deconv3d_norm_lrelu(conv4, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Ddeconv5')
        conv5 = res_shortcut(conv5, conv1, is_train=is_train, name='shortcut2')

        conv6 = deconv3d_norm_lrelu(conv5, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Ddeconv6')
        conv6 = res_shortcut(conv6, conv0, is_train=is_train, name='shortcut3')

        conv7 = deconv3d_norm_lrelu(conv6, num_features, 3, strides=(2, 2, 2), is_train=is_train, name='3Ddeconv7')
        conv7 = res_shortcut(conv7, conv, is_train=is_train, name='shortcut4')

        conv8 = tf.layers.conv3d(conv7, 1, 3, strides=(1, 1, 1), padding='same', use_bias=False, name='conv3d')
        output = tf.squeeze(conv8, axis=-1)
        # initial disparity estimation
        output = tf.transpose(output, [0, 2, 3, 1])        # [n,h,w,d]
    return output


def res_shortcut(x, y, is_train, name):
    '''
    x: previous layer
    y: low-level feature
    '''
    f = y.shape.as_list()[-1]
    with tf.variable_scope(name):
        y = conv3d_norm_lrelu(y, f, 3, strides=(1, 1, 1), is_train=is_train, name='conv1')
        y = conv3d_norm_lrelu(y, f, 3, strides=(1, 1, 1), is_train=is_train, name='conv2')
        output = tf.add(x, y)
    return output


def refinement(disp_logits, seg_embedding, is_train, reuse):
    with tf.variable_scope('refinement', reuse=reuse):
        d = disp_logits.shape.as_list()[-1]
        seg_embedding = tf.image.resize_images(seg_embedding, tf.shape(disp_logits)[1:3])
        concated = tf.concat([disp_logits, seg_embedding], axis=-1)

        conv1 = conv_norm_lrelu(concated, 64, kernel_size=5, strides=(2, 2), is_train=is_train, withLrelu=False, name="conv1")

        l2a = convBlock(conv1, [32, 32, 64], strides=(1, 1), is_train=is_train, name='l2a')
        l2b = identityBlock(l2a, [32, 32, 64], is_train=is_train, name='l2b')
        l2c = identityBlock(l2b, [32, 32, 64], is_train=is_train, name='l2c')

        l3a = convBlock(l2c, [64, 64, 128], strides=(1, 1), is_train=is_train, name='l3a')
        l3b = identityBlock(l3a, [64, 64, 128], is_train=is_train, name='l3b')
        l3c = identityBlock(l3b, [64, 64, 128], is_train=is_train, name='l3c')

        l4a = deconv_norm_lrelu(l3c, 64, 3, strides=(2, 2), is_train=is_train, name='deconvl4a')
        residul = tf.layers.conv2d(l4a, d, 3, padding='same', name='output')
        output = tf.add(disp_logits, residul)
    return output


def left_concatenate_features(left, right, max_num_disparity):
    '''
    Based on left feature map, concatenate all potential disparity from right feature map
    left: feature tensor, with shape [bz, h, w, feature_size]
    right: feature tensor, with shape [bz, h, w, feature_size]
    return: score [bz,d,h,w,2*feature_size], d is the MAX_NUM_DISPARITY
    '''
    with tf.variable_scope('left_match_features'):
        width = tf.shape(left)[2]
        max_num_features = max_num_disparity // 4
        concated_features = []
        # concate features from left and right images
        for d in range(max_num_features+1):
            left_features = left
            right_features = right[:, :, :width-d, :]
            right_features = tf.pad(right_features, [[0, 0], [0, 0], [d, 0], [0, 0]])
            features = tf.concat([left_features, right_features], axis=-1)
            concated_features.append(features)

        # concated_features with shape [n,d,h,w,f*2]
        concated_features = tf.stack(concated_features, axis=1)
    return concated_features


def right_concatenate_features(right, left, max_num_disparity):
    '''
    Based on right feature map, concatenate all potential disparity from left feature map
    right: feature tensor, with shape [bz, h, w, feature_size]
    left: feature tensor, with shape [bz, h, w, feature_size]
    return: score [bz,d,h,w,2*feature_size], d is the MAX_NUM_DISPARITY
    '''
    with tf.variable_scope('right_match_features'):
        max_num_features = max_num_disparity // 4
        concated_features = []
        # concate features from left and right images
        for d in range(max_num_features+1):
            right_features = right
            left_features = left[:, :, d:, :]
            left_features = tf.pad(left_features, [[0, 0], [0, 0], [0, d], [0, 0]])
            features = tf.concat([right_features, left_features], axis=-1)
            concated_features.append(features)

        # concated_features with shape [n,d,h,w,f*2]
        concated_features = tf.stack(concated_features, axis=1)
    return concated_features


def reconstruct_right(inputs, disp, name='recons_right'):
    '''
    Reconstruct right image from left image and left disparity
    '''
    with tf.variable_scope(name):
        return reconstruct_img(inputs, disp)


def reconstruct_left(inputs, disp, name='recons_left'):
    '''
    Reconstruct left image from right image and right disparity
    '''
    with tf.variable_scope(name):
        return reconstruct_img(inputs, -disp)


def reconstruct_img(inputs, disp, name='reconstruct_img'):
    '''
    This function is borrowed from https://github.com/mrharicot/monodepth/blob/master/bilinear_sampler.py
    inputs: left images
    disp: disparity maps from left side
    return: reconstructed right image
    '''
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0

            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)

            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)

            return weight_l * pix_l + weight_r * pix_r

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0, _width_f - 1.0, _width),
                                   tf.linspace(0.0, _height_f - 1.0, _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1])

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch = tf.shape(inputs)[0]
        _height = tf.shape(inputs)[1]
        _width = tf.shape(inputs)[2]
        _num_channels = tf.shape(inputs)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f = tf.cast(_width, tf.float32)

        output = _transform(inputs, disp)

    return output


def predict(inputs, target_shape, name='prediction'):
    '''
    Make prediction from disp logits, [bz, h, w, MAX_NUM_DISPARITY]
    return: [bz, h, w, 1]
    '''
    with tf.variable_scope(name):
        # bilinear upsample
        inputs = resize_tensor(inputs, target_shape)
        score = tf.nn.softmax(inputs*SCALE, axis=-1)
        # times with indices
        ind = tf.range(target_shape[-1], dtype=tf.float32)
        ind = tf.reshape(ind, [1, 1, 1, -1])
        score = score * ind
        # summation
        score = tf.reduce_sum(score, axis=-1)
        score = tf.expand_dims(score, axis=-1)
    return score


def resize_tensor(inputs, shape, name='resize_tensor'):
    '''
    Inputs: tensor with shape [bn,h,w,f]
    shape: a list, [H,W,F]
    return: a tensor with shape [bn,H,W,F] by bilinear interpolation
    '''
    with tf.variable_scope(name):
        inputs = tf.image.resize_images(inputs, [shape[0], shape[1]])     # [bn,H,W,f]
        inputs = tf.transpose(inputs, [0, 1, 3, 2])     # [bn,H,f,W]
        inputs = tf.image.resize_images(inputs, [shape[0], shape[2]])       # [bn,H,F,W]
        output = tf.transpose(inputs, [0, 1, 3, 2])
    return output


def unary_term(left, right, recons_left, recons_right):
    with tf.variable_scope('unary_term'):
        left = left[:, :, THRESHOLD:, :]
        recons_left = recons_left[:, :, THRESHOLD:, :]
        right = right[:, :, :-THRESHOLD, :]
        recons_right = recons_right[:, :, :-THRESHOLD, :]

        # SSIM
        with tf.name_scope('SSIM_loss'):
            SSIM_left = compute_SSIM_loss(recons_left, left, name='SSIM_left')
            SSIM_right = compute_SSIM_loss(recons_right, right, name='SSIM_right')
            SSIM_loss = SSIM_left + SSIM_right

        # reconstruction error of image
        with tf.name_scope('recon_img_loss'):
            l_img = tf.reduce_mean(tf.abs(left - recons_left))
            r_img = tf.reduce_mean(tf.abs(right - recons_right))

        # reconstruction error of gradient image
        with tf.name_scope('recon_grad_loss'):
            left_grad_x = gradient(left, axis='x')
            left_grad_y = gradient(left, axis='y')

            right_grad_x = gradient(right, axis='x')
            right_grad_y = gradient(right, axis='y')

            recons_left_grad_x = gradient(recons_left, axis='x')
            recons_left_grad_y = gradient(recons_left, axis='y')

            recons_right_grad_x = gradient(recons_right, axis='x')
            recons_right_grad_y = gradient(recons_right, axis='y')

            left_grad_loss = tf.reduce_mean(tf.abs(left_grad_x - recons_left_grad_x)) + tf.reduce_mean(tf.abs(left_grad_y - recons_left_grad_y))
            right_grad_loss = tf.reduce_mean(tf.abs(right_grad_x - recons_right_grad_x)) + tf.reduce_mean(tf.abs(right_grad_y - recons_right_grad_y))

        loss = 0.85*SSIM_loss + 0.15*(l_img + r_img) + 0.15*(left_grad_loss + right_grad_loss)
        # loss = 0.6*(l_img + r_img) + 0.6*(left_grad_loss + right_grad_loss)
    return loss


def unary_term_refined(left, right, recons_left, recons_right, left_diff, right_diff):
    with tf.variable_scope('unary_term'):
        left = left[:, :, THRESHOLD:, :]
        recons_left = recons_left[:, :, THRESHOLD:, :]
        right = right[:, :, :-THRESHOLD, :]
        recons_right = recons_right[:, :, :-THRESHOLD, :]

        left_mask = tf.less(left_diff, 3)
        left_mask = left_mask[:, :, THRESHOLD:, :]
        left_mask = left_mask[..., 0]
        right_mask = tf.less(right_diff, 3)
        right_mask = right_mask[:, :, :-THRESHOLD, :]
        right_mask = right_mask[..., 0]

        # SSIM
        with tf.name_scope('SSIM_loss'):
            SSIM_left = compute_SSIM_loss(recons_left, left, name='SSIM_left')
            SSIM_right = compute_SSIM_loss(recons_right, right, name='SSIM_right')
            SSIM_loss = SSIM_left + SSIM_right

        # reconstruction error of image
        with tf.name_scope('recon_img_loss'):
            l_img = tf.reduce_mean(tf.boolean_mask(tf.abs(left - recons_left), left_mask))
            r_img = tf.reduce_mean(tf.boolean_mask(tf.abs(right - recons_right), right_mask))

        # reconstruction error of gradient image
        with tf.name_scope('recon_grad_loss'):
            left_grad_x = gradient(left, axis='x')
            left_grad_x = tf.boolean_mask(left_grad_x, left_mask)
            left_grad_y = gradient(left, axis='y')
            left_grad_y = tf.boolean_mask(left_grad_y, left_mask)

            right_grad_x = gradient(right, axis='x')
            right_grad_x = tf.boolean_mask(right_grad_x, right_mask)
            right_grad_y = gradient(right, axis='y')
            right_grad_y = tf.boolean_mask(right_grad_y, right_mask)

            recons_left_grad_x = gradient(recons_left, axis='x')
            recons_left_grad_x = tf.boolean_mask(recons_left_grad_x, left_mask)
            recons_left_grad_y = gradient(recons_left, axis='y')
            recons_left_grad_y = tf.boolean_mask(recons_left_grad_y, left_mask)

            recons_right_grad_x = gradient(recons_right, axis='x')
            recons_right_grad_x = tf.boolean_mask(recons_right_grad_x, right_mask)
            recons_right_grad_y = gradient(recons_right, axis='y')
            recons_right_grad_y = tf.boolean_mask(recons_right_grad_y, right_mask)

            left_grad_loss = tf.reduce_mean(tf.abs(left_grad_x - recons_left_grad_x)) + tf.reduce_mean(tf.abs(left_grad_y - recons_left_grad_y))
            right_grad_loss = tf.reduce_mean(tf.abs(right_grad_x - recons_right_grad_x)) + tf.reduce_mean(tf.abs(right_grad_y - recons_right_grad_y))

        loss = 0.85*SSIM_loss + 0.15*(l_img + r_img) + 0.15*(left_grad_loss + right_grad_loss)
        # loss = 0.6*(l_img + r_img) + 0.6*(left_grad_loss + right_grad_loss)
    return loss


def gradient(inputs, axis):
    if axis == 'x':
        inputs = tf.pad(inputs, [[0, 0], [0, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        grad = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]
    elif axis == 'y':
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0], [0, 0]], 'SYMMETRIC')
        grad = inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
    else:
        raise ValueError('axis should be either x or y')
    return grad


def regularization_term(disp, img):
    with tf.variable_scope('regularization_term'):
        # second gradient of disp
        disp_gradient_x = tf.abs(disp[:, :, 2:, :] + disp[:, :, :-2, :] - 2*disp[:, :, 1:-1, :])
        disp_gradient_y = tf.abs(disp[:, 2:, :, :] + disp[:, :-2, :, :] - 2*disp[:, 1:-1, :, :])

        # second gradient of image
        img_gradient_x = tf.abs(img[:, :, 2:, :] + img[:, :, :-2, :] - 2*img[:, :, 1:-1, :])
        img_gradient_y = tf.abs(img[:, 2:, :, :] + img[:, :-2, :, :] - 2*img[:, 1:-1, :, :])

        # weights
        weights_x = tf.exp(-tf.reduce_mean(img_gradient_x, 3, keepdims=True))
        weights_y = tf.exp(-tf.reduce_mean(img_gradient_y, 3, keepdims=True))

        # compute smoothness
        smoothness_x = disp_gradient_x * weights_x
        smoothness_y = disp_gradient_y * weights_y
        loss = tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
    return loss


def smooth_term(disp, seg_embedding, diff):
    with tf.variable_scope('regularization_term'):
        # gradient of disp
        disp_gradient_x = tf.abs(gradient(disp, 'x'))
        disp_gradient_y = tf.abs(gradient(disp, 'y'))

        embedding_gradient_x = tf.abs(gradient(seg_embedding, 'x'))
        embedding_gradient_y = tf.abs(gradient(seg_embedding, 'y'))

        mean_x = tf.reduce_mean(embedding_gradient_x, 3, keepdims=True)
        mean_y = tf.reduce_mean(embedding_gradient_y, 3, keepdims=True)

        # weights
        diff = tf.clip_by_value(diff, 0, 3)
        weights_x = tf.exp(-(3-diff)) + tf.exp(-mean_x*5)
        weights_y = tf.exp(-(3-diff)) + tf.exp(-mean_y*5)

        # compute smoothness
        smoothness_x = disp_gradient_x * weights_x
        smoothness_y = disp_gradient_y * weights_y
        loss = tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
    return loss


def consistency_term(img, recons_img, is_left):
    with tf.variable_scope('consistency_term'):
        if is_left:
            loss = tf.reduce_mean(tf.abs(img[:, :, THRESHOLD:, :] - recons_img[:, :, THRESHOLD:, :]))
        else:
            loss = tf.reduce_mean(tf.abs(img[:, :, :-THRESHOLD, :] - recons_img[:, :, :-THRESHOLD, :]))
    return loss


def compute_SSIM_loss(x, y, name='SSIM_loss'):
    with tf.variable_scope(name):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = tf.layers.average_pooling2d(x, 3, 1, 'valid')
        mu_y = tf.layers.average_pooling2d(y, 3, 1, 'valid')

        sigma_x = tf.layers.average_pooling2d(x**2, 3, 1, 'valid') - mu_x**2
        sigma_y = tf.layers.average_pooling2d(y**2, 3, 1, 'valid') - mu_y**2
        sigma_xy = tf.layers.average_pooling2d(x*y, 3, 1, 'valid') - mu_x*mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d
        # SSIM = tf.clip_by_value((1 - SSIM) / 2, 0, 1)
        SSIM = (1 - SSIM) / 2
        loss = tf.reduce_mean(SSIM)
    return loss


def compute_loss(left, right, left_disp, right_disp, left_embedding, right_embedding, weights_list, name):
    with tf.variable_scope(name):
        flag = False
        if name == 'Initial_loss':
            flag = True

        # reconstruct input image, same size of original inputs
        with tf.name_scope('reconst_img'):
            recons_right = reconstruct_right(left, right_disp)
            recons_left = reconstruct_left(right, left_disp)
            re_recons_right = reconstruct_right(recons_left, right_disp)
            re_recons_left = reconstruct_left(recons_right, left_disp)

        with tf.name_scope('reconst_disp'):
            recons_right_disp = reconstruct_right(left_disp, right_disp)
            recons_left_disp = reconstruct_left(right_disp, left_disp)

        # compute loss
        with tf.name_scope('loss'):
            l_consis = consistency_term(left, re_recons_left, is_left=True) + consistency_term(right, re_recons_right, False)
            if flag:
                # initial loss
                l_unary = unary_term(left, right, recons_left, recons_right)
                l_reg = regularization_term(left_disp, left) + regularization_term(right_disp, right)
                loss = weights_list[0]*l_unary + weights_list[1]*l_reg + weights_list[2]*l_consis
            else:
                # refinement loss
                left_embedding = tf.image.resize_images(left_embedding, tf.shape(left_disp)[1:3])
                right_embedding = tf.image.resize_images(right_embedding, tf.shape(right_disp)[1:3])
                left_diff = tf.abs(left_disp - recons_left_disp)
                right_diff = tf.abs(right_disp - recons_right_disp)
                l_unary = unary_term(left, right, recons_left, recons_right)
                l_smooth = smooth_term(left_disp, left_embedding, left_diff) + smooth_term(right_disp, right_embedding, right_diff)
                loss = weights_list[3]*l_unary + weights_list[4]*l_smooth + weights_list[5]*l_consis

    return loss
