from network import *
from metrics import *
from loss import *


class GDNInpainting(object):
    def __init__(self, config):
        self.config = config
        self.img_size = config.INPUT_SIZE
        self.res_num = config.RES_NUM
        self.base_channel = config.BASE_CHANNEL
        self.sample_num = config.SAMPLE_NUM
        self.exp_base = config.EXPBASE
        self.gamma = config.GAMMA
        self.model_name = 'Inpainting'
        self.psnr = PSNR(255.0)
        self.gen_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )
        self.dis_optimizer = tf.train.AdamOptimizer(
            learning_rate=float(config.LR) * float(config.D2G_LR),
            beta1=float(config.BETA1),
            beta2=float(config.BETA2)
        )

    def build_whole_model(self, images, masks):
        # normalization [0, 255] to [0, 1]
        images = images / 255
        masks = masks / 255

        # masked
        images_masked = (images * (1 - masks)) + masks

        # inpaint
        outputs, gen_loss, dis_loss = self.inpaint_model(images, images_masked, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        # recover [0, 1] to [0, 255]
        images = images * 255
        images_masked = images_masked * 255
        outputs_merged = outputs_merged * 255
        outputs = outputs * 255

        # summary
        whole_image = tf.concat([images, images_masked, outputs, outputs_merged], axis=2)
        psnr = self.psnr(images, outputs_merged)
        tf.summary.image('train_image', whole_image, max_outputs=1)
        tf.summary.scalar('psnr', psnr)

        return gen_loss, dis_loss, psnr

    def build_validation_model(self, images, masks):
        # normalization [0, 255] to [0, 1]
        images = images / 255
        masks = masks / 255

        # masked
        images_masked = (images * (1 - masks)) + masks
        inputs = tf.concat([images_masked, masks], axis=3)

        outputs = self.inpaint_generator(inputs, self.res_num, self.base_channel, self.sample_num, reuse=True)
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        
        pred_masks, annotations, weight = self.discriminator(outputs, reuse=True)
        weight = tf.pow(tf.constant(float(self.exp_base)), weight)

        # mask hole ratio
        hr = tf.reduce_sum(masks, axis=[1, 2, 3]) / (self.img_size * self.img_size)

        # calculate validation loss
        gen_loss = 0
        dis_loss = 0
        with tf.variable_scope('validation_loss'):
            # discriminator loss
            dis_seg_loss = focal_loss(annotations, masks, hr, self.gamma)
            dis_loss += dis_seg_loss

            # generator l1 loss
            gen_l1_loss = l1_loss(weight * outputs, weight * images)
            gen_loss += gen_l1_loss
            gen_seg_loss = l1_loss(outputs, images)

        # recover [0, 1] to [0, 255]
        images = images * 255
        images_masked = images_masked * 255
        outputs_merged = outputs_merged * 255
        outputs = outputs * 255

        # summary
        whole_image = tf.concat([images, images_masked, outputs, outputs_merged], axis=2)
        tf.summary.image('validation_image', whole_image, max_outputs=1)
        psnr = self.psnr(images, outputs_merged)

        return gen_l1_loss, gen_seg_loss, dis_loss, psnr

    def build_optim(self, gen_loss, dis_loss):
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + '_generator')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + '_discriminator')
        g_gradient = self.gen_optimizer.compute_gradients(gen_loss, var_list=g_vars)
        d_gradient = self.dis_optimizer.compute_gradients(dis_loss, var_list=d_vars)

        return self.gen_optimizer.apply_gradients(g_gradient), self.dis_optimizer.apply_gradients(d_gradient)

    def inpaint_model(self, images, images_masked, masks):
        # input model
        inputs = tf.concat([images_masked, masks], axis=3)

        # process outputs
        output = self.inpaint_generator(inputs, self.res_num, self.base_channel, self.sample_num)
        outputs_merged = (output * masks) + (images * (1 - masks))
        gen_loss = 0
        dis_loss = 0

        # create discriminator
        prediction, annotations, weight = self.discriminator(output)
        weight = tf.pow(tf.constant(float(self.exp_base)), weight)

        # mask hole ratio
        hr = tf.reduce_sum(masks, axis=[1, 2, 3]) / (self.img_size * self.img_size)

        with tf.variable_scope('inpaint_loss'):
            # discriminator loss
            dis_seg_loss = focal_loss(annotations, masks, hr, self.gamma)
            dis_loss += dis_seg_loss

            # generator l1 loss
            gen_weighted_loss = l1_loss(weight * output, weight * images)
            gen_loss += gen_weighted_loss
            gen_l1_loss = l1_loss(output, images)

        # summary all of loss
        tf.summary.scalar('loss/dis_loss', dis_loss)
        tf.summary.scalar('loss/gen_l1_loss', gen_l1_loss)
        tf.summary.scalar('loss/gen_weighted_loss', gen_weighted_loss)

        return output, gen_loss, dis_loss

    def inpaint_generator(self, x, residual_num, channel, sample, reuse=False):
        with tf.variable_scope('inpaint_generator', reuse=reuse):
            with tf.variable_scope('encoder_1'):
                x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, scope='conv')
                x = instance_norm(x, scope='ins_norm')
                x = relu(x)

            # Down-Sampling
            for i in range(2, sample + 2):
                with tf.variable_scope('encoder_downsample_' + str(i)):
                    channel = channel * 2
                    x = conv(x, channel, kernel=4, stride=2, pad=1, use_bias=False, scope='conv_1')
                    x = instance_norm(x, scope='ins_norm_1')
                    x = relu(x)

            # Bottleneck
            for i in range(1, residual_num + 1):
                x = resblock(x, channel, rate=2, use_bias=False, scope='resblock_' + str(i))

            # Up-Sampling
            for i in range(0, sample):
                with tf.variable_scope('decoder_upsample_' + str(i + 1)):
                    channel = channel // 2
                    x = deconv(x, channel, kernel=4, stride=2, use_bias=False, scope='deconv')
                    x = instance_norm(x, scope='ins_norm_1')
                    x = relu(x)

            with tf.variable_scope('decoder' + str(sample + 1)):
                x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', use_bias=False, scope='conv')

            x = (tf.nn.tanh(x) + 1) / 2

            return x

    def discriminator(self, x, layer=2, reuse=False):
        with tf.variable_scope('inpaint_discriminator', reuse=reuse):
            conv1 = tf.layers.conv2d(x, 32, kernel_size=4, strides=1, padding='SAME', name='conv1')
            conv1 = tf.nn.leaky_relu(conv1, alpha=0.2)
            conv2 = tf.layers.conv2d(conv1, 64, kernel_size=4, strides=1, padding='SAME', name='conv2')
            conv2 = tf.nn.leaky_relu(conv2, alpha=0.2)
            conv3 = tf.layers.conv2d(conv2, 128, kernel_size=4, strides=2, padding='SAME', name='conv3')
            conv3 = tf.nn.leaky_relu(conv3, alpha=0.2)
            conv4 = tf.layers.conv2d(conv3, 256, kernel_size=4, strides=2, padding='SAME', name='conv4')
            conv4 = tf.nn.leaky_relu(conv4, alpha=0.2)
            conv5 = tf.layers.conv2d(conv4, 256, kernel_size=4, strides=1, padding='SAME', name='conv5')
            conv5 = tf.nn.leaky_relu(conv5, alpha=0.2)
            x = deconv(conv5, 128, kernel=4, stride=2, use_bias=True, scope='deconv_1')
            x = deconv(x, layer, kernel=4, stride=2, use_bias=True, scope='deconv_2')
            output = tf.cast(tf.argmax(x, dimension=3, name="prediction"), dtype=tf.float32)
            map = tf.nn.softmax(x, axis=-1)
            output = tf.concat([output, map[:, :, :, 1]], axis=2)
            output = tf.expand_dims(output, dim=-1)
        return output, x, tf.expand_dims(map[:, :, :, 1], dim=-1)
