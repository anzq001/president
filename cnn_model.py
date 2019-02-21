import tensorflow as tf


class CNN_Model:
    def __init__(self, class_num):
        # shape = [filter_height, filter_width, in_channels, out_channels]
        self.scope1_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1, dtype=tf.float32),
                                          name="scop1_weights", dtype=tf.float32)
        self.scope1_bias = tf.Variable(tf.constant(shape=[64], value=1, dtype=tf.float32),
                                       name="scope1_bias", dtype=tf.float32)
        self.scope2_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                                          name="scope1", dtype=tf.float32)
        self.scope2_bias = tf.Variable(tf.truncated_normal(shape=[16], stddev=1, dtype=tf.float32),
                                       name="scope2_bias", dtype=tf.float32)
        self.class_num = class_num

    def predict(self, images):
        # 卷积层一
        with tf.variable_scope("cov1") as scope:
            # image_shape = [batch, in_height, in_width, in_channels]
            conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.scope1_weights, strides=[1, 1, 1, 1], padding="SAME"),
                                  self.scope1_bias)
            conv1_activate = tf.nn.relu(conv1, name=scope.name)

        # 池化层一
        with tf.variable_scope("pool1") as scope:
            pool1 = tf.nn.max_pool(conv1_activate, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
            # lrn是normalization的一种，目的是为了抑制神经元的输出，LRN的设计借鉴了神经生物学的一个概念叫做侧抑制
            norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=1, alpha=0.001, beta=0.5, name=scope.name)

        # 卷积层二
        with tf.variable_scope("conv2") as scope:
            conv2 = tf.nn.bias_add(tf.nn.conv2d(norm1, self.scope2_weights, strides=[1, 1, 1, 1], padding="SAME"),
                                   bias=self.scope2_bias)
            conv2_activate = tf.nn.relu(conv2, name=scope.name)

        # 池化层二
        with tf.variable_scope("pool2") as scope:
            # 先抑制，再池化
            norm2 = tf.nn.lrn(conv2_activate, depth_radius=4, bias=1, alpha=0.001, beta=0.5, name="norm2")
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name=scope.name)

        # 卷积层三：全连接层
        with tf.variable_scope("full_connection") as scope:
            image_batch = tf.shape(images)[0]
            full_input = tf.reshape(pool2, shape=[image_batch, -1])
            dim = full_input.get_shape()[1].value
            full_weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], mean=0, stddev=0.1, dtype=tf.float32),
                                       name="full_weights", dtype=tf.float32)
            full_bias = tf.Variable(tf.constant(value=0, shape=[128], dtype=tf.float32),
                                    name="full_bias", dtype=tf.float32)
            full_output = tf.nn.bias_add(tf.matmul(full_input, full_weights), full_bias)
            full_relu = tf.nn.relu(full_output)

        # 卷积层四：判断层
        with tf.variable_scope("judge") as scope:
            judge_weights = tf.Variable(tf.truncated_normal(shape=[128, self.class_num], mean=0, stddev=0.1, dtype=tf.float32),
                                        name="judge_weight", dtype=tf.float32)
            judge_bias = tf.Variable(tf.constant(value=0, shape=[self.class_num], dtype=tf.float32),
                                     name="judge_bias", dtype=tf.float32)
            linear = tf.add(tf.matmul(full_relu, judge_weights), judge_bias)

        return linear

    def loss(self, logits, labels):
        onehot_labels = tf.one_hot(labels, self.class_num)
        # 计算损失
        with tf.variable_scope("loss") as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
            loss = tf.reduce_mean(cross_entropy, name=scope.name)

        return loss

    def optimizer(self, loss):
        # 优化函数
        with tf.variable_scope("optimizer") as scope:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

