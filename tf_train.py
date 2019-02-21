from load_data import get_filenames
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cnn_model
data_dir = "./raw_data"
train_ratio = 0.8
max_epoch = 5000


def get_batch(filenames, labels, batch_size, capacity):
    tf_filenames = tf.cast(filenames, tf.string)
    tf_labels = tf.cast(labels, tf.int32)
    input_queue = tf.train.slice_input_producer([tf_filenames, tf_labels])
    img = tf.read_file(input_queue[0])
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, [64, 64])
    # img = tf.image.per_image_standardization(img)
    image_batch, label_batch = tf.train.batch([img, input_queue[1]], batch_size=batch_size, num_threads=32)
    return image_batch, label_batch


if __name__ == "__main__":
    train_filenames, train_label, test_filenames, test_label = get_filenames(data_dir, train_ratio)
    class_num = len(set(train_label + test_label))
    train_batch, label_batch = get_batch(train_filenames, train_label, 20, 64)

    model = cnn_model.CNN_Model(class_num)
    tf_logits = model.predict(train_batch)
    tf_loss = model.loss(tf_logits, label_batch)
    tf_train_op = model.optimizer(tf_loss)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(tf.global_variables_initializer())
        for i in range(max_epoch):
            _, loss, _ = sess.run([tf_logits, tf_loss, tf_train_op])

            if i%100 == 0:
                print("损失为: " + str(loss))

