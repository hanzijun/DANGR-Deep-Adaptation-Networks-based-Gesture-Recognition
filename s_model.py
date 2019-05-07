"""
The model trained on source dataset which we refer to source_model.
Created on March 6, 2019
Author: zijun han
"""
import tensorflow as tf
import numpy as np
import random
import pickle
from sklearn.manifold import TSNE
# import normalization
import matplotlib.pyplot as plt
# import pylab
np.set_printoptions(threshold=np.inf)


with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_th.pkl', 'rb') as f:
    sh = pickle.load(f)
with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_tq.pkl', 'rb') as f:
    sq = pickle.load(f)
with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_tqie2.pkl', 'rb') as f:
    sqie = pickle.load(f)
with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_tz.pkl', 'rb') as f:
    sz = pickle.load(f)

with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_th.pkl', 'rb') as f:
    gh = pickle.load(f)
with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_tq.pkl', 'rb') as f:
    gq = pickle.load(f)
with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_tqie.pkl', 'rb') as f:
    gqie = pickle.load(f)
with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_tz.pkl', 'rb') as f:
    gz = pickle.load(f)
with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_t2.pkl', 'rb') as f:
    g = pickle.load(f)
train_data_real = g
# train_data_real = gh + gq + gqie + gz
# train_data_real = sh[0:200] + sq[0:200] + sqie[0:200] + sz[0:200]
test_ground_truth = sh[200:300] + sq[200:300] + sqie[200:300] + sz[200:300]

# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_th.pkl', 'rb') as f:
#     th1 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_tq.pkl', 'rb') as f:
#     tq1 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_tqie.pkl', 'rb') as f:
#     tqie1 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_tz.pkl', 'rb') as f:
#     tz1 = pickle.load(f)
# test_ground_truth = th1+tq1+tqie1+tz1

# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_th.pkl', 'rb') as f:
#     th2 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_tq.pkl', 'rb') as f:
#     tq2 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_tqie.pkl', 'rb') as f:
#     tqie2 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_tz.pkl', 'rb') as f:
#     tz2 = pickle.load(f)
# test_ground_truth = th2+tq2+tqie2+tz2

# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_th.pkl', 'rb') as f:
#     th3 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_tq.pkl', 'rb') as f:
#     tq3 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_tqie.pkl', 'rb') as f:
#     tqie3 = pickle.load(f)
# with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_tz.pkl', 'rb') as f:
#     tz3 = pickle.load(f)
# test_ground_truth  = th3+tq3+tqie3+tz3

class SourceModel():
    def __init__(
            self,
            m=28,
            n=100,
            k=4,
            batch_size=256,
            learning_rate=0.0005,
            training_epochs=1,
            param_file=False,
            is_train=False
                 ):
        self.m, self.n, self.k = m, n, k
        self.batch_size = batch_size
        self.lr = learning_rate
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.buildNetwork()
        print "Neural networks build!"
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
            if param_file is True:
                self.saver.restore(self.sess, "./GAN_train/GAN-train-fake3.ckpt")
                print("loading neural-network params...")
                self.learn()
                self.show()
            else:
                print "learning with initialization!"
                self.learn()
                self.show()
        else:
            self.saver.restore(self.sess, "./GAN_train/GAN-train-fake3.ckpt")
            print "loading neural-network params..."
            self.show()

    def buildNetwork(self):

            self.x = tf.placeholder(tf.float32, shape = [None, self.m, self.n, 1], name='image_origin')
            self.y = tf.placeholder(tf.float32, shape = [None, self.k], name='true_label')
            self.y_ = tf.placeholder(tf.float32, shape = [None, self.k], name='predict')

            with tf.variable_scope('sharedModel'):
                w_initializer = tf.random_normal_initializer(stddev=0.02)
                b_initializer = tf.constant_initializer(0.01)

                w_e_conv1 = tf.get_variable('w1', [5, 5, 1, 32], initializer=w_initializer)
                b_e_conv1 = tf.get_variable('b1', [32, ], initializer=b_initializer)
                con1_s = lrelu(tf.add(self.conv2d(self.x, w_e_conv1), b_e_conv1))
                self.fly = tf.reshape(con1_s, (-1, 14 * 50 * 32))
                w_e_conv2 = tf.get_variable('w2', [5, 5, 32, 64], initializer=w_initializer)
                b_e_conv2= tf.get_variable('b2', [64, ], initializer=b_initializer)
                con2_s = lrelu(tf.add(self.conv2d(con1_s, w_e_conv2), b_e_conv2))

                w_e_conv3 = tf.get_variable('w3', [5, 5, 64, 128], initializer=w_initializer)
                b_e_conv3= tf.get_variable('b3', [128, ], initializer=b_initializer)
                con3_s = lrelu(tf.add(self.conv2d(con2_s, w_e_conv3), b_e_conv3))

                # layer1 = conv2d(self.x, 32, name='d_h1_conv2d')
                # layer1 = lrelu(layer1)
                # layer2 = conv2d(layer1, 64, name='d_h2_conv2d')
                # layer2 = lrelu(layer2)
                # layer3 = conv2d(layer2, 128, name='d_h3_conv2d')
                # layer3 = lrelu(layer3)
                # layer1 = lrelu(tf.layers.batch_normalization(layer1, training=self.is_train, name='g_h1_batch_norm'))
                # layer2 = lrelu(tf.layers.batch_normalization(layer2, training=self.is_train, name='g_h2_batch_norm'))
                # layer3 = lrelu(tf.layers.batch_normalization(layer3, training=self.is_train, name='g_h3_batch_norm'))

            with tf.variable_scope('fineTuning'):
                w_e_conv4 = tf.get_variable('w4', [5, 5, 128, 256], initializer=w_initializer)
                b_e_conv4= tf.get_variable('b4', [256, ], initializer=b_initializer)
                con4_s = lrelu(tf.add(self.conv2d(con3_s, w_e_conv4), b_e_conv4))
                # con4_s = tf.add(self.conv2d(con3_s, w_e_conv4), b_e_conv4)
                # layer4 = conv2d(layer3, 256, name='d_h4_conv2d')
                # layer4 = lrelu(tf.layers.batch_normalization(layer4, training=self.is_train, name='g_h4_batch_norm'))
                # layer4 = lrelu(layer4)
                # layer5 = conv2d(layer4, 256, name='d_h5_conv2d')
                # layer5 = lrelu(tf.layers.batch_normalization(layer5, training=self.is_train, name='g_h5_batch_norm'))
                # layer5 = lrelu(layer5)
                # layer2 = tf.layers.conv2d(layer1, 64, 3, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
                # layer2 = tf.nn.relu(layer2)
                # layer3 = tf.layers.conv2d(layer2, 128, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
                # layer3 = tf.nn.relu(layer3)
                # # layer3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='VALID')
                # layer4 = tf.layers.conv2d(layer3, 256, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
                # layer4 = tf.nn.relu(layer4)
                # layer5 = tf.layers.conv2d(layer4, 256, 3, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(seed=2))
                # layer5 = tf.nn.relu(layer5)
            with tf.variable_scope('tailed'):
                self.flatten = tf.reshape(con4_s, (-1,  2*7*256))
                w_fc1 = tf.get_variable('w5', [2*7*256, 500], initializer=w_initializer)
                b_fc1 = tf.get_variable('b5', [500], initializer=b_initializer)
                self.logits = tf.nn.sigmoid(tf.matmul(self.flatten, w_fc1) + b_fc1)

                w_fc2 = tf.get_variable('w6', [500, self.k], initializer=w_initializer)
                b_fc2 = tf.get_variable('b6', [self.k], initializer=b_initializer)
                res = tf.nn.sigmoid(tf.matmul(self.logits, w_fc2) + b_fc2)

                self.y_ = res

            with tf.variable_scope('loss'):

                self.loss = tf.reduce_mean(tf.pow(self.y - self.y_, 2))

            with tf.variable_scope('train'):

                self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            """
            This is the origin model setup for mnist and M-mnist
            """
            # with tf.variable_scope('SharedModel'):
            #     w_initializer = tf.random_normal_initializer(stddev=0.01)
            #     b_initializer = tf.constant_initializer(0.01)
            #     self.w_e_conv1 = tf.get_variable('w1', [3, 3, 1, 16], initializer=w_initializer)
            #     b_e_conv1 = tf.get_variable('b1', [16, ], initializer=b_initializer)
            #     self.conv1 = tf.nn.relu(tf.add(self.conv2d(self.input, self.w_e_conv1), b_e_conv1))
            #     print self.conv1.shape
            #     self.w_e_conv2 = tf.get_variable('w2', [3, 3, 16, 64], initializer=w_initializer)
            #     b_e_conv2 = tf.get_variable('b2', [64, ], initializer=b_initializer)
            #     self.conv2 = tf.nn.relu(tf.add(self.conv2d(self.conv1, self.w_e_conv2), b_e_conv2))
            #     print self.conv2.shape
            #     self.w_e_conv3 = tf.get_variable('w3', [3, 3, 64, 128], initializer=w_initializer)
            #     b_e_conv3 = tf.get_variable('b3', [128, ], initializer=b_initializer)
            #     conv3 = tf.nn.relu(tf.add(self.conv2d(self.conv2, self.w_e_conv3), b_e_conv3))
            #     print conv3.shape
            #     self.conv3 = tf.reshape(conv3, [-1, 4 * 4 * 128])
            # with tf.variable_scope('tailed'):
            #     self.w2_fc = tf.get_variable('tailed_w4', [4 * 4 * 128, GESTURE], initializer=w_initializer, )
            #     self.b2_fc = tf.get_variable('tailed_b4', [GESTURE, ], initializer=b_initializer,)
            #     result = tf.nn.relu(tf.matmul(self.conv3, self.w2_fc) + self.b2_fc)
            #     self.predict = result

    def conv2d(self,  x, W):
        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

    def learn(self):

            total_batch = 300
            for epoch in range(self.training_epochs):
                # Loop over all batches
                for i in range(total_batch):
                    # batch = np.array(random.sample(csi_train, self.batch_size))
                    random.shuffle(train_data_real)
                    batchs = [
                        train_data_real[k: k + self.batch_size]
                        for k in xrange(0, 8000, self.batch_size)]

                    for batch in batchs:
                        data, label = self.transition(np.array(batch))
                        batch_xs = np.reshape(data, (-1, self.m, self.n, 1))
                        batch_ys = label

                        _, c = self.sess.run([self.optimizer, self.loss], feed_dict={self.x: batch_xs, self.y: batch_ys})

                    if i % 10 == 0:
                        print("Epoch:", '%04d' % (epoch + 1),"iteration: %d"%(i), "cost=", "{:.8f}".format(c))
                    if i % 10 == 0:
                        self.show()

            print("Optimization Finished!")
            # self.saver.save(self.sess, "./GAN_train/GAN-train-fake3.ckpt")

    def show(self):

        data, lab = self.transition(np.array(test_ground_truth), is_train=False)
        y_pre = self.sess.run(
                self.y_, feed_dict={self.x: np.reshape(data, (-1, self.m, self.n, 1))})
        y_lab = lab
        y_pre = [np.argmax(i) for i in y_pre]
        # print y_pre
        # print y_lab
        accuarcy = sum(int(y == y_) for (y, y_) in zip(y_pre, y_lab))
        print "accuarcy: {0} / {1} ".format(accuarcy,400)

        batch_xs = data.reshape((-1, 28,100, 1))
        batch_ys = lab
        features = self.sess.run(self.fly, feed_dict={self.x: batch_xs})
        src_features = TSNE(n_components=2).fit_transform(features)
        slide,riot,down,push=[],[],[],[]
        for i in range(len(batch_ys)):
            if batch_ys[i]==0:
                slide.append(np.array(src_features[i]))
            elif batch_ys[i] == 1:
                riot.append(np.array(src_features[i]))
            elif batch_ys[i] == 2:
                down.append(np.array(src_features[i]))
            else:
                push.append(np.array(src_features[i]))
        slide = np.array(slide)
        riot = np.array(riot)
        down = np.array(down)
        push = np.array(push)
        o1 = plt.scatter(slide[:, 0], slide[:, 1], c='', edgecolors='#FF6347', alpha=0.8,label = 'slide')
        o2 = plt.scatter(riot[:, 0], riot[:, 1], c='', edgecolors='y', alpha=0.8,label = 'riot')
        o3 = plt.scatter(down[:, 0], down[:, 1], c='', edgecolors='g', alpha=0.8,label='down')
        o4 = plt.scatter(push[:, 0], push[:, 1], c='', edgecolors='b', alpha=0.8,label= 'push')
        plt.legend(loc='best')
        plt.title('Source domain')
        # plt.xlim((-30, 40))
        # plt.ylim((-40, 30))
        plt.savefig('/home/han/plot123_2.png', dpi=900)
        plt.show()

        # x= []
        # for i in batch_ys:
        #     label = i
        #     if label == 0:
        #         x.append('#FF6347')
        #     elif label == 1:
        #         x.append('y')
        #     elif label == 2:
        #         x.append('g')
        #     elif label == 3:
        #         x.append('b')
        #     elif label == 4:
        #         x.append('#DC143C')
        #     elif label == 5:
        #         x.append('k')
        #     elif label == 6:
        #         x.append('m')
        #     elif label == 7:
        #         x.append('#00FFFF')
        #     elif label == 8:
        #         x.append('#20B2AA')
        #     elif label == 9:
        #         x.append('#708090')
        #
        # plt.scatter(src_features[:, 0], src_features[:, 1], c='', edgecolors=x, alpha=0.8)
        # plt.legend(loc='upper left')
        # plt.title('target2')
        # plt.show()

    def transition(self, batch, is_train = True):
        data = None
        for i in batch[:,0]:
            i = i[:, 0:100]
            data = np.array([i]) if data is None else np.append(data, [i], axis=0)

        if is_train is True:
            label = None
            for j in batch[:,1]:
                label = np.array([self.convert(j[0])]) if label is None else np.append(label, [self.convert(j[0])], axis=0)
            label = np.reshape(label, (-1, 4))

        else:
            label = []
            for j in batch[:,1]:
                label.append(j[0])

        return data, label

    def convert(self, number):
        e = np.zeros((4,1))
        e[number] = 1
        return e

    def test(self):
        """
        This is the function for performance view.
        """

        correct_prediction = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.predict, 1))
        batch_xs = np.reshape(mnist.test.images, [-1, HEIGHT, WEIGHT, DEPTH])
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print(self.sess.run(accuracy, feed_dict={self.input: batch_xs, self.label: mnist.test.labels}))

        batch_xs = batch_xs[1000: 3000]
        batch_ys = mnist.test.labels[1000: 3000]
        features = self.sess.run(self.conv3, feed_dict={self.input: batch_xs})
        src_features = TSNE(n_components=2).fit_transform(features)
        x= []
        for i in batch_ys:
            label = self.changeLabel(i)
            if label == 0:
                x.append('#FF6347')
            elif label == 1:
                x.append('y')
            elif label == 2:
                x.append('g')
            elif label == 3:
                x.append('b')
            elif label == 4:
                x.append('#DC143C')
            elif label == 5:
                x.append('k')
            elif label == 6:
                x.append('m')
            elif label == 7:
                x.append('#00FFFF')
            elif label == 8:
                x.append('#20B2AA')
            elif label == 9:
                x.append('#708090')

        plt.scatter(src_features[:, 0], src_features[:, 1], c='', edgecolors=x, alpha=0.8)
        plt.title('Non-adapted')
        pylab.show()

    def changeLabel(self, one_hot):
        return np.argmax(one_hot)

def weight_variable(shape, name, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return var

def bias_variable(shape, name, bias_start=0.01, trainable = True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return var

def conv2d(x, output_channels, name, k_h=5, k_w=5,reuse =False):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        w = weight_variable(shape=[k_h, k_w, x_shape[-1], output_channels], name='weights')
        b = bias_variable([output_channels], name='biases')
        conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME') + b
        return conv

def lrelu(x, leak=0.02):
    return tf.maximum(x, leak * x)

if __name__ =="__main__":
    SourceModel()

