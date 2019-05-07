"""
The model re-trained on target dataset which we refer to target_model.
Created on March 6, 2019
Author: zijun han
"""
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import mmd
import torch.utils.data as Data

class TargetModel():
    def __init__(
            self,
            m=28,
            n=100,
            k=4,
            batch_size=100,
            learning_rate=0.001,
            training_epochs=50 ,
            is_train=False,
                 ):
        self.m, self.n, self.k = m, n, k
        self.batch_size = batch_size
        self.lr = learning_rate
        self.is_train = is_train
        self.training_epochs = training_epochs
        self.buildNetwork()
        print "Neural networks built!"

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if is_train is True:
                self.saver.restore(self.sess, './GAN_train/GAN-train-fake3.ckpt') # load source model
                print("loading source model params...")
                var_list = [self.w_fc1, self.b_fc1,self.w_fc2,self.b_fc2]
                initfc = tf.variables_initializer(var_list, name='init')  # FIXME: Any better initialization?
                self.sess.run(initfc)                                                                 # initialize the tailed layers
                self.learn()                                                                                 # time to train
                self.show()
        else:
            self.saver.restore(self.sess, "./params/target3.ckpt")    # load trained model
            self.show()                                                                                 # performance view

    def buildNetwork(self):
            """
            Here, the 'SharedModel'  is the transfered layer composed of CNN, the 'tailer' is the fully connected layer which is randomly initialized.
            source data with label is used to calculate class error, and compute mmd loss together with target data.
            :return: initialized model.
            """
            self.source_input = tf.placeholder(tf.float32, shape = [None, self.m, self.n, 1], name='source_input')
            self.target_input= tf.placeholder(tf.float32, shape = [None, self.m, self.n, 1], name='target_input')
            self.predict = tf.placeholder(tf.float32, shape = [None, self.k], name ='y_predict')
            self.source_label= tf.placeholder(tf.float32, shape = [None,  self.k], name='y_label')

            with tf.variable_scope('sharedModel'):
                w_initializer = tf.random_normal_initializer(stddev=0.02)
                b_initializer = tf.constant_initializer(0.01)

                #     self.w_e_conv1 = tf.get_variable('w1', [3, 3, 1, 16], initializer=w_initializer, trainable=False)
                #     b_e_conv1 = tf.get_variable('b1', [16, ], initializer=b_initializer,  trainable=False)
                #     self.conv1_source = tf.nn.relu(tf.add(self.conv2d(self.source_input, self.w_e_conv1), b_e_conv1))

                w_e_conv1 = tf.get_variable('w1', [5, 5, 1, 32], initializer=w_initializer, trainable=False)
                b_e_conv1 = tf.get_variable('b1', [32, ], initializer=b_initializer, trainable=False)
                con1_s = lrelu(tf.add(self.conv2d(self.source_input, w_e_conv1), b_e_conv1))
                con1_t = lrelu(tf.add(self.conv2d(self.target_input, w_e_conv1), b_e_conv1))
                # self.fly = tf.reshape(con1_t, (-1, 14 * 50 * 32))

                w_e_conv2 = tf.get_variable('w2', [5, 5, 32, 64], initializer=w_initializer, trainable=False)
                b_e_conv2= tf.get_variable('b2', [64, ], initializer=b_initializer, trainable=False)
                con2_s = lrelu(tf.add(self.conv2d(con1_s, w_e_conv2), b_e_conv2))
                con2_t = lrelu(tf.add(self.conv2d(con1_t, w_e_conv2), b_e_conv2))
                self.fly = tf.reshape(con2_t, (-1, 7 * 25 * 64))
                w_e_conv3 = tf.get_variable('w3', [5, 5, 64, 128], initializer=w_initializer, trainable=False)
                b_e_conv3= tf.get_variable('b3', [128, ], initializer=b_initializer, trainable=False)
                con3_s = lrelu(tf.add(self.conv2d(con2_s, w_e_conv3), b_e_conv3))
                con3_t = lrelu(tf.add(self.conv2d(con2_t, w_e_conv3), b_e_conv3))
                self.con3_s = tf.reshape(con3_s, [-1, 4*13*128])
                self.con3_t = tf.reshape(con3_t, [-1, 4*13*128])
                # con1_s = conv2d(self.source_input, 32, name='d_h1_conv2d')
                # con1_s = lrelu(con1_s)
                # con1_t = conv2d(self.target_input, 32, name='d_h1_conv2d', reuse=True)
                # con1_t = lrelu(con1_t)
                # con2_s = conv2d(con1_s, 64, name='d_h2_conv2d')
                # con2_s = lrelu(con2_s)
                # con2_t = conv2d(con1_t, 64, name='d_h2_conv2d', reuse=True)
                # con2_t = lrelu(con2_t)
                # con3_s = conv2d(con2_s, 128, name='d_h3_conv2d')
                # con3_s = lrelu(con3_s)
                # con3_t = conv2d(con2_t, 128, name='d_h3_conv2d', reuse=True)
                # con3_t = lrelu(con3_t)

            with tf.variable_scope('fineTuning'):
                w_e_conv4 = tf.get_variable('w4', [5, 5, 128, 256], initializer=w_initializer)
                b_e_conv4= tf.get_variable('b4', [256, ], initializer=b_initializer)
                con4_s = lrelu(tf.add(self.conv2d(con3_s, w_e_conv4), b_e_conv4))
                con4_t = lrelu(tf.add(self.conv2d(con3_t, w_e_conv4), b_e_conv4))

            with tf.variable_scope('tailed'):
                self.con4_s= tf.reshape(con4_s, [-1, 2*7*256])
                self.con4_t = tf.reshape(con4_t, [-1, 2*7*256])

                self.w_fc1 = tf.get_variable('w5', [2*7*256, 500], initializer=w_initializer)
                self.b_fc1 = tf.get_variable('b5', [500,], initializer=b_initializer)
                self.res = tf.nn.sigmoid(tf.matmul(self.con4_s, self.w_fc1) + self.b_fc1)
                self.ret = tf.nn.sigmoid(tf.matmul(self.con4_t, self.w_fc1) + self.b_fc1)

                self.w_fc2 = tf.get_variable('w6', [500, self.k], initializer=w_initializer)
                self.b_fc2 = tf.get_variable('b6', [self.k,], initializer=b_initializer)
                pre_s = tf.nn.sigmoid(tf.matmul(self.res, self.w_fc2) + self.b_fc2)
                pre_t = tf.nn.sigmoid(tf.matmul(self.ret, self.w_fc2) + self.b_fc2)

                self.p_s = pre_s
                self.p_t = pre_t

            with tf.variable_scope('loss'):
                alpha, theta, gama1, gama2 = 0.3, 0.3, 0.3, 0.3

                self.class_loss = tf.reduce_mean(tf.pow(self.source_label - self.p_s, 2))
                self.mmd_loss =  mmd.mmd_rbf_accelerate(source=self.con3_s, target=self.con3_t) * alpha + mmd.mmd_rbf_accelerate(source=self.con4_s, target=self.con4_t)* theta + \
                                 mmd.mmd_rbf_accelerate(source=self.res, target=self.ret) * gama1 + mmd.mmd_rbf_accelerate(source=self.p_s, target=self.p_t) * gama2
                self.loss = self.class_loss + self.mmd_loss

            with tf.variable_scope('train'):

                train_vars = tf.trainable_variables()
                fine_vars = [var for var in train_vars if var.name.startswith("fineTuning")]
                tail_vars = [var for var in train_vars if var.name.startswith("tailed")]

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    # train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, var_list=fine_vars)
                    # train_op2 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, var_list=tail_vars)
                    opt1 = tf.train.GradientDescentOptimizer(0.0001)
                    opt2 = tf.train.GradientDescentOptimizer(0.005)

                    grads = tf.gradients(self.loss, fine_vars + tail_vars)
                    grads1 = grads[:len(fine_vars)]
                    grads2 = grads[len(fine_vars):]
                    train_op1 = opt1.apply_gradients(zip(grads1,fine_vars))
                    train_op2 = opt2.apply_gradients(zip(grads2, tail_vars))
                    self.train_op = tf.group(train_op1, train_op2)

                # self.train_op = tf.group(fine_opt, tail_opt)

                # fine_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fineTuning')
                # fine_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss, var_list=fine_vars)
                #
                # tail_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='tailed')
                # tail_opt = tf.train.AdamOptimizer(learning_rate=0.005).minimize(self.loss,var_list=tail_vars)

            # with tf.variable_scope('SharedModel'):
            #     w_initializer = tf.random_normal_initializer(stddev=0.01)
            #     b_initializer = tf.constant_initializer(0.01)
            #     self.w_e_conv1 = tf.get_variable('w1', [3, 3, 1, 16], initializer=w_initializer, trainable=False)
            #     b_e_conv1 = tf.get_variable('b1', [16, ], initializer=b_initializer,  trainable=False)
            #     self.conv1_source = tf.nn.relu(tf.add(self.conv2d(self.source_input, self.w_e_conv1), b_e_conv1))
            #     self.conv1_target =  tf.nn.relu(tf.add(self.conv2d(self.target_input, self.w_e_conv1), b_e_conv1))
            #     self.w_e_conv2 = tf.get_variable('w2', [3, 3, 16, 64], initializer=w_initializer,  trainable=False)
            #     b_e_conv2 = tf.get_variable('b2', [64, ], initializer=b_initializer,  trainable=False)
            #     self.conv2_source = tf.nn.relu(tf.add(self.conv2d(self.conv1_source, self.w_e_conv2), b_e_conv2))
            #     self.conv2_target = tf.nn.relu(tf.add(self.conv2d(self.conv1_target, self.w_e_conv2), b_e_conv2))
            #     self.w_e_conv3 = tf.get_variable('w3', [3, 3, 64, 128], initializer=w_initializer,  trainable=False)
            #     self.b_e_conv3 = tf.get_variable('b3', [128, ], initializer=b_initializer, trainable=False)
            #     conv3_source = tf.nn.relu(tf.add(self.conv2d(self.conv2_source, self.w_e_conv3), self.b_e_conv3))
            #     conv3_target = tf.nn.relu(tf.add(self.conv2d(self.conv2_target, self.w_e_conv3), self.b_e_conv3))
            #     self.conv3_source = tf.reshape(conv3_source, [-1, 4 * 4 * 128])
            #     self.conv3_target = tf.reshape(conv3_target, [-1, 4 * 4 * 128])
            # with tf.variable_scope('tailed'):
            #     self.w2_fc = tf.get_variable('tailed_w4', [4 * 4 * 128, GESTURE], initializer=w_initializer,)
            #     self.b2_fc = tf.get_variable('tailed_b4', [GESTURE, ], initializer=b_initializer,)
            #     self.result_s = tf.nn.relu(tf.matmul(self.conv3_source, self.w2_fc) + self.b2_fc)
            #     self.result_t = tf.nn.relu(tf.matmul(self.conv3_target, self.w2_fc) + self.b2_fc)
            #     self.p_s = self.result_s
            #     self.p_t = self.result_t
    # def conv2d(self,  x, W):
    #     return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')
    def conv2d(self,  x, W):

        return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

    def learn(self):
        """
        The transferring and re-training process
        :return: target model
        """
        for j in range(self.training_epochs):
            source_iter = iter(source_data)
            target_iter = iter(target_data)
            for i in range(1, len_source+1):

                batch_s = source_iter.next()
                xs, ys = self.transition(batch_s)
                xs = np.reshape(xs, [-1, self.m, self.n, 1])
                ys = np.reshape(ys, [-1, self.k])

                batch_t = target_iter.next()
                xt, _ = self.transition(batch_t)
                xt = np.reshape(xt, [-1, self.m, self.n, 1])

                if i % len_target == 0:
                    target_iter = iter(target_data)

                _, c1, c2 = self.sess.run([self.train_op, self.class_loss, self.mmd_loss], feed_dict={self.source_input: xs, self.source_label: ys, self.target_input: xt})

                if np.any(np.isnan(xs)) or np.any(np.isnan(ys)):
                    print "Input Nan Type Error!! "

                if i % 20 == 0:
                    # print("Total Epoch:", '%d' % (j), "Int Epoch:", '%d' % (i), "class loss=", "{:.9f}".format(c1))
                    print "Total Epoch:", '%d' % (j), "Int Epoch:",'%d' % (i), "class loss=", "{:.9f}".format(c1), "mmd loss=","{:.9f}".format(c2)
                if i % 20 == 1:
                    self.show()

        print("Optimization Finished!")
        self.saver.save(self.sess, "./params/target3.ckpt")

    def show(self):

        data, label = self.trans(np.array(target), is_train=False)
        t_pre = self.sess.run(
            self.p_t, feed_dict={self.target_input: np.reshape(data, (-1, self.m, self.n, 1))})
        t_lab = label
        t_pre = [np.argmax(i) for i in t_pre]
        print t_lab
        print t_pre
        acc_target = sum(int(y == y_) for (y, y_) in zip(t_pre, t_lab))

        print "acc_target: {0} / {1} ".format(acc_target, 400)
        # print "acc_source: {0} / {1} ".format(acc_source, 400)
        batch_xs = data.reshape((-1, 28, 100, 1))
        batch_ys = label
        features = self.sess.run(self.fly, feed_dict={self.target_input: batch_xs})
        src_features = TSNE(n_components=2).fit_transform(features)
        slide, riot, down, push = [], [], [], []
        for i in range(len(batch_ys)):
            if batch_ys[i] == 0:
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
        o1 = plt.scatter(slide[:, 0], slide[:, 1], c='', edgecolors='#FF6347', alpha=0.8, label='slide')
        o2 = plt.scatter(riot[:, 0], riot[:, 1], c='', edgecolors='y', alpha=0.8, label='riot')
        o3 = plt.scatter(down[:, 0], down[:, 1], c='', edgecolors='g', alpha=0.8, label='down')
        o4 = plt.scatter(push[:, 0], push[:, 1], c='', edgecolors='b', alpha=0.8, label='push')
        plt.legend(loc='best')
        plt.title('Target domain (C)')
        # plt.xlim((-30, 40))
        # plt.ylim((-40, 30))
        plt.savefig('/home/han/plot123_2.png', dpi=900)
        plt.show()
        # x = []
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
        # plt.title('target3')
        # plt.show()

    def convert(self, number):
        e = np.zeros((4, 1))
        e[number] = 1
        return e

    def trans(self, batch, is_train = True):
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

    def transition(self, batch, is_train=True):
        data = None
        for i in batch[0]:
            i = i.numpy()
            i = i[:, 0:100]
            data = np.array([i]) if data is None else np.append(data, [i], axis=0)

        if is_train is True:
            label = None
            be = [i.numpy() for i in batch[1]][0]
            for j in be:
                label = np.array([self.convert(j)]) if label is None else np.append(label, [self.convert(j)], axis=0)
            label = np.reshape(label, (-1, 4))

        else:
            label = [i.numpy() for i in batch[1]][0]

        return data, label


def lrelu(x, leak=0.02):
    return tf.maximum(x, leak * x)


if __name__ =="__main__":
    batch_size = 100
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_th.pkl', 'rb') as f:
        sh = pickle.load(f)
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_tq.pkl', 'rb') as f:
        sq = pickle.load(f)
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_tqie.pkl', 'rb') as f:
        sqie = pickle.load(f)
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/source/s_tz.pkl', 'rb') as f:
        sz = pickle.load(f)
    test_ground_truth = sh[200:300] + sq[200:300] + sqie[200:300] + sz[200:300]
    # with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_th.pkl', 'rb') as f:
    #     gh = pickle.load(f)
    # with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_tq.pkl', 'rb') as f:
    #     gq = pickle.load(f)
    # with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_tqie.pkl', 'rb') as f:
    #     gqie = pickle.load(f)
    # with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_tz.pkl', 'rb') as f:
    #     gz = pickle.load(f)
    with open('/home/han/PycharmProjects/c-GAN-earth_move/GAN_data/gan_t2.pkl', 'rb') as f:
        g = pickle.load(f)
    source = g

    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_th.pkl', 'rb') as f:
    #     th1 = pickle.load(f)
    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_tq.pkl', 'rb') as f:
    #     tq1 = pickle.load(f)
    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_tqie.pkl', 'rb') as f:
    #     tqie1 = pickle.load(f)
    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target1/t_tz2.pkl', 'rb') as f:
    #     tz1 = pickle.load(f)
    # target = th1 + tq1 + tqie1 + tz1

    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_th.pkl', 'rb') as f:
    #     th2 = pickle.load(f)
    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_tq.pkl', 'rb') as f:
    #     tq2 = pickle.load(f)
    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_tqie.pkl', 'rb') as f:
    #     tqie2 = pickle.load(f)
    # with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target2/t_tz2.pkl', 'rb') as f:
    #     tz2 = pickle.load(f)
    # target = th2+tq2+tqie2+tz2

    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_th.pkl', 'rb') as f:
        th3 = pickle.load(f)
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_tq2.pkl', 'rb') as f:
        tq3 = pickle.load(f)
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_tqie.pkl', 'rb') as f:
        tqie3 = pickle.load(f)
    with open('/home/han/PycharmProjects/conditional-GAN/data_extract/target3/t_tz2.pkl', 'rb') as f:
        tz3 = pickle.load(f)
    target  = th3+tq3+tqie3+tz3

    source_data = Data.DataLoader(dataset=source, batch_size=batch_size, shuffle=True)
    target_data = Data.DataLoader(dataset=target, batch_size=batch_size, shuffle=True)

    source_iter = iter(source_data)
    target_iter = iter(target_data)
    len_source = len(source_data)
    len_target = len(target_data)

    TargetModel()
