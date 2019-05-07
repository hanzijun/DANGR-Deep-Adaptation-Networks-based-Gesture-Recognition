#coding:utf-8
from tensorflow.examples.tutorials.mnist import input_data
import pylab
from ops_modify import *
import numpy as np
import pickle
import random
import os

with open('../conditional-GAN/data_extract/source/s_th.pkl', 'rb') as f:
    sh = pickle.load(f)
with open('../conditional-GAN/data_extract/source/s_tq.pkl', 'rb') as f:
    sq = pickle.load(f)
with open('../conditional-GAN/data_extract/source/s_tqie.pkl', 'rb') as f:
    sqie = pickle.load(f)
with open('../conditional-GAN/data_extract/source/s_tz.pkl', 'rb') as f:
    sz = pickle.load(f)

train_data = sh[0:200] + sq[0:200] + sqie[0:200] + sz[0:200]

global_step = tf.Variable(0, name='global_step', trainable=False)
label = tf.placeholder(tf.float32, [BATCH_SIZE, 4], name='label')
images = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 100, 1], name='images')
niose = tf.placeholder(tf.float32, [BATCH_SIZE, 100], name='noise')

with tf.variable_scope(tf.get_variable_scope()) as scope:
    G_out = generator(niose, label)
    D_logits_real = discriminator(images, label)
    D_logits_fake = discriminator(G_out, label, reuse=True)
    samples = sampler(niose, label)

def transition(batch, is_train=True):
    data = None
    for i in batch[:, 0]:
            i = i[:, 0:100]
            data = np.array([i]) if data is None else np.append(data, [i], axis=0)

    if is_train is True:
        label = None
        for j in batch[:, 1]:
            label = np.array([convert(j[0])]) if label is None else np.append(label, [convert(j[0])], axis=0)
        label = np.reshape(label, (-1, 4))

    else:
        label = []
        for j in batch[:, 1]:
            label.append(j[0])

    return data, label

def convert(number):
    e = np.zeros((4, 1))
    e[number] = 1
    return e

# label for generating dataset
sample_labels = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                                                    [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                                                    [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                                                    [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

d_loss_real = -tf.reduce_mean(D_logits_real)
d_loss_fake = tf.reduce_mean(D_logits_fake)
d_loss = d_loss_real + d_loss_fake
g_loss = -d_loss_fake

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

# for tf.layers.batch_normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optim = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(0.001, beta1=0.5).minimize(g_loss, var_list=g_vars, global_step=global_step)

is_train = False
is_param = True
if is_train:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        if is_param:
            saver.restore(sess,'./check_point/CGAN_model_52.ckpt')
            # saver.restore(sess, tf.train.latest_checkpoint('./check_point'))
            print "loading params"
        for i in range(1001):
                random.shuffle(train_data)
                batchs = [
                    train_data[k: k + BATCH_SIZE]
                    for k in xrange(0, 800, BATCH_SIZE)]

                for batch in batchs:
                    data, tag = transition(np.array(batch))

                    batch_xs = np.reshape(data, (BATCH_SIZE, 28, 100, 1))
                    batch_xs = batch_xs / 1.5
                    batch_ys = tag

                    batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
                    sess.run([d_optim], feed_dict={images: batch_xs, niose: batch_z, label: batch_ys})
                    sess.run([g_optim], feed_dict={images: batch_xs, niose: batch_z, label: batch_ys})
                    # sess.run([g_optim], feed_dict={images: batch_xs, niose: batch_z, label: batch_ys})

                if i % 10 == 0:
                        errD = d_loss.eval(feed_dict={images: batch_xs, label: batch_ys, niose: batch_z})
                        errG = g_loss.eval({niose: batch_z, label: batch_ys})
                        print("epoch:[%d], d_loss: %.8f, g_loss: %.8f" % (i, errD, errG))

                if i % 50 == 1:
                    sample = sess.run(samples, feed_dict={niose: batch_z, label: sample_labels})
                    sample = sample * 1.5
                    samples_path = './pics/'
                    save_images(sample, [10,4], samples_path + '%d.png' % (i))
                    print('save image')

                if i % 50 == 2:
                    checkpoint_path = os.path.join('./check_point/CGAN_model_%d.ckpt' % (i))
                    saver.save(sess, checkpoint_path)
                    print('save check_point')
else:
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, './check_point/CGAN_model_52.ckpt')

            gan_data = []
            for i in range(200):
                sample_noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))#n-sample=1
                gen_samples = sess.run(generator(niose, label, training =False), feed_dict={niose: sample_noise, label : sample_labels})
                gen_samples = gen_samples.reshape(-1, 28, 100)
                gen_samples = gen_samples * 1.5
                tags = [[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3],[0],[1],[2],[3]]
                for i,j in enumerate(gen_samples):
                        gan_data.append([j,tags[i]])

        with open('./GAN_data/gan_t2.pkl', 'wb') as f:
            pickle.dump(gan_data, f)

        # pylab.figure()
        # pylab.subplot(4,1,1)
        # pylab.imshow(csi_generator[0])
        # pylab.legend(loc="best")
        #
        # pylab.subplot(4,1,2)
        # pylab.imshow(csi_generator[1])
        # pylab.legend(loc="best")
        #
        # pylab.subplot(4,1,3)
        # pylab.imshow(csi_generator[2])
        # pylab.legend(loc="best")
        #
        # pylab.subplot(4,1,4)
        # pylab.imshow(csi_generator[3])
        # pylab.legend(loc="best")
        # pylab.show()
