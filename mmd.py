import tensorflow as tf


def guassian_kernel(source, target, batch_size = 128, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = batch_size * 2
    total = tf.concat([source, target], axis=0)
    total0 = tf.expand_dims(total, 0)
    total1 = tf.expand_dims(total, 1)
    L2_distance = tf.reduce_sum((total0-total1)**2, axis=2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_accelerate(source, target, batch_size = 100, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    kernels = guassian_kernel(source, target, batch_size,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def mmd_rbf_noaccelerate(source, target, batch_size = 128, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = tf.reduce_mean(XX + YY - XY -YX)
    return loss
