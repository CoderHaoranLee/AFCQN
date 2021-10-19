# -*- coding:utf-8 -*-
import tensorflow as tflow
import numpy as np

learning_rate = 0.01
epoch_size = 20
step_size = 10       #正负样本共计为20
test_size = 20
data_size = 200


cnn_layer_num = 2
input_size = (30, 30)
class_num = 2
# 占位符输入
x = tflow.placeholder(tflow.float32, [None, input_size[0], input_size[1], 3])
y = tflow.placeholder(tflow.float32, [None, class_num])
keep_prob = tflow.placeholder(tflow.float32)

# 卷积操作
def conv2d(name, l_input, w, b, st):
    return tflow.nn.relu(tflow.nn.bias_add(tflow.nn.conv2d(l_input, w, st, padding='SAME'), b), name=name)


# 最大下采样操作
def max_pool(name, l_input, k):
    return tflow.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 归一化操作
def norm(name, l_input, lsize=4):
    return tflow.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 存储所有的网络参数
weights = {
    'wc1': tflow.Variable(tflow.random_normal([3, 3, 3, 18])),
    'wc2': tflow.Variable(tflow.random_normal([3, 3, 18, 36])),
    # 'wc3': tflow.Variable(tflow.random_normal([3, 3, 192, 384])),
    # 'wc4': tflow.Variable(tflow.random_normal([3, 3, 384, 384])),
    # 'wc5': tflow.Variable(tflow.random_normal([3, 3, 384, 256])),
    'wd1': tflow.Variable(tflow.random_normal([2 * 2 * 36, 144])),
    'wd2': tflow.Variable(tflow.random_normal([144, 144])),
    'out': tflow.Variable(tflow.random_normal([144, 2]))
}
biases = {
    'bc1': tflow.Variable(tflow.random_normal([18])),
    'bc2': tflow.Variable(tflow.random_normal([36])),
    # 'bc3': tflow.Variable(tflow.random_normal([384])),
    # 'bc4': tflow.Variable(tflow.random_normal([384])),
    # 'bc5': tflow.Variable(tflow.random_normal([256])),
    'bd1': tflow.Variable(tflow.random_normal([144])),
    'bd2': tflow.Variable(tflow.random_normal([144])),
    'out': tflow.Variable(tflow.random_normal([class_num]))
}


# 定义整个网络
def alex_net(_X, _weights, _biases, _dropout):
    # 向量转为矩阵


    # 第一层卷积
    # 卷积
    conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'],[1,4,4,1])
    # 下采样
    pool1 = max_pool('pool1', conv1, k=2)
    # 归一化
    norm1 = norm('norm1', pool1, lsize=4)

    # 第二层卷积
    # 卷积
    conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'],[1,1,1,1])
    # 下采样
    pool2 = max_pool('pool2', conv2, k=2)
    # 归一化
    norm2 = norm('norm2', pool2, lsize=4)

    # # 第三层卷积
    # # 卷积
    # conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
    # # 归一化
    # norm3 = norm('norm3', conv3, lsize=4)
    #
    # # 第四层卷积
    # # 卷积
    # conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
    # # 归一化
    # norm4 = norm('norm4', conv4, lsize=4)
    #
    # # 第五层卷积
    # # 卷积
    # conv5 = conv2d('conv5', norm4, _weights['wc5'], _biases['bc5'])
    # # 下采样
    # pool5 = max_pool('pool5', conv5, k=2)
    # # 归一化
    # norm5 = norm('norm5', pool5, lsize=4)

    # 全连接层1，先把特征图转为向量
    dense1 = tflow.reshape(norm2, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tflow.nn.relu(tflow.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    dense1 = tflow.nn.dropout(dense1, _dropout)

    # 全连接层2
    dense2 = tflow.reshape(dense1, [-1, _weights['wd2'].get_shape().as_list()[0]])
    dense2 = tflow.nn.relu(tflow.matmul(dense2, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation
    dense2 = tflow.nn.dropout(dense2, _dropout)

    # 网络输出层
    out = tflow.matmul(dense2, _weights['out']) + _biases['out']
    return out

if __name__ == "__main__":
    saver = tflow.train.Saver()
    # 构建模型
    pred = alex_net(x, weights, biases, keep_prob)
    y_result = tflow.nn.softmax(pred)

    # 定义损失函数和学习步骤
    cost = tflow.reduce_mean(tflow.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tflow.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # 测试网络
    correct_pred = tflow.equal(tflow.argmax(pred, 1), tflow.argmax(y, 1))
    accuracy = tflow.reduce_mean(tflow.cast(correct_pred, tflow.float32))

    # 初始化所有的共享变量
    init = tflow.initialize_all_variables()
