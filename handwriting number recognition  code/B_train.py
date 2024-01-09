import numpy as np
import pickle
from tqdm import tqdm
from numpy.lib.function_base import select
import matplotlib.pylab as plt
from A_mnist import load_mnist


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


# (x_train,t_train),(x_val,t_val)=load_mnist(normalize=True,one_hot_label=True)
# 两层神经网络的类
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # 损失函数
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    # 数值微分法
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    # 误差反向传播法
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    # 准确率
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


if __name__ == '__main__':
    (x_train, t_train), (x_val, t_val) = load_mnist(normalize=True, one_hot_label=True)
    net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    # 超参数
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    epoch = 20
    batch_per_epoch = train_size / batch_size

    # 记录准确率与损失函数
    train_acc_list = [0]
    val_acc_list = [0]
    train_loss_list = [1]



    # 总共20个epoch
    for i in tqdm(range(epoch)):

        for j in range(int(batch_per_epoch)):

            # 随机取小批量数据
            batch_mask = np.random.choice(train_size, batch_size, replace=False)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            # 计算梯度
            # 数值微分 计算很慢
            # grad=net.numerical_gradient(x_batch,t_batch)
            # 误差反向传播法 计算很快
            grad = net.gradient(x_batch, t_batch)
            # 更新参数 权重W和偏重b
            for key in ['W1', 'b1', 'W2', 'b2']:
                net.params[key] -= learning_rate * grad[key]


        # 记录准确度
        train_acc = net.accuracy(x_train, t_train)
        val_acc = net.accuracy(x_val, t_val)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        print('train acc:' + str(train_acc) + '   val acc:' + str(val_acc))

        # 记录损失函数
        loss = net.loss(x_train, t_train)
        train_loss_list.append(loss)
        print('loss:' + str(loss))

    print(train_acc_list)
    print(val_acc_list)
    print(train_loss_list)

    markers = {'train': 'o', 'val': 's'}
    x = np.arange(len(train_acc_list))

    # 创建一个包含两个子图的画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # 绘制训练准确率曲线
    ax1.plot(x, train_acc_list, label='train acc')
    ax1.plot(x, val_acc_list, label='val acc', linestyle='--')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='lower right')

    # 绘制损失函数曲线
    ax2.plot(x, train_loss_list, label='loss', color='red')
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


    print('训练次数:' + str(i+1) + '    loss:' + str(loss))

    # 保存模型参数
    with open('model_params.pkl', 'wb') as f:
        pickle.dump(net.params, f)