import numpy as np  # 导入头文件
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def load_mnist():  # 函数功能： 载入本地.npz类型minst数据
    path = 'mnist.npz'  # 放置mnist.py的目录、注意斜杠和r
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train, x_test, y_test)

def calculate_loss(model, X, y):  # 函数功能：计算损失
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model[
        'b3'], model['W4'], model['b4'], model['W5'], model['b5']
    z1 = X.dot(W1) + b1  # dot点积
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3  # Forward
    a3 = np.tanh(z3)
    z4 = a3.dot(W4) + b4
    a4 = np.tanh(z4)
    z5 = a4.dot(W5) + b5
    exp_scores = np.exp(z5)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 标准化
    num_examples = X.shape[0]
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # data_loss += reg_lambda / 2 * (
    #         np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)) + np.sum(np.square(W4)))
    return 1. / num_examples * data_loss


def predict(model, x):  # Forward
    W1, b1, W2, b2, W3, b3, W4, b4, W5, b5 = \
        model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], \
        model['b3'], model['W4'], model['b4'], model['W5'], model['b5']
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    z3 = a2.dot(W3) + b3
    a3 = np.tanh(z3)
    z4 = a3.dot(W4) + b4
    a4 = np.tanh(z4)
    z5 = a4.dot(W5) + b5
    exp_scores = np.exp(z5)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 标准化
    #输出概率最大的点
    return np.argmax(probs, axis=1)  # argmax返回使得f（x）取得最大值的变量点x的点集合


def build_model(X, y, nn_hdim, epsilon, reg_lambda, num_passes=20000, print_loss=False):
    np.random.seed(0)  # 随机数生成，seed种子，固定随机数
    num_examples = X.shape[0]
    nn_input_dim = nn_hdim[0]
    print('input dim', nn_input_dim)

    hdim1 = nn_hdim[1]
    W1 = np.random.randn(nn_input_dim, hdim1) / np.sqrt(hdim1)
    b1 = np.zeros((1, hdim1))  # zeros生成零矩阵1Xhdiml1
    print('fc: %d -> %d' % (nn_input_dim, hdim1))
    hdim2 = nn_hdim[2]
    W2 = np.random.randn(hdim1, hdim2) / np.sqrt(hdim2)
    b2 = np.zeros((1, hdim2))
    print('fc: %d -> %d' % (hdim1, hdim2))
    hdim3 = nn_hdim[3]
    W3 = np.random.randn(hdim2, hdim3) / np.sqrt(hdim3)
    b3 = np.zeros((1, hdim3))
    print('fc: %d -> %d' % (hdim2, hdim3))
    hdim4 = nn_hdim[4]
    W4 = np.random.randn(hdim3, hdim4) / np.sqrt(hdim4)
    b4 = np.zeros((1, hdim4))
    print('fc: %d -> %d' % (hdim3, hdim4))
    hdim5 = nn_hdim[5]
    W5 = np.random.randn(hdim4, hdim5) / np.sqrt(hdim5)
    b5 = np.zeros((1, hdim5))
    print('fc: %d -> %d' % (hdim4, hdim5))

    # train：
    model = {}
    bs = 128 #batchsize
    nbs_per_epoch = int(num_examples / bs)
    for i in range(0, num_passes):
        j = i % nbs_per_epoch
        if 0 == j:
            ridx = np.asarray(list(range(num_examples)))
            np.random.shuffle(ridx)
            X = X[ridx, :]
            y = y[ridx]
        Xb = X[j * bs:(j + 1) * bs, :]
        yb = y[j * bs:(j + 1) * bs]
        # Forward propagation
        z1 = Xb.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        a2 = np.tanh(z2)
        z3 = a2.dot(W3) + b3
        a3 = np.tanh(z3)
        z4 = a3.dot(W4) + b4
        a4 = np.tanh(z4)
        z5 = a4.dot(W5) + b5
        exp_scores = np.exp(z5)
        # Backpropagation
        delta_loss = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        delta_loss[range(bs), yb] -= 1
        dW5 = (a4.T).dot(delta_loss)
        db5 = np.sum(delta_loss, axis=0, keepdims=True)
        delta5 = delta_loss.dot(W5.T) * (1 - np.power(a4, 2))  # power返回给定数字乘幂
        dW4 = (a3.T).dot(delta5)
        db4 = np.sum(delta5, axis=0, keepdims=True)
        delta4 = delta5.dot(W4.T) * (1 - np.power(a3, 2))
        dW3 = (a2.T).dot(delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = delta4.dot(W3.T) * (1 - np.power(a2, 2))
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = (Xb.T).dot(delta2)
        db1 = np.sum(delta2, axis=0)
        # dW5 += reg_lambda * W5
        # dW4 += reg_lambda * W4
        # dW3 += reg_lambda * W3
        # dW2 += reg_lambda * W2
        # dW1 += reg_lambda * W1
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        W3 += -epsilon * dW3
        b3 += -epsilon * db3
        W4 += -epsilon * dW4
        b4 += -epsilon * db4
        W5 += -epsilon * dW5
        b5 += -epsilon * db5
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
                 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4, 'W5': W5, 'b5': b5}

        if print_loss and i % 2000 == 0:
            epsilon *= 0.99
            y_pred = predict(model, X_test)
            accuracy = sum(0 == (y_pred - Y_test)) / Y_test.shape[0]
            print("loss after iteration {}: {:.2f}, testing accuracy: {:.2f}%".
                  format(i, calculate_loss(model, X, y), accuracy * 100))
    return model


# load
(train_images, train_labels, test_images, test_labels) = load_mnist()  # 调用函数载入数据
n_train, w, h = train_images.shape  # shape读取矩阵长度
X_train = train_images.reshape((n_train, w * h))  # reshape将制定矩阵变换为特定维数矩阵
Y_train = train_labels  # 标签
n_test, w, h = test_images.shape
X_test = test_images.reshape((n_test, w * h))
Y_test = test_labels
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
# train
X_train = (X_train.astype(float) - 128.0) / 128.0  # astype实现变量类型的转换
X_test = (X_test.astype(float) - 128.0) / 128.0
num_examples, input_dim = X_train.shape
epsilon = 0.001
reg_lambda = 0.00
model = build_model(X_train, Y_train, [input_dim, 256, 128, 63, 32, 10], epsilon, reg_lambda, 20000, print_loss=True)
# test output
X_test0=X_test[0:3,:]
y_pred0 = predict(model, X_test0)
print(y_pred0)
X_test0=X_test0.reshape(3,w,h)
plt.figure('第一张图预测')
plt.imshow(X_test0[0,:,:])
plt.figure('第二张图预测')
plt.imshow(X_test0[1,:,:])
plt.figure('第三张图预测')
plt.imshow(X_test0[2,:,:])
pylab.show()
