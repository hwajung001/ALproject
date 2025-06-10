import numpy as np
from common.layers import Convolution, BatchNormalization, Relu, Pooling, Affine
from common.functions import softmax, cross_entropy_error

def fake_quantize(x, num_bits=8):
    qmin, qmax = 0., 2.**num_bits - 1.
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return x
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = np.clip(np.round(qmin - x_min / scale), qmin, qmax)
    q_x = np.clip(np.round(zero_point + x / scale), qmin, qmax)
    return scale * (q_x - zero_point)

class Flatten:
    def __init__(self):
        self.orig_shape = None

    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)


class MiniVGGNet:
    def __init__(self, input_dim=(3, 32, 32), num_classes=100):
        in_channels, _, _ = input_dim
        weight_std = np.sqrt(2. / in_channels)

        self.conv1 = Convolution(np.random.randn(64, in_channels, 3, 3) * weight_std, np.zeros(64), stride=1, pad=1)
        self.bn1   = BatchNormalization(np.ones(64), np.zeros(64))
        self.relu1 = Relu()

        self.conv2 = Convolution(np.random.randn(64, 64, 3, 3) * weight_std, np.zeros(64), stride=1, pad=1)
        self.bn2   = BatchNormalization(np.ones(64), np.zeros(64))
        self.relu2 = Relu()
        self.pool1 = Pooling(2, 2, stride=2)

        self.conv3 = Convolution(np.random.randn(128, 64, 3, 3) * weight_std, np.zeros(128), stride=1, pad=1)
        self.bn3   = BatchNormalization(np.ones(128), np.zeros(128))
        self.relu3 = Relu()

        self.conv4 = Convolution(np.random.randn(128, 128, 3, 3) * weight_std, np.zeros(128), stride=1, pad=1)
        self.bn4   = BatchNormalization(np.ones(128), np.zeros(128))
        self.relu4 = Relu()
        self.pool2 = Pooling(2, 2, stride=2)

        self.conv5 = Convolution(np.random.randn(256, 128, 3, 3) * weight_std, np.zeros(256), stride=1, pad=1)
        self.bn5   = BatchNormalization(np.ones(256), np.zeros(256)) #conv5 also 
        self.relu5 = Relu()
        self.pool3 = Pooling(2, 2, stride=2)

        self.flatten = Flatten()
        self.fc1 = Affine(np.random.randn(4096, 512) * weight_std, np.zeros(512))
        self.relu6 = Relu()
        self.fc2 = Affine(np.random.randn(512, num_classes) * 0.01, np.zeros(num_classes))

        self.layers = [
            self.conv1, self.bn1, self.relu1,
            self.conv2, self.bn2, self.relu2, self.pool1,
            self.conv3, self.bn3, self.relu3,
            self.conv4, self.bn4, self.relu4, self.pool2,
            self.conv5, self.bn5, self.relu5, self.pool3, #conv5
            self.flatten, self.fc1, self.relu6, self.fc2
        ]

    def forward(self, x, train_flg=True):
        for layer in self.layers:
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def backward(self, dout):
        dout = self.relu6.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)

        dout = self.pool3.backward(dout)
        dout = self.relu5.backward(dout)
        dout = self.bn5.backward(dout)
        dout = self.conv5.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu4.backward(dout)
        dout = self.bn4.backward(dout)
        dout = self.conv4.backward(dout)

        dout = self.relu3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        return dout

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.fc2.backward(dout)
        dout = self.relu6.backward(dout)
        dout = self.fc1.backward(dout)
        dout = self.flatten.backward(dout)

        dout = self.pool3.backward(dout)
        dout = self.relu5.backward(dout)
        dout = self.bn5.backward(dout)
        dout = self.conv5.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu4.backward(dout)
        dout = self.bn4.backward(dout)
        dout = self.conv4.backward(dout)

        dout = self.relu3.backward(dout)
        dout = self.bn3.backward(dout)
        dout = self.conv3.backward(dout)

        dout = self.pool1.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

        grads = {
            'W1': self.conv1.W, 'b1': self.conv1.b,
            'gamma1': self.bn1.gamma, 'beta1': self.bn1.beta,
            'W2': self.conv2.W, 'b2': self.conv2.b,
            'gamma2': self.bn2.gamma, 'beta2': self.bn2.beta,
            'W3': self.conv3.W, 'b3': self.conv3.b,
            'gamma3': self.bn3.gamma, 'beta3': self.bn3.beta,
            'W4': self.conv4.W, 'b4': self.conv4.b,
            'gamma4': self.bn4.gamma, 'beta4': self.bn4.beta,
            'W5': self.conv5.W, 'b5': self.conv5.b,
            'gamma5': self.bn5.gamma, 'beta5': self.bn5.beta,  
            'W6': self.fc1.W, 'b6': self.fc1.b,
            'W7': self.fc2.W, 'b7': self.fc2.b,
        }

        return grads

    def predict(self, x, batch_size=100):
        return np.concatenate([self.forward(x[i:i+batch_size], False) for i in range(0, x.shape[0], batch_size)], axis=0)

    def loss(self, x, t):
        y = self.forward(x, True)
        y_softmax = softmax(y)  
        return cross_entropy_error(y_softmax, t)

    def accuracy(self, x, t, batch_size=100):
        pred = np.argmax(self.predict(x, batch_size), axis=1)
        true = t if t.ndim == 1 else np.argmax(t, axis=1)
        return np.mean(pred == true)

    def clip_weights(self, clip_value=1.0): 
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc1, self.fc2]:
            layer.W = np.clip(layer.W, -clip_value, clip_value)
