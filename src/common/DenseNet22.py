import numpy as np
from common.layers import Convolution, BatchNormalization, Relu, Affine
from common.functions import softmax, cross_entropy_error
from common.util import im2col, col2im

def fake_quantize(x, num_bits=8):
    qmin, qmax = 0., 2.**num_bits - 1.
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return x
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = np.clip(np.round(qmin - x_min / scale), qmin, qmax)
    q_x = np.clip(np.round(zero_point + x / scale), qmin, qmax)
    return scale * (q_x - zero_point)

class DenseLayer:
    def __init__(self, in_channels, growth_rate):
        self.bn = BatchNormalization(np.ones(in_channels), np.zeros(in_channels))
        self.relu = Relu()
        self.conv = Convolution(
            np.random.randn(growth_rate, in_channels, 3, 3) * np.sqrt(2. / in_channels),
            np.zeros(growth_rate), stride=1, pad=1)

    def forward(self, x, train_flg=True):
        out = self.bn.forward(x, train_flg)
        out = self.relu.forward(out)
        out = self.conv.forward(out)
        self.out = out
        return np.concatenate([x, out], axis=1)

    def backward(self, dout):
        dx_main = dout[:, -self.out.shape[1]:, :, :]
        dx_input = dout[:, :-self.out.shape[1], :, :]
        dx_main = self.conv.backward(dx_main)
        dx_main = self.relu.backward(dx_main)
        dx_main = self.bn.backward(dx_main)
        return dx_input + dx_main

class TransitionLayer:
    def __init__(self, in_channels):
        out_channels = in_channels // 2
        self.bn = BatchNormalization(np.ones(in_channels), np.zeros(in_channels))
        self.relu = Relu()
        self.conv = Convolution(
            np.random.randn(out_channels, in_channels, 1, 1) * np.sqrt(2. / in_channels),
            np.zeros(out_channels), stride=1, pad=0)
        self.pool = lambda x: x[:, :, ::2, ::2]  # 2x2 average pool (stride=2)

    def forward(self, x, train_flg=True):
        out = self.bn.forward(x, train_flg)
        out = self.relu.forward(out)
        out = self.conv.forward(out)
        return self.pool(out)

    def backward(self, dout):
        N, C, H, W = dout.shape
        d_upsampled = np.zeros((N, C, H*2, W*2))
        d_upsampled[:, :, ::2, ::2] = dout  # unpool
        d_conv = self.conv.backward(d_upsampled)
        d_relu = self.relu.backward(d_conv)
        return self.bn.backward(d_relu)

class DenseBlock:
    def __init__(self, num_layers, in_channels, growth_rate):
        self.layers = []
        channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(channels, growth_rate)
            self.layers.append(layer)
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x, train_flg=True):
        for layer in self.layers:
            x = layer.forward(x, train_flg)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class DenseNet22:
    def __init__(self, input_dim=(3, 32, 32), num_classes=100, growth_rate=12):
        self.growth_rate = growth_rate
        self.conv1 = Convolution(
            np.random.randn(16, 3, 3, 3) * np.sqrt(2. / 3), np.zeros(16), stride=1, pad=1)
        self.bn1 = BatchNormalization(np.ones(16), np.zeros(16))
        self.relu1 = Relu()

        self.block1 = DenseBlock(6, 16, growth_rate)
        self.trans1 = TransitionLayer(self.block1.out_channels)

        self.block2 = DenseBlock(6, self.block1.out_channels // 2, growth_rate)
        self.trans2 = TransitionLayer(self.block2.out_channels)

        self.block3 = DenseBlock(6, self.block2.out_channels // 2, growth_rate)

        final_channels = self.block3.out_channels
        self.fc = Affine(np.random.randn(final_channels, num_classes) * np.sqrt(2. / final_channels), np.zeros(num_classes))

    def forward(self, x, train_flg=True):
        out = self.relu1.forward(self.bn1.forward(self.conv1.forward(x), train_flg))
        out = self.block1.forward(out, train_flg)
        out = self.trans1.forward(out, train_flg)
        out = self.block2.forward(out, train_flg)
        out = self.trans2.forward(out, train_flg)
        out = self.block3.forward(out, train_flg)
        self.feature_map = out
        out = out.mean(axis=(2, 3))  # global avg pool
        self.pooled = out
        return self.fc.forward(out)

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = dout[:, :, None, None]
        dout = dout.repeat(self.feature_map.shape[2], axis=2)
        dout = dout.repeat(self.feature_map.shape[3], axis=3)
        dout = self.block3.backward(dout)
        dout = self.trans2.backward(dout)
        dout = self.block2.backward(dout)
        dout = self.trans1.backward(dout)
        dout = self.block1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        return self.conv1.backward(dout)

    def predict(self, x, batch_size=100):
        return np.concatenate([self.forward(x[i:i+batch_size], False) for i in range(0, x.shape[0], batch_size)], axis=0)

    def loss(self, x, t):
        return cross_entropy_error(softmax(self.forward(x, True)), t)

    def accuracy(self, x, t, batch_size=100):
        pred = np.argmax(self.predict(x, batch_size), axis=1)
        true = t if t.ndim == 1 else np.argmax(t, axis=1)
        return np.mean(pred == true)

    def clip_weights(self, clip_value=1.0):
        self.conv1.W = np.clip(self.conv1.W, -clip_value, clip_value)
        self.fc.W = np.clip(self.fc.W, -clip_value, clip_value)
        for block in [self.block1, self.block2, self.block3]:
            for layer in block.layers:
                layer.conv.W = np.clip(layer.conv.W, -clip_value, clip_value)
        for trans in [self.trans1, self.trans2]:
            trans.conv.W = np.clip(trans.conv.W, -clip_value, clip_value)
