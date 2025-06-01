import numpy as np

# Fake Quantization 함수
def fake_quantize(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max == x_min:
        return x  # avoid divide by zero
    
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - x_min / scale
    zero_point = np.clip(np.round(zero_point), qmin, qmax)

    q_x = zero_point + x / scale
    q_x = np.clip(np.round(q_x), qmin, qmax)
    fq_x = scale * (q_x - zero_point)
    return fq_x

# 모델 레이어 및 ResNet-20 정의
from common.layers import Convolution, Affine, Relu, BatchNormalization
from common.functions import softmax, cross_entropy_error
from common.util import im2col, col2im

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        # Fake Quantization
        W_q = fake_quantize(self.W)
        b_q = fake_quantize(self.b)
        x_q = fake_quantize(self.x)

        out = np.dot(x_q, W_q) + b_q
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, _, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # Fake Quantization
        W_q = fake_quantize(self.W)
        b_q = fake_quantize(self.b)
        x_q = fake_quantize(x)

        col = im2col(x_q, FH, FW, self.stride, self.pad)
        col_W = W_q.reshape(FN, -1).T
        out = np.dot(col, col_W) + b_q
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout).transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1):
        self.stride = stride
        self.equal_in_out = (in_channels == out_channels and stride == 1)

        self.conv1 = Convolution(
            W=np.random.randn(out_channels, in_channels, 3, 3) * np.sqrt(2. / in_channels),
            b=np.zeros(out_channels),
            stride=stride,
            pad=1
        )
        self.bn1 = BatchNormalization(gamma=np.ones(out_channels), beta=np.zeros(out_channels))
        self.relu1 = Relu()

        self.conv2 = Convolution(
            W=np.random.randn(out_channels, out_channels, 3, 3) * np.sqrt(2. / out_channels),
            b=np.zeros(out_channels),
            stride=1,
            pad=1
        )
        self.bn2 = BatchNormalization(gamma=np.ones(out_channels), beta=np.zeros(out_channels))
        self.relu2 = Relu()

        if not self.equal_in_out:
            self.shortcut = Convolution(
                W=np.random.randn(out_channels, in_channels, 1, 1) * np.sqrt(2. / in_channels),
                b=np.zeros(out_channels),
                stride=stride,
                pad=0
            )
            self.bn_shortcut = BatchNormalization(gamma=np.ones(out_channels), beta=np.zeros(out_channels))

    def forward(self, x, train_flg=True, skip_prob=0.0):
        self.x = x

        if train_flg and np.random.rand() < skip_prob:
            return x  # skip this residual block
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, train_flg)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out, train_flg)
        self.out_main = out

        if self.equal_in_out:
            shortcut = x
        else:
            shortcut = self.shortcut.forward(x)
            shortcut = self.bn_shortcut.forward(shortcut, train_flg)
        self.out_shortcut = shortcut

        out += shortcut
        out = self.relu2.forward(out)

        return out

    def backward(self, dout):
        dout = self.relu2.backward(dout)

        dshortcut = dout.copy()
        dmain = dout.copy()

        dmain = self.bn2.backward(dmain)
        dmain = self.conv2.backward(dmain)

        dmain = self.relu1.backward(dmain)
        dmain = self.bn1.backward(dmain)
        dmain = self.conv1.backward(dmain)

        if not self.equal_in_out:
            dshortcut = self.bn_shortcut.backward(dshortcut)
            dshortcut = self.shortcut.backward(dshortcut)

        dx = dmain + dshortcut
        return dx

class ResNet20:
    def __init__(self, input_dim=(3, 32, 32), num_classes=100):
        self.params = []
        self.trainable_layers = []

        self.conv1 = Convolution(
            W=np.random.randn(16, 3, 3, 3) * np.sqrt(2. / 3),
            b=np.zeros(16),
            stride=1,
            pad=1
        )
        self.bn1 = BatchNormalization(gamma=np.ones(16), beta=np.zeros(16))
        self.relu1 = Relu()

        self.layer1 = [ResidualBlock(16, 16, stride=1) for _ in range(3)]
        self.layer2 = [ResidualBlock(16 if i == 0 else 32, 32, stride=2 if i == 0 else 1) for i in range(3)]
        self.layer3 = [ResidualBlock(32 if i == 0 else 64, 64, stride=2 if i == 0 else 1) for i in range(3)]

        self.fc = Affine(W=np.random.randn(64, num_classes) * np.sqrt(2. / 64), b=np.zeros(num_classes))

    def clip_weights(self, clip_value=1.0):
    # 개별 레이어의 weight들을 [-clip_value, clip_value]로 제한
        self.conv1.W = np.clip(self.conv1.W, -clip_value, clip_value)
        self.fc.W = np.clip(self.fc.W, -clip_value, clip_value)

        for block in self.layer1 + self.layer2 + self.layer3:
            block.conv1.W = np.clip(block.conv1.W, -clip_value, clip_value)
            block.conv2.W = np.clip(block.conv2.W, -clip_value, clip_value)
            if not block.equal_in_out:
                block.shortcut.W = np.clip(block.shortcut.W, -clip_value, clip_value)

    def forward(self, x, train_flg=True, skip_prob=0.0):
        self.input = x

        if train_flg and np.random.rand() < skip_prob:
            return x  # skip this residual block
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, train_flg)
        out = self.relu1.forward(out)

        for block in self.layer1:
            out = block.forward(out, train_flg)
        for block in self.layer2:
            out = block.forward(out, train_flg)
        for block in self.layer3:
            out = block.forward(out, train_flg)

        self.feature_map = out

        N, C, H, W = out.shape
        out = out.mean(axis=(2, 3))

        self.pooled = out
        out = self.fc.forward(out)
        return out

    def predict(self, x, batch_size=100):
        y_list = []
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = self.forward(x_batch, train_flg=False)
            y_list.append(y_batch)
        return np.concatenate(y_list, axis=0)

    def loss(self, x, t):
        y = self.forward(x, train_flg=True)
        return cross_entropy_error(softmax(y), t)

    def accuracy(self, x, t, batch_size=100):
        acc = 0.0
        total = x.shape[0]
        for i in range(0, total, batch_size):
            x_batch = x[i:i+batch_size]
            t_batch = t[i:i+batch_size]

            y = self.predict(x_batch)
            y = np.argmax(y, axis=1)

            if t.ndim != 1:
                t_batch = np.argmax(t_batch, axis=1)

            acc += np.sum(y == t_batch)

        return acc / total

    def backward(self, dout):
        dout = self.fc.backward(dout)
        dout = dout.reshape(self.feature_map.shape[0], self.feature_map.shape[1], 1, 1)
        dout = dout.repeat(self.feature_map.shape[2], axis=2).repeat(self.feature_map.shape[3], axis=3)

        for block in reversed(self.layer3):
            dout = block.backward(dout)
        for block in reversed(self.layer2):
            dout = block.backward(dout)
        for block in reversed(self.layer1):
            dout = block.backward(dout)

        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout