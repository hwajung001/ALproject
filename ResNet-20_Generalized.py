import numpy as np
import copy
import pickle
import os
import urllib.request
import tarfile
from common.util import shuffle_dataset
import matplotlib.pyplot as plt

# CIFAR-100 다운로드 및 압축 해제
def download_cifar100(dest="./cifar-100-python"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"

    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

    def safe_extract(tar, path=".", members=None):
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(path, members)

    if not os.path.exists(dest):
        os.makedirs(dest, exist_ok=True)
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:gz") as tar:
            safe_extract(tar, path="./")
        print("CIFAR-100 downloaded and extracted.")
    else:
        print("CIFAR-100 already downloaded.")

# 데이터 배치 로딩
def load_batch(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        data = data_dict[b'data']
        fine_labels = np.array(data_dict[b'fine_labels'])
        coarse_labels = np.array(data_dict[b'coarse_labels'])
        return data, fine_labels, coarse_labels

# 메타데이터 로딩
def load_meta(data_dir="./cifar-100-python"):
    with open(os.path.join(data_dir, "meta"), 'rb') as f:
        meta_dict = pickle.load(f, encoding='bytes')
        fine_label_names = [name.decode('utf-8') for name in meta_dict[b'fine_label_names']]
        coarse_label_names = [name.decode('utf-8') for name in meta_dict[b'coarse_label_names']]
        return {"fine_label_names": fine_label_names, "coarse_label_names": coarse_label_names}

# 정규화 함수
def normalize(x):
    mean = np.array([0.507, 0.487, 0.441]).reshape(1, 3, 1, 1)
    std = np.array([0.267, 0.256, 0.276]).reshape(1, 3, 1, 1)
    return (x - mean) / std

# 전체 데이터 로딩
def load_cifar100(data_dir="./cifar-100-python"):
    x_train, y_train_fine, y_train_coarse = load_batch(os.path.join(data_dir, "train"))
    x_test, y_test_fine, y_test_coarse = load_batch(os.path.join(data_dir, "test"))

    x_train = x_train.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    val_size = int(0.1 * len(x_train))
    x_val, y_val_fine, y_val_coarse = (
        x_train[:val_size], y_train_fine[:val_size], y_train_coarse[:val_size]
    )
    x_train, y_train_fine, y_train_coarse = (
        x_train[val_size:], y_train_fine[val_size:], y_train_coarse[val_size:]
    )

    x_train, y_train_fine = shuffle_dataset(x_train, y_train_fine)
    x_train, y_train_coarse = shuffle_dataset(x_train, y_train_coarse)

    return (x_train, y_train_fine, y_train_coarse), (x_val, y_val_fine, y_val_coarse), (x_test, y_test_fine, y_test_coarse)

download_cifar100()
(x_train, y_train_fine, y_train_coarse), (x_val, y_val_fine, y_val_coarse), (x_test, y_test_fine, y_test_coarse) = load_cifar100()
meta = load_meta()

# 증강 함수
def random_crop(x, crop_size=32, padding=4):
    n, c, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
    cropped = np.empty((n, c, crop_size, crop_size), dtype=x.dtype)
    for i in range(n):
        top = np.random.randint(0, padding * 2 + 1)
        left = np.random.randint(0, padding * 2 + 1)
        cropped[i] = padded[i, :, top:top+crop_size, left:left+crop_size]
    return cropped

def horizontal_flip(x):
    return x[:, :, :, ::-1]

def cutout(x, size=16):  
    x_cut = x.copy()
    n, c, h, w = x.shape
    for i in range(n):
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1 = np.clip(cy - size // 2, 0, h)
        y2 = np.clip(cy + size // 2, 0, h)
        x1 = np.clip(cx - size // 2, 0, w)
        x2 = np.clip(cx + size // 2, 0, w)
        x_cut[i, :, y1:y2, x1:x2] = 0
    return x_cut

def color_jitter(x, brightness=0.3, contrast=0.3):
    x_jittered = x.copy()
    for i in range(x.shape[0]):
        b = 1 + np.random.uniform(-brightness, brightness)
        c = 1 + np.random.uniform(-contrast, contrast)
        mean = x_jittered[i].mean(axis=(1, 2), keepdims=True)
        x_jittered[i] = (x_jittered[i] - mean) * c + mean
        x_jittered[i] = np.clip(x_jittered[i] * b, 0, 1)
    return x_jittered

# 증강 적용 함수 (on-the-fly)
def apply_augmentations(x, crop_size=32, padding=4, cutout_size=16):
    # random crop
    n, c, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='reflect')
    cropped = np.empty((n, c, crop_size, crop_size), dtype=x.dtype)
    for i in range(n):
        top = np.random.randint(0, padding * 2 + 1)
        left = np.random.randint(0, padding * 2 + 1)
        cropped[i] = padded[i, :, top:top+crop_size, left:left+crop_size]

    # horizontal flip
    if np.random.rand() < 0.5:
        cropped = cropped[:, :, :, ::-1]

    # cutout
    for i in range(n):
        cy, cx = np.random.randint(crop_size), np.random.randint(crop_size)
        y1 = np.clip(cy - cutout_size // 2, 0, crop_size)
        y2 = np.clip(cy + cutout_size // 2, 0, crop_size)
        x1 = np.clip(cx - cutout_size // 2, 0, crop_size)
        x2 = np.clip(cx + cutout_size // 2, 0, crop_size)
        cropped[i, :, y1:y2, x1:x2] = 0

    return cropped

def smooth_labels(y, smoothing=0.1, num_classes=100):
    confidence = 1.0 - smoothing
    label_shape = (y.shape[0], num_classes)
    smooth = np.full(label_shape, smoothing / (num_classes - 1))
    smooth[np.arange(y.shape[0]), y] = confidence
    return smooth

meta = load_meta()
label_names = meta['fine_label_names']

# 원본 x_train 로드
x_train, y_train_fine, _ = load_batch("./cifar-100-python/train")
x_train = x_train.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
x_train = normalize(x_train)

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

# 모델 구조 출력

def count_params(layer):
    count = 0
    if hasattr(layer, 'W'):
        count += np.prod(layer.W.shape)
    if hasattr(layer, 'b'):
        count += np.prod(layer.b.shape)
    return count

def print_resnet20_summary(model, input_shape=(1, 3, 32, 32)):
    print("=" * 75, flush=True)
    print(f"{'Layer (type)':<35}{'Output Shape':<25}{'Param #':>10}", flush=True)
    print("=" * 75, flush=True)

    x = np.zeros(input_shape)
    total_params = 0
    layer_idx = 1

    # Conv1
    x = model.conv1.forward(x)
    p = count_params(model.conv1)
    print(f"{layer_idx:>2}. {'Conv1':<32}{str(x.shape):<25}{p:>10,}", flush=True)
    total_params += p
    layer_idx += 1

    x = model.bn1.forward(x, train_flg=False)
    x = model.relu1.forward(x)

    # Residual Blocks
    for i, layer_block in enumerate([model.layer1, model.layer2, model.layer3]):
        for j, block in enumerate(layer_block):
            residual = x.copy()

            # Conv1
            x = block.conv1.forward(x)
            p = count_params(block.conv1)
            name = f"Block[{i+1}-{j+1}]_Conv1"
            print(f"{layer_idx:>2}. {name:<32}{str(x.shape):<25}{p:>10,}", flush=True)
            total_params += p
            layer_idx += 1

            x = block.bn1.forward(x, train_flg=False)
            x = block.relu1.forward(x)

            # Conv2
            x = block.conv2.forward(x)
            p = count_params(block.conv2)
            name = f"Block[{i+1}-{j+1}]_Conv2"
            print(f"{layer_idx:>2}. {name:<32}{str(x.shape):<25}{p:>10,}", flush=True)
            total_params += p
            layer_idx += 1

            x = block.bn2.forward(x, train_flg=False)

            # Shortcut (optional)
            if not block.equal_in_out:
                x_sc = block.shortcut.forward(residual)
                p = count_params(block.shortcut)
                name = f"└─ Shortcut[{i+1}-{j+1}]"
                print(f"{'':>3} {name:<32}{str(x_sc.shape):<25}{p:>10,}", flush=True)
                total_params += p
                x = x + x_sc
                x = block.bn_shortcut.forward(x, train_flg=False)
            else:
                x = x + residual

            x = block.relu2.forward(x)

    # Global Average Pooling
    x = x.mean(axis=(2, 3))
    print(f"{'':>3} {'GlobalAvgPool':<32}{str(x.shape):<25}{'0':>10}", flush=True)

    # FC
    x = model.fc.forward(x)
    p = count_params(model.fc)
    print(f"{layer_idx:>2}. {'FC':<32}{str(x.shape):<25}{p:>10,}", flush=True)
    total_params += p

    print("=" * 75, flush=True)
    print(f"{'Total weight layers:':<60}{'20'}", flush=True)
    print(f"{'Total params:':<60}{total_params:,}", flush=True)
    print("=" * 75, flush=True)

model = ResNet20()
print_resnet20_summary(model, input_shape=(1, 3, 32, 32))

# 모델 복제 및 EMA 업데이트 함수

def clone_model(model):
    return copy.deepcopy(model)

def update_ema_model(ema_model, model, decay=0.999):
    for ema_layer, model_layer in zip(ema_model.layer1 + ema_model.layer2 + ema_model.layer3,
                                      model.layer1 + model.layer2 + model.layer3):
        for attr in ['conv1', 'conv2', 'shortcut']:
            if hasattr(model_layer, attr):
                model_conv = getattr(model_layer, attr)
                ema_conv = getattr(ema_layer, attr)
                ema_conv.W = decay * ema_conv.W + (1 - decay) * model_conv.W
                ema_conv.b = decay * ema_conv.b + (1 - decay) * model_conv.b

    ema_model.fc.W = decay * ema_model.fc.W + (1 - decay) * model.fc.W
    ema_model.fc.b = decay * ema_model.fc.b + (1 - decay) * model.fc.b


def cosine_annealing_with_warmup(epoch, total_epochs, base_lr, warmup_epochs=5):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * base_lr * (1 + math.cos(math.pi * progress))
    

import time
from common.optimizer import Adam
from common.functions import softmax

class Trainer:
    def __init__(self, model, model_name, train_data, val_data, test_data, epochs=20, batch_size=64, optimizer_name='sgd', lr=0.01):
        self.model = model
        self.model_name = model_name
        self.train_x, self.train_t = train_data
        self.val_x, self.val_t = val_data
        self.test_x, self.test_t = test_data
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_size = self.train_x.shape[0]
        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)
        self.max_iter = self.epochs * self.iter_per_epoch

        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        if optimizer_name == 'adam':
            self.optimizer = Adam(lr=lr)
        else:
            raise ValueError("Unsupported optimizer")
        self.ema_model = clone_model(self.model)

    def get_param_dict_and_grad(self):
        param_dict, grad_dict = {}, {}
        if hasattr(self.model.fc, 'W'):
            param_dict['fc_W'] = self.model.fc.W
            param_dict['fc_b'] = self.model.fc.b
            grad_dict['fc_W'] = self.model.fc.dW
            grad_dict['fc_b'] = self.model.fc.db

        idx = 0
        for layer in self.model.layer1 + self.model.layer2 + self.model.layer3:
            for attr in ['conv1', 'conv2', 'shortcut']:
                if hasattr(layer, attr):
                    conv = getattr(layer, attr)
                    param_dict[f'{idx}_W'] = conv.W
                    param_dict[f'{idx}_b'] = conv.b
                    grad_dict[f'{idx}_W'] = conv.dW
                    grad_dict[f'{idx}_b'] = conv.db
                    idx += 1
        return param_dict, grad_dict

    def loss_grad(self, x, t):
        y = self.model.forward(x, train_flg=True)
        batch_size = x.shape[0]
        if t.size == y.size:
            return (softmax(y) - t) / batch_size
        else:
            dx = softmax(y)
            dx[np.arange(batch_size), t] -= 1
            return dx / batch_size

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.train_x[batch_mask]
        x_batch = apply_augmentations(x_batch)
        t_batch = self.train_t[batch_mask]
        if t_batch.ndim == 1:
            t_batch = smooth_labels(t_batch, smoothing=0.1, num_classes=100)

        loss = self.model.loss(x_batch, t_batch)
        self.model.backward(self.loss_grad(x_batch, t_batch))

        if hasattr(self.model, 'clip_weights'):
            self.model.clip_weights(clip_value=1.0)

        params, grads = self.get_param_dict_and_grad()
        self.optimizer.update(params, grads)
        update_ema_model(self.ema_model, self.model)

        return loss

    def train(self):
        for epoch in range(self.epochs):
            self.optimizer.lr = cosine_annealing_with_warmup(epoch, self.epochs, base_lr=0.01)
            print(f"[Epoch {epoch + 1}]", flush=True)
            epoch_loss = 0
            start_time = time.time()

            for i in range(self.iter_per_epoch):
                loss = self.train_step()
                epoch_loss += loss
                if i % 10 == 0:
                    print(f"  Iter {i:3d}/{self.iter_per_epoch}: Loss {loss:.4f}", flush=True)

            avg_loss = epoch_loss / self.iter_per_epoch
            self.train_loss_list.append(avg_loss)

            train_acc = self.model.accuracy(self.train_x[:1000], self.train_t[:1000])
            val_acc = self.ema_model.accuracy(self.val_x, self.val_t)
            val_loss = self.batched_loss(self.val_x, self.val_t, batch_size=128)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(val_acc)
            self.val_loss_list.append(val_loss)

            elapsed = time.time() - start_time
            print(f"Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} (Time: {elapsed:.2f}s)\n", flush=True)

            if (epoch + 1) % 10 == 0:
                model_filename = f"{self.model_name}_epoch_{epoch+1}.pkl"
                self.save_model(model_filename)
                print(f">>> Saved model to {model_filename}", flush=True)

    def save_model(self, filename):
        params, _ = self.get_param_dict_and_grad()
        model_state = {k: v.copy() for k, v in params.items()}

        optimizer_state = {
            'lr': self.optimizer.lr,
            'beta1': getattr(self.optimizer, 'beta1', None),
            'beta2': getattr(self.optimizer, 'beta2', None),
            'eps': getattr(self.optimizer, 'eps', None),
            'm': getattr(self.optimizer, 'm', {}),
            'v': getattr(self.optimizer, 'v', {}),
            't': getattr(self.optimizer, 't', 0),
        }

        save_data = {
            'model': model_state,
            'optimizer': optimizer_state,
            'train_loss_list': self.train_loss_list,
            'train_acc_list': self.train_acc_list,
            'test_acc_list': self.test_acc_list,
            'val_loss_list': self.val_loss_list
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def save_log(self, filename='log.npz'):
        np.savez(filename,
                 loss=np.array(self.train_loss_list),
                 train_acc=np.array(self.train_acc_list),
                 test_acc=np.array(self.test_acc_list),
                 val_loss=np.array(self.val_loss_list))
        print(f"Log saved to {filename}", flush=True)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        params, _ = self.get_param_dict_and_grad()
        for k in params:
            if k in state['model']:
                params[k][...] = state['model'][k]
            else:
                print(f"[WARN] Key {k} not found in checkpoint!", flush=True)

        opt = state['optimizer']
        self.optimizer.lr = opt['lr']
        self.optimizer.beta1 = opt['beta1']
        self.optimizer.beta2 = opt['beta2']
        self.optimizer.eps = opt['eps']
        self.optimizer.m = opt['m']
        self.optimizer.v = opt['v']
        self.optimizer.t = opt['t']

        # 복원된 로그
        self.train_loss_list = state.get('train_loss_list', [])
        self.train_acc_list = state.get('train_acc_list', [])
        self.test_acc_list = state.get('test_acc_list', [])
        self.val_loss_list = state.get('val_loss_list', [])

    def batched_loss(self, x, t, batch_size=128):
        total_loss = 0.0
        total_count = 0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            t_batch = t[i:i+batch_size]
            loss = self.model.loss(x_batch, t_batch)
            total_loss += loss * len(x_batch)
            total_count += len(x_batch)
        return total_loss / total_count

# === 자동 튜닝 실험 루프 추가 ===

import numpy as np

(x_train, y_train_fine, y_train_coarse),
(x_val, y_val_fine, y_val_coarse),
(x_test, y_test_fine, y_test_coarse) = load_cifar100()

# fine label 기준으로 사용
train_data = (x_train, y_train_fine)
val_data = (x_val, y_val_fine)
test_data = (x_test, y_test_fine)

# 고정된 두 실험 조건
model_configs = [
    {"lr": 0.01, "batch_size": 64},
    {"lr": 0.001, "batch_size": 32},
]

# 튜닝 단계별 하이퍼파라미터 구성
skip_probs = [0.05, 0.1, 0.2, 0.3]
smoothing_values = [0.05, 0.1, 0.15]
ema_warmup_configs = [
    {"ema_decay": 0.995, "warmup_epochs": 3},
    {"ema_decay": 0.9999, "warmup_epochs": 7}
]

# 각 모델 설정마다 총 9회 실험 (4 + 3 + 2)
def run_experiments():
    for model_id, cfg in enumerate(model_configs):
        print(f"=== [모델 설정 {model_id+1}] LR={cfg['lr']}, BS={cfg['batch_size']} ===")
        # 1단계: skip_prob 튜닝
        for skip in skip_probs:
            model = ResNet20()
            trainer = Trainer(
                model=model,
                model_name=f"ResNet20_cfg{model_id+1}_skip{skip}",
                train_data=(x_train, y_train_fine),
                val_data=(x_val, y_val_fine),
                test_data=(x_test, y_test_fine),
                epochs=10,
                batch_size=cfg["batch_size"],
                optimizer_name="adam",
                lr=cfg["lr"]
            )
            trainer.skip_prob = skip
            trainer.smoothing = 0.1
            trainer.ema_decay = 0.999
            trainer.warmup_epochs = 5
            trainer.train()
            trainer.save_log(f"log_cfg{model_id+1}_skip{skip}.npz")
            trainer.save_model(f"model_cfg{model_id+1}_skip{skip}.pkl")

        # Best skip_prob = 0.1 고정 (예시), smoothing 튜닝
        for smooth in smoothing_values:
            model = ResNet20()
            trainer = Trainer(
                model=model,
                model_name=f"ResNet20_cfg{model_id+1}_smooth{smooth}",
                train_data=(x_train, y_train_fine),
                val_data=(x_val, y_val_fine),
                test_data=(x_test, y_test_fine),
                epochs=10,
                batch_size=cfg["batch_size"],
                optimizer_name="adam",  
                lr=cfg["lr"]
            )
            trainer.skip_prob = 0.1
            trainer.smoothing = smooth
            trainer.ema_decay = 0.999
            trainer.warmup_epochs = 5
            trainer.train()
            trainer.save_log(f"log_cfg{model_id+1}_smooth{smooth}.npz")
            trainer.save_model(f"model_cfg{model_id+1}_smooth{smooth}.pkl")
            

        # EMA/Warmup 튜닝
        for ema_cfg in ema_warmup_configs:
            model = ResNet20()
            trainer = Trainer(
                model=model,
                model_name=f"ResNet20_cfg{model_id+1}_ema{ema_cfg['ema_decay']}_warmup{ema_cfg['warmup_epochs']}",
                train_data=(x_train, y_train_fine),
                val_data=(x_val, y_val_fine),
                test_data=(x_test, y_test_fine),
                epochs=10,
                batch_size=cfg["batch_size"],
                optimizer_name="adam",
                lr=cfg["lr"]
            )
            trainer.skip_prob = 0.1
            trainer.smoothing = 0.1
            trainer.ema_decay = ema_cfg["ema_decay"]
            trainer.warmup_epochs = ema_cfg["warmup_epochs"]
            trainer.train()
            trainer.save_log(f"log_cfg{model_id+1}_ema{ema_cfg['ema_decay']}_warmup{ema_cfg['warmup_epochs']}.npz")
            trainer.save_model(f"model_cfg{model_id+1}_ema{ema_cfg['ema_decay']}_warmup{ema_cfg['warmup_epochs']}.pkl")

if __name__ == "__main__":
    run_experiments()
    
