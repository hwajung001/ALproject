
import numpy as np
import pickle
import os
import urllib.request
import tarfile
import matplotlib.pyplot as plt
from common.util import shuffle_dataset

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

# 메타데이터 로딩 (fine, coarse label names 포함)
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


# 데이터 다운로드 및 로딩
download_cifar100()
(x_train, y_train_fine, y_train_coarse), (x_val, y_val_fine, y_val_coarse), (x_test, y_test_fine, y_test_coarse) = load_cifar100()
meta = load_meta()

# 데이터셋 정보 출력
print("CIFAR-100 Dataset Loaded!")
print(f"Train X: {x_train.shape}, Fine Y: {y_train_fine.shape}, Coarse Y: {y_train_coarse.shape}")
print(f"Val   X: {x_val.shape}, Fine Y: {y_val_fine.shape}, Coarse Y: {y_val_coarse.shape}")
print(f"Test  X: {x_test.shape}, Fine Y: {y_test_fine.shape}, Coarse Y: {y_test_coarse.shape}")

def visualize_distribution(labels, label_names=None, title="Label Distribution", filename="label_distribution.png"):
    counts = np.bincount(labels)
    plt.figure(figsize=(14, 4))
    plt.bar(range(len(counts)), counts)
    if label_names:
        plt.xticks(ticks=np.arange(len(label_names)), labels=label_names, rotation=90)
    plt.title(title)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 레이블 분포 시각화
visualize_distribution(
    y_train_fine, 
    meta['fine_label_names'], 
    "Fine Label Distribution (Train)",
    filename="fine_label_distribution_train.png"
)

visualize_distribution(
    y_train_coarse, 
    meta['coarse_label_names'], 
    "Coarse Label Distribution (Train)",
    filename="coarse_label_distribution_train.png"
)

import matplotlib.pyplot as plt

def show_example_images(images, fine_labels, coarse_labels, fine_names, coarse_names, num_samples=8, filename="example_images.png"):
    plt.figure(figsize=(14, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = images[i].transpose(1, 2, 0)  # (3, 32, 32) → (32, 32, 3)
        mean = np.array([0.507, 0.487, 0.441])
        std = np.array([0.267, 0.256, 0.276])
        img = img.transpose(2, 0, 1)  # → (3, 32, 32)
        img = img * std[:, None, None] + mean[:, None, None]
        img = img.transpose(1, 2, 0)  # → (32, 32, 3)

        plt.imshow(np.clip(img, 0, 1))
        plt.axis('off')
        coarse = coarse_names[coarse_labels[i]]
        fine = fine_names[fine_labels[i]]
        plt.title(f"{coarse}\n({fine})", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 메타 정보 로드 및 예시 시각화
meta = load_meta()
show_example_images(
    images=x_train,
    fine_labels=y_train_fine,
    coarse_labels=y_train_coarse,
    fine_names=meta['fine_label_names'],
    coarse_names=meta['coarse_label_names'],
    num_samples=5,
    filename="example_cifar100_train_samples.png"
)

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

class Dropout:
    def __init__(self, dropout_ratio=0.3):
        self.dropout_ratio = dropout_ratio

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class ResidualBlock:
    def __init__(self, in_channels, out_channels, stride=1, dropout_ratio=0.3):
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
        self.dropout = Dropout(dropout_ratio)  

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

    def forward(self, x, train_flg=True):
        self.x = x

        out = self.conv1.forward(x)
        out = self.bn1.forward(out, train_flg)
        out = self.relu1.forward(out)
        out = self.dropout.forward(out, train_flg)  

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

        dmain = self.dropout.backward(dmain)  
        dmain = self.relu1.backward(dmain)
        dmain = self.bn1.backward(dmain)
        dmain = self.conv1.backward(dmain)

        if not self.equal_in_out:
            dshortcut = self.bn_shortcut.backward(dshortcut)
            dshortcut = self.shortcut.backward(dshortcut)

        dx = dmain + dshortcut
        return dx
    
class ResNet20:
    def __init__(self, input_dim=(3, 32, 32), num_classes=100, dropout_ratio=0.3):
        self.params = []
        self.trainable_layers = []
        self.dropout_ratio = dropout_ratio

        self.conv1 = Convolution(
            W=np.random.randn(16, 3, 3, 3) * np.sqrt(2. / 3),
            b=np.zeros(16),
            stride=1,
            pad=1
        )
        self.bn1 = BatchNormalization(gamma=np.ones(16), beta=np.zeros(16))
        self.relu1 = Relu()

        self.layer1 = [ResidualBlock(16, 16, stride=1, dropout_ratio=self.dropout_ratio) for _ in range(3)]
        self.layer2 = [ResidualBlock(16 if i == 0 else 32, 32, stride=2 if i == 0 else 1, dropout_ratio=self.dropout_ratio) for i in range(3)]
        self.layer3 = [ResidualBlock(32 if i == 0 else 64, 64, stride=2 if i == 0 else 1, dropout_ratio=self.dropout_ratio) for i in range(3)]

        self.fc = Affine(W=np.random.randn(64, num_classes) * np.sqrt(2. / 64), b=np.zeros(num_classes))

    def clip_weights(self, clip_value=1.0):
        self.conv1.W = np.clip(self.conv1.W, -clip_value, clip_value)
        self.fc.W = np.clip(self.fc.W, -clip_value, clip_value)
        for block in self.layer1 + self.layer2 + self.layer3:
            block.conv1.W = np.clip(block.conv1.W, -clip_value, clip_value)
            block.conv2.W = np.clip(block.conv2.W, -clip_value, clip_value)
            if not block.equal_in_out:
                block.shortcut.W = np.clip(block.shortcut.W, -clip_value, clip_value)

    def forward(self, x, train_flg=True):
        self.input = x

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
        out = out.mean(axis=(2, 3))  # global average pooling

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

            # Dropout (표시는 되지만 파라미터는 없음)
            x = block.dropout.forward(x, train_flg=False)
            print(f"{'':>3} {'Dropout':<32}{str(x.shape):<25}{'0':>10}", flush=True)

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

# ResNet-20 모델 생성 및 요약 출력
model = ResNet20()
print_resnet20_summary(model, input_shape=(1, 3, 32, 32))


import numpy as np
import time
import pickle
from common.functions import softmax, cross_entropy_error

class Adam:
    """Adam optimizer with optional weight decay (L2 regularization)."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay  
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            if self.weight_decay > 0:
                grads[key] += self.weight_decay * params[key]

            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


import numpy as np
import time
import pickle
from common.functions import softmax
from common.util import shuffle_dataset

class Trainer:
    def __init__(self, model, model_name, train_data, val_data, test_data,
                 epochs=20, batch_size=64, optimizer_name='adam', lr=0.01,
                 weight_decay=0.0, patience=5):
        self.model = model
        self.model_name = model_name
        self.train_x, self.train_t = train_data
        self.val_x, self.val_t = val_data
        self.test_x, self.test_t = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience

        self.train_size = self.train_x.shape[0]
        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)

        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        self.best_val_loss = float('inf')
        self.wait = 0

        if optimizer_name == 'adam':
            self.optimizer = Adam(lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Unsupported optimizer")

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
        t_batch = self.train_t[batch_mask]

        loss = self.model.loss(x_batch, t_batch)
        self.model.backward(self.loss_grad(x_batch, t_batch))

        if hasattr(self.model, 'clip_weights'):
            self.model.clip_weights(clip_value=1.0)

        params, grads = self.get_param_dict_and_grad()
        self.optimizer.update(params, grads)

        return loss

    def evaluate(self, x, t):
        loss = self.model.loss(x, t)
        acc = self.model.accuracy(x, t)
        return loss, acc

    def get_wrong_indices(self, x, t):
        y_pred = np.argmax(self.model.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.where(y_pred != t)[0]

    def train(self):
        for epoch in range(self.epochs):
            print(f"[Epoch {epoch + 1}]", flush=True)
            self.train_x, self.train_t = shuffle_dataset(self.train_x, self.train_t)

            epoch_loss = 0
            start_time = time.time()

            for i in range(self.iter_per_epoch):
                loss = self.train_step()
                epoch_loss += loss
                if i % 10 == 0:
                    print(f"  Iter {i:3d}/{self.iter_per_epoch}: Train Loss {loss:.4f}", flush=True)

            avg_loss = epoch_loss / self.iter_per_epoch
            val_loss, val_acc = self.evaluate(self.val_x, self.val_t)
            train_acc = self.model.accuracy(self.train_x[:1000], self.train_t[:1000])
            test_acc = self.model.accuracy(self.test_x, self.test_t)

            self.train_loss_list.append(avg_loss)
            self.val_loss_list.append(val_loss)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            elapsed = time.time() - start_time
            print(f"Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Test acc: {test_acc:.4f} (Time: {elapsed:.2f}s)", flush=True)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
                self.save_model(f"{self.model_name}_best.pkl")
                print(">>> Model improved. Saved as best model.", flush=True)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(">>> Early stopping triggered.", flush=True)
                    break

            wrong_idx = self.get_wrong_indices(self.val_x, self.val_t)
            np.save(f"{self.model_name}_wrong_indices_epoch_{epoch+1}.npy", wrong_idx)

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
            'val_loss_list': self.val_loss_list,
            'train_acc_list': self.train_acc_list,
            'test_acc_list': self.test_acc_list
        }

        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)

    def save_log(self, filename='log.npz'):
        np.savez(filename,
                 loss=np.array(self.train_loss_list),
                 val_loss=np.array(self.val_loss_list),
                 train_acc=np.array(self.train_acc_list),
                 test_acc=np.array(self.test_acc_list))
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
        self.val_loss_list = state.get['val_loss_list', []] 
        self.train_acc_list = state.get('train_acc_list', [])
        self.test_acc_list = state.get('test_acc_list', [])

    def load_best_and_evaluate(self):
        filename = f"{self.model_name}_best.pkl"
        if not os.path.exists(filename):
            print(f"[ERROR] Best model file not found: {filename}")
            return

        self.load_model(filename)
        test_loss, test_acc = self.evaluate(self.test_x, self.test_t)
        print(f"[BEST MODEL EVAL] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        return test_loss, test_acc  
    

def main():
    weight_decays = [1e-4, 5e-4]
    dropout_ratios = [0.3, 0.5]
    results = []

    for wd in weight_decays:
        for dr in dropout_ratios:
            model_name = f"ResNet20_wd{wd}_dropout{dr}_0.01_64"
            print(f"\n[Training {model_name}]", flush=True)

            model = ResNet20(dropout_ratio=dr)  
            trainer = Trainer(
                model=model,
                model_name=model_name,
                train_data=(x_train, y_train_fine),
                val_data=(x_val, y_val_fine),
                test_data=(x_test, y_test_fine),
                epochs=20,
                batch_size=64,
                optimizer_name='adam',
                lr=0.01,
                weight_decay=wd,
                patience=5
            )

            trainer.train()
            trainer.save_log(f"{model_name}_log.npz")
            trainer.save_model(f"{model_name}_epoch20.pkl")

            trainer.load_best_and_evaluate()
            trainer.save_confusion_matrix(x_val, y_val_fine, label_names=meta['fine_label_names'],
                                          save_path=f"{model_name}_confusion.png")
            trainer.visualize_misclassified(x_val, y_val_fine,
                                            label_names=meta['fine_label_names'],
                                            save_path=f"{model_name}_misclassified.png")

            final_acc = trainer.test_acc_list[-1]
            results.append((wd, dr, final_acc))

    print("\n[하이퍼파라미터 튜닝 결과 요약]")
    for wd, dr, acc in results:
        print(f"weight_decay={wd}, dropout={dr} → test_acc={acc:.4f}", flush=True)
    
    for wd in weight_decays:
        for dr in dropout_ratios:
            model_name = f"ResNet20_wd{wd}_dropout{dr}_0.001_32"
            print(f"\n[Training {model_name}]", flush=True)

            model = ResNet20(dropout_ratio=dr)  
            trainer = Trainer(
                model=model,
                model_name=model_name,
                train_data=(x_train, y_train_fine),
                val_data=(x_val, y_val_fine),
                test_data=(x_test, y_test_fine),
                epochs=20,
                batch_size=32,
                optimizer_name='adam',
                lr=0.001,
                weight_decay=wd,
                patience=5
            )

            trainer.train()
            trainer.save_log(f"{model_name}_log.npz")
            trainer.save_model(f"{model_name}_epoch20.pkl")

            trainer.load_best_and_evaluate()
            trainer.save_confusion_matrix(x_val, y_val_fine, label_names=meta['fine_label_names'],
                                          save_path=f"{model_name}_confusion.png")
            trainer.visualize_misclassified(x_val, y_val_fine,
                                            label_names=meta['fine_label_names'],
                                            save_path=f"{model_name}_misclassified.png")

            final_acc = trainer.test_acc_list[-1]
            results.append((wd, dr, final_acc))

    print("\n[하이퍼파라미터 튜닝 결과 요약]")
    for wd, dr, acc in results:
        print(f"weight_decay={wd}, dropout={dr} → test_acc={acc:.4f}", flush=True)

if __name__ == "__main__":
    main()