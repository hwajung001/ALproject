#!/usr/bin/env python
# coding: utf-8

# 데이터셋 로드 및 전처리
import numpy as np
import pickle
import os
import urllib.request
import tarfile

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

def load_cifar100(data_dir="./cifar-100-python"):
    def load_batch(filename):
        with open(filename, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
            data = dict[b'data']
            labels = dict[b'fine_labels']
            coarse_labels = dict[b'coarse_labels']
            return data, labels, coarse_labels

    x_train, y_train, y_train_coarse = load_batch(os.path.join(data_dir, "train"))
    x_test, y_test, y_test_coarse = load_batch(os.path.join(data_dir, "test"))

    x_train = x_train.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train_coarse = np.array(y_train_coarse)
    y_test_coarse = np.array(y_test_coarse)

    val_size = int(0.1 * len(x_train))
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), (y_train_coarse, y_test_coarse)

# 데이터 다운로드 및 로딩
download_cifar100()
(x_train, y_train), (x_val, y_val), (x_test, y_test), (y_train_coarse, y_test_coarse) = load_cifar100()

# 데이터셋 정보 출력
print(" CIFAR-100 Dataset Loaded!")
print(f"Train X shape: {x_train.shape}, Train Y shape: {y_train.shape}")
print(f"Val   X shape: {x_val.shape}, Val   Y shape: {y_val.shape}")
print(f"Test  X shape: {x_test.shape}, Test  Y shape: {y_test.shape}")
print(f"Coarse Labels - Train: {y_train_coarse.shape}, Test: {y_test_coarse.shape}")

import numpy as np
from common.layers import Convolution, Affine, Relu, BatchNormalization
from common.functions import softmax, cross_entropy_error
from common.util import im2col, col2im

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

    def forward(self, x, train_flg=True):
        self.x = x

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


import numpy as np
import time
import pickle
from common.optimizer import SGD, Adam
from common.functions import softmax, cross_entropy_error

class Trainer:
    def __init__(self, model, train_data, test_data, epochs=20, batch_size=64, optimizer_name='sgd', lr=0.01):
        self.model = model
        self.train_x, self.train_t = train_data
        self.test_x, self.test_t = test_data
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_size = self.train_x.shape[0]
        self.iter_per_epoch = max(self.train_size // self.batch_size, 1)
        self.max_iter = self.epochs * self.iter_per_epoch

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

        # prepare optimizer
        if optimizer_name == 'sgd':
            self.optimizer = SGD(lr=lr)
        elif optimizer_name == 'adam':
            self.optimizer = Adam(lr=lr)
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

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.train_x[batch_mask]
        t_batch = self.train_t[batch_mask]

        loss = self.model.loss(x_batch, t_batch)
        self.model.backward(self.loss_grad(x_batch, t_batch))

        params, grads = self.get_param_dict_and_grad()
        self.optimizer.update(params, grads)

        return loss

    def loss_grad(self, x, t):
        y = self.model.forward(x, train_flg=True)
        batch_size = x.shape[0]
        if t.size == y.size:
            return (softmax(y) - t) / batch_size
        else:
            dx = softmax(y)
            dx[np.arange(batch_size), t] -= 1
            return dx / batch_size

    def train(self):
        for epoch in range(self.epochs):
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
            test_acc = self.model.accuracy(self.test_x, self.test_t)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            elapsed = time.time() - start_time
            print(f"Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f} (Time: {elapsed:.2f}s)\n", flush=True)

            # 5 에폭마다 모델 저장
            if (epoch + 1) % 5 == 0:
                model_filename = f"checkpoint_epoch_{epoch+1}.pkl"
                self.save_model(model_filename)
                print(f">>> Saved model to {model_filename}\n", flush=True)

    def save_log(self, filename='log.npz'):
        np.savez(filename, loss=self.train_loss_list, train_acc=self.train_acc_list, test_acc=self.test_acc_list)

    def save_model(self, filename='model_and_opt.pkl'):
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

        with open(filename, 'wb') as f:
            pickle.dump({'model': model_state, 'optimizer': optimizer_state}, f)

    def load_model(self, filename='model_and_opt.pkl'):
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        params, _ = self.get_param_dict_and_grad()
        for k in params:
            if k in state['model']:
                params[k][...] = state['model'][k]
            else:
                print(f"[WARN] Key {k} not found in checkpoint!")

        opt = state['optimizer']
        self.optimizer.lr = opt['lr']
        self.optimizer.beta1 = opt['beta1']
        self.optimizer.beta2 = opt['beta2']
        self.optimizer.eps = opt['eps']
        self.optimizer.m = opt['m']
        self.optimizer.v = opt['v']
        self.optimizer.t = opt['t']


if __name__ == "__main__":
    # 모델 및 트레이너 초기화
    model = ResNet20()
    trainer = Trainer(
        model=model,
        train_data=(x_train, y_train),
        test_data=(x_val, y_val),   
        epochs=30,
        batch_size=32,
        optimizer_name='adam',
        lr=0.005
    )

    # 학습 시작
    trainer.train()

    # 최종 모델 및 로그 저장
    trainer.save_log("final_log.npz")
