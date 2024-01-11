import numpy as np
import torch
import torch.nn as nn
import common.dqn_model

a = torch.randint(0, 10, [4, 34], dtype=torch.float32)
a = a.unsqueeze(0).unsqueeze(0)
print(a)
print(a.shape)

# first layer: conv
# conv2d = nn.Conv2d(1, 1, (2, 2), stride=1, padding='same', bias=False)
conv2d = nn.Conv2d(1, 1, (2, 2))
print(conv2d.weight)
print(conv2d.bias)

# kernels = torch.tensor([[[[1, 0], [2, 1]]]], dtype=torch.float32)
# conv2d.weight = nn.Parameter(kernels, requires_grad=False)
# print(conv2d.weight)
a = conv2d(a)
print(a)
print(a.shape)

# second layer:pool
pool = nn.MaxPool2d((2, 2))
a = pool(a)
print(a)
print(a.shape)

flatten = nn.Flatten(start_dim=0)
a = flatten(a)
print(a)
print(a.shape)

linear = nn.Linear(16, 34)
a = linear(a)
print(a)
print(a.shape)

b = (4, 37)
c = np.zeros(b, dtype=np.float32)
print(c)

if __name__ == '__main__':
    arr = np.zeros((4, 34), dtype=np.float32)
    obs = torch.tensor(arr)
    print(obs.dtype)

    a = np.zeros(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    a = a[np.newaxis, :]
    b = b[np.newaxis, :]
    print(a.shape)
    print(np.concatenate((a, b), axis=0))
