from ppq.core import CUDA
from ppq import ppq_tensor_round, RoundingPolicy
import torch
from math import sqrt

EXECUTING_DEVICE = 'cuda'
TEST_TIMES = 128
Q_MIN, Q_MAX = -128, 127
ROUNDING_POLICY = RoundingPolicy.ROUND_HALF_EVEN

# Test tensorwise quantize
for i in range(TEST_TIMES):
    t = torch.rand(size=[128, 3, 224, 224]).to(EXECUTING_DEVICE)
    s = torch.rand(size=[1]).to(EXECUTING_DEVICE)
    o = torch.randint(low=0, high=255, size=[1]).float().to(EXECUTING_DEVICE)

    # torch quantize
    qt = ppq_tensor_round(t / s, policy=ROUNDING_POLICY) + o
    qt = qt.clip(Q_MIN, Q_MAX)
    qt = (qt - o) * s

    # t = t.int()
    cuda_qt = CUDA.LinearQuantize_T(
        t, s, o, Q_MIN, Q_MAX, ROUNDING_POLICY.value)

    assert torch.abs(cuda_qt - qt).max().item() < 1e-6

# Test channelwise quantize
for i in range(TEST_TIMES):
    t = torch.rand(size=[128, 3, 224, 224]).to(EXECUTING_DEVICE)
    s = torch.rand(size=[3]).to(EXECUTING_DEVICE)
    o = torch.randint(low=0, high=255, size=[3]).float().to(EXECUTING_DEVICE)
    c = 1

    # torch quantize
    shape = [1 if axis != c else -1 for axis in range(t.ndim)]
    s, o = s.view(shape), o.view(shape)
    qt = ppq_tensor_round(t / s, policy=ROUNDING_POLICY) + o
    qt = qt.clip(Q_MIN, Q_MAX)
    qt = (qt - o) * s

    # t = t.int()
    cuda_qt = CUDA.LinearQuantize_C(
        t, s, o, c, Q_MIN, Q_MAX, ROUNDING_POLICY.value)
    assert torch.abs(cuda_qt - qt).max().item() < 1e-6

# Test Histogram
for i in range(TEST_TIMES):
    t = torch.rand(size=[128, 3, 224, 224]).to(EXECUTING_DEVICE)
    s = 0.01

    # torch hist
    hist = torch.histc(torch.abs(t), bins=50, min=0, max=0.5)

    # t = t.int()
    cuda_hist = torch.zeros(size=[50]).to(EXECUTING_DEVICE).int()
    cuda_hist = CUDA.Histogram_T(t, cuda_hist, s)
    assert torch.abs(hist - cuda_hist).max().item() < 100


# Test Backwards
t = torch.Tensor([112,75,3,5,-5,6,7,8,9,-25]*100).to(EXECUTING_DEVICE)
s = torch.Tensor([2]).to(EXECUTING_DEVICE).float()
o = torch.Tensor([0]).to(EXECUTING_DEVICE).float()
qt = CUDA.LinearQuantize_T(t, s, o, Q_MIN, Q_MAX)
g = torch.zeros_like(t) + 1
t.requires_grad = True
s.requires_grad = True
o.requires_grad = True

# print(qt)
for i in range(50):
    gs_eta = ((qt - t) / s * g).sum()
    ga, gs, go = CUDA.LinearQuantize_T_B(t, qt, s, o, g, Q_MIN, Q_MAX)

    # print(t)
    # print(ga)
    print(gs - gs_eta)
    print(gs)
    print(go)

t = torch.rand(size=[3,3,224,224]).to(EXECUTING_DEVICE)
s = torch.Tensor([1, 1, 1]).float().to(EXECUTING_DEVICE)
o = torch.zeros(size=[3]).to(EXECUTING_DEVICE).float()

# t = torch.Tensor([[112,75,3,5,-5],[5,7,8,9,-25],[600,7,8,9,-25],[600,7,8,9,-25],[600,7,8,9,-25],[600,7,8,9,-25]]).to(EXECUTING_DEVICE)
# s = torch.Tensor([2, 3, 2, 2, 2, 2]).to(EXECUTING_DEVICE).float()
# o = torch.Tensor([0, 0, 0, 0, 0, 0]).to(EXECUTING_DEVICE).float()
qt = CUDA.LinearQuantize_C(t, s, o, 1, Q_MIN, Q_MAX)
g = torch.zeros_like(t) + 1
t.requires_grad = True
s.requires_grad = True
o.requires_grad = True

# print(qt)
for i in range(1):
    ga, gs, go = CUDA.LinearQuantize_C_B(t, qt, s, o, g, Q_MIN, Q_MAX, 1)

    # print(qt)
    # print(ga)
    print(gs)
    print(go)


# Test Clip
t = torch.Tensor([-1, -2, -3, 1,2,3,4,5,6,7,8,9,10]).to(EXECUTING_DEVICE)
r = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,0]).to(EXECUTING_DEVICE)
l = torch.Tensor([2]).to(EXECUTING_DEVICE)
t = CUDA.TensorClip_T(t, r, l)
print(t)

t = torch.Tensor([[1,2,3], [4,5,6], [-1,-2,-3]]).to(EXECUTING_DEVICE)
r = torch.Tensor([[0,0,0], [4,5,6], [0,0,0]]).to(EXECUTING_DEVICE)
l = torch.Tensor([1, 2, 2]).to(EXECUTING_DEVICE)
t = CUDA.TensorClip_C(t, r, l, 0)
print(t)

for i in range(1000):
    t = torch.rand(size=[10000]).to(EXECUTING_DEVICE)
    r = torch.rand(size=[10000]).to(EXECUTING_DEVICE)
    l = torch.rand(size=[1]).to(EXECUTING_DEVICE)

    out = r + (t - r).clip(min=-l.item(), max=l.item())
    print(torch.abs(CUDA.TensorClip_T(t, r, l) - out).sum())


# Test Rounding loss
for i in range(100):
    t = torch.rand(size=[3,3,224,224]).to(EXECUTING_DEVICE)
    s = torch.Tensor([1]).float().to(EXECUTING_DEVICE)
    o = torch.zeros(size=[1]).to(EXECUTING_DEVICE).float()

    t.requires_grad = True

    qt = CUDA.LinearQuantize_T(t, s, o, Q_MIN, Q_MAX)
    rl_torch = torch.sum(torch.abs(qt - t)) / sqrt(qt.numel())
    rl_torch.backward()
    rl = CUDA.RoundingLoss_LT(t, s, o, Q_MIN, Q_MAX)
    rl_b = CUDA.RoundingLoss_LT_B(t, torch.Tensor([1]).float().to(EXECUTING_DEVICE), s, o, Q_MIN, Q_MAX)
    print(rl_torch - rl)
    print(rl_b - t.grad)


# Test Rounding loss
for i in range(100):
    t = torch.rand(size=[3,3,224,224]).to(EXECUTING_DEVICE)
    s = torch.Tensor([1, 2, 3]).float().to(EXECUTING_DEVICE)
    o = torch.zeros(size=[3]).to(EXECUTING_DEVICE).float()

    t.requires_grad = True

    qt = CUDA.LinearQuantize_C(t, s, o, 1, Q_MIN, Q_MAX)
    rl_torch = torch.sum(torch.abs(qt - t)) / sqrt(qt.numel())
    rl_torch.backward()
    rl = CUDA.RoundingLoss_LC(t, s, o, 1, Q_MIN, Q_MAX)
    rl_b = CUDA.RoundingLoss_LC_B(t, torch.Tensor([1]).float().to(EXECUTING_DEVICE), s, o, 1, Q_MIN, Q_MAX)
    print(rl_torch - rl)
    print(rl_b - t.grad)
