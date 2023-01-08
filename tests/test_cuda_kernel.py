from ppq.core import CUDA
from ppq.core.ffi import CUDA_COMPLIER
from ppq import ppq_tensor_round, RoundingPolicy, ppq_numerical_round, torch_snr_error
from typing import List
import torch
from math import sqrt
from tqdm import tqdm

EXECUTING_DEVICE = 'cuda'
TEST_TIMES = 128
Q_MIN, Q_MAX = 0, 255
ROUNDING_POLICY = RoundingPolicy.ROUND_HALF_EVEN

CUDA_COMPLIER.complie()


def __TEST_QUANTIZE_LT__(size: List[int], iterations: int, sym: bool):
    for _ in tqdm(range(iterations), desc='QUANTIZE LT TESTING...'):
        t = torch.rand(size=size).cuda() * 32
        s = torch.rand(size=[1]).cuda()
        if sym:
            o = torch.zeros(size=[1]).cuda()
        else:
            o = torch.randint(low=0, high=255, size=[1]).float().cuda()

        # torch quantize
        qt = ppq_tensor_round(t / s, policy=ROUNDING_POLICY) + o
        qt = qt.clip(Q_MIN, Q_MAX)
        qt = (qt - o) * s

        # t = t.int()
        cuda_qt = CUDA.LinearQuantize_T(
            t, s, o, Q_MIN, Q_MAX, ROUNDING_POLICY.value)

        diff = qt - cuda_qt
        if diff.abs().max() != 0:
            raise Exception('Test Failed.')


def __TEST_QUANTIZE_LC__(size: List[int], iterations: int, sym: bool, c: int=1):
    for _ in tqdm(range(iterations), desc='QUANTIZE LC TESTING...'):
        num_of_channel = size[c]
        t = torch.rand(size=size).cuda() * 32
        s = torch.rand(size=[num_of_channel]).cuda()
        if sym:
            o = torch.zeros(size=[num_of_channel]).cuda()
        else:
            o = torch.randint(low=0, high=255, size=[num_of_channel]).float().cuda()

        shape = [1 if axis != c else -1 for axis in range(t.ndim)]
        s, o = s.view(shape), o.view(shape)
        qt = ppq_tensor_round(t / s, policy=ROUNDING_POLICY) + o
        qt = qt.clip(Q_MIN, Q_MAX)
        qt = (qt - o) * s

        # t = t.int()
        cuda_qt = CUDA.LinearQuantize_C(
            t, s, o, c, Q_MIN, Q_MAX, ROUNDING_POLICY.value)

        diff = qt - cuda_qt
        if diff.abs().max() != 0:
            print(diff)
            raise Exception('Test Failed.')


def __TEST_QUANTIZE_LT_B__(size: List[int], iterations: int, sym: bool):
    def ref_grad_func(value: torch.Tensor, dy: torch.Tensor, 
                      scale: torch.Tensor, offset: torch.Tensor):
        
        qt = ppq_tensor_round(value / scale, policy=ROUNDING_POLICY) + offset
        clipped_qt = qt.clip(Q_MIN, Q_MAX)
        
        dx = torch.where(clipped_qt != qt, torch.zeros_like(dy), dy)
        ds = torch.where(clipped_qt == qt, (((qt - offset) * scale) - value) * dy / scale, torch.zeros_like(dy))
        ds += torch.where(qt > Q_MAX, (Q_MAX - offset) * dy, torch.zeros_like(dy))
        ds += torch.where(qt < Q_MIN, (Q_MIN - offset) * dy, torch.zeros_like(dy))
        ds = ds.sum() / sqrt(value.numel() * (Q_MAX - Q_MIN))
        return dx, ds

    for _ in tqdm(range(iterations), desc='QUANTIZE LT_B TESTING...'):
        t = torch.rand(size=size).cuda() * 50
        s = torch.rand(size=[1]).cuda()
        if sym:
            o = torch.zeros(size=[1]).cuda()
        else:
            o = torch.randint(low=0, high=255, size=[1]).float().cuda()
        dy = torch.rand_like(t)
        
        cuda_grad_x, cuda_grad_s = CUDA.LinearQuantize_T_B(
            t, s, o, dy, Q_MIN, Q_MAX, ROUNDING_POLICY.value)
        
        ref_grad_x, ref_grad_s = ref_grad_func(t, dy, s, o)
        diff = ref_grad_x.flatten() - cuda_grad_x.flatten()
        if diff.abs().max() != 0:
            raise Exception('Test Failed.')

        snr = torch_snr_error(cuda_grad_s.reshape([1, -1]), ref_grad_s.reshape([1, -1])).item()
        if snr > 1e-3:
            print(f'[Warning]. Floating Precision Error. {snr:.4f} {cuda_grad_s.item():4f} {ref_grad_s.item():4f}')
            if cuda_grad_s.item() > 1:
                raise Exception('Test Failed.')
        
def __TEST_QUANTIZE_LC_B__(size: List[int], iterations: int, sym: bool, c: int=1):
    def ref_grad_func(value: torch.Tensor, dy: torch.Tensor, 
                      scale: torch.Tensor, offset: torch.Tensor, c: int):

        shape = [1 if axis != c else -1 for axis in range(t.ndim)]
        scale = scale.view(shape)
        offset = offset.view(shape)
        
        qt = ppq_tensor_round(value / scale, policy=ROUNDING_POLICY) + offset
        clipped_qt = qt.clip(Q_MIN, Q_MAX)

        dx = torch.where(clipped_qt != qt, torch.zeros_like(dy), dy)
        ds = torch.where(clipped_qt == qt, (((qt - offset) * scale) - value) * dy / scale, torch.zeros_like(dy))
        ds += torch.where(qt > Q_MAX, (Q_MAX - offset) * dy, torch.zeros_like(dy))
        ds += torch.where(qt < Q_MIN, (Q_MIN - offset) * dy, torch.zeros_like(dy))
        ds = ds.transpose(0, c).flatten(1).sum(dim=-1) / sqrt(value.numel() * (Q_MAX - Q_MIN))
        return dx, ds

    for _ in tqdm(range(iterations), desc='QUANTIZE LC_B TESTING...'):
        num_of_channel = size[c]
        t = torch.rand(size=size).cuda() * 32
        s = torch.rand(size=[num_of_channel]).cuda()
        if sym:o = torch.zeros(size=[num_of_channel]).cuda()
        else: o = torch.randint(low=0, high=255, size=[num_of_channel]).float().cuda()
        dy = torch.rand_like(t)
        
        cuda_grad_x, cuda_grad_s = CUDA.LinearQuantize_C_B(
            t, s, o, dy, Q_MIN, Q_MAX, c, ROUNDING_POLICY.value)
        
        ref_grad_x, ref_grad_s = ref_grad_func(t, dy, s, o, c)
        diff = ref_grad_x.flatten() - cuda_grad_x.flatten()
        if diff.abs().max() != 0:
            print(ref_grad_x)
            print(cuda_grad_x)
            raise Exception('Test Failed.')

        snr = torch_snr_error(cuda_grad_s.reshape([1, -1]), ref_grad_s.reshape([1, -1])).item()
        if snr > 1e-5:
            print(f'[Warning]. Floating Precision Error. {snr:.4f} {cuda_grad_s.item():4f} {ref_grad_s.item():4f}')
            if cuda_grad_s.item() > 1:
                raise Exception('Test Failed.')

__TEST_QUANTIZE_LT__(size=[1, 1, 1, 1], iterations=100, sym=True)
__TEST_QUANTIZE_LT__(size=[1, 1, 1, 1], iterations=100, sym=False)
__TEST_QUANTIZE_LT__(size=[5, 12, 13, 4], iterations=100, sym=True)
__TEST_QUANTIZE_LT__(size=[1, 7, 15, 41], iterations=100, sym=False)
__TEST_QUANTIZE_LT__(size=[50, 120, 130, 4], iterations=30, sym=True)
__TEST_QUANTIZE_LT__(size=[12, 74, 15, 411], iterations=30, sym=False)
__TEST_QUANTIZE_LT__(size=[50, 7, 130, 1], iterations=30, sym=True)
__TEST_QUANTIZE_LT__(size=[12, 4, 15, 3], iterations=30, sym=False)
__TEST_QUANTIZE_LT__(size=[5011, 7, 7, 1], iterations=30, sym=True)
__TEST_QUANTIZE_LT__(size=[122552, 1, 10, 4], iterations=30, sym=False)
__TEST_QUANTIZE_LT__(size=[10, 10, 124, 47], iterations=30, sym=True)
__TEST_QUANTIZE_LT__(size=[19, 42, 150, 3], iterations=30, sym=False)

__TEST_QUANTIZE_LC__(size=[1, 1, 1, 1], iterations=100, sym=True)
__TEST_QUANTIZE_LC__(size=[1, 1, 1, 1], iterations=100, sym=False)
__TEST_QUANTIZE_LC__(size=[5, 12, 13, 4], iterations=100, sym=True)
__TEST_QUANTIZE_LC__(size=[1, 7, 15, 41], iterations=100, sym=False)
__TEST_QUANTIZE_LC__(size=[50, 120, 130, 4], iterations=30, sym=True)
__TEST_QUANTIZE_LC__(size=[12, 74, 15, 411], iterations=30, sym=False)
__TEST_QUANTIZE_LC__(size=[50, 7, 130, 1], iterations=30, sym=True)
__TEST_QUANTIZE_LC__(size=[12, 4, 15, 3], iterations=30, sym=False)
__TEST_QUANTIZE_LC__(size=[5011, 7, 7, 1], iterations=30, sym=True, c=0)
__TEST_QUANTIZE_LC__(size=[122552, 1, 10, 4], iterations=30, sym=False, c=0)
__TEST_QUANTIZE_LC__(size=[10, 10, 124, 47], iterations=30, sym=True, c=3)
__TEST_QUANTIZE_LC__(size=[19, 42, 150, 3], iterations=30, sym=False, c=3)

__TEST_QUANTIZE_LT_B__(size=[1, 1, 1, 1], iterations=100, sym=True)
__TEST_QUANTIZE_LT_B__(size=[1, 1, 1, 1], iterations=100, sym=False)
__TEST_QUANTIZE_LT_B__(size=[5, 12, 13, 4], iterations=100, sym=True)
__TEST_QUANTIZE_LT_B__(size=[1, 7, 15, 41], iterations=100, sym=False)
__TEST_QUANTIZE_LT_B__(size=[5, 12, 2130, 4], iterations=30, sym=True)
__TEST_QUANTIZE_LT_B__(size=[12, 74, 315, 41], iterations=30, sym=False)
__TEST_QUANTIZE_LT_B__(size=[50, 17, 13, 1], iterations=30, sym=True)
__TEST_QUANTIZE_LT_B__(size=[12, 41, 15, 3], iterations=30, sym=False)
__TEST_QUANTIZE_LT_B__(size=[501, 7, 73, 1], iterations=30, sym=True)
__TEST_QUANTIZE_LT_B__(size=[1222, 12, 10, 4], iterations=30, sym=False)
__TEST_QUANTIZE_LT_B__(size=[10, 104, 14, 47], iterations=30, sym=True)
__TEST_QUANTIZE_LT_B__(size=[19, 42, 120, 3], iterations=30, sym=False)

__TEST_QUANTIZE_LC_B__(size=[1, 1, 1, 1], iterations=100, sym=True)
__TEST_QUANTIZE_LC_B__(size=[1, 1, 1, 1], iterations=100, sym=False)
__TEST_QUANTIZE_LC_B__(size=[5, 12, 14, 12], iterations=100, sym=True)
__TEST_QUANTIZE_LC_B__(size=[1, 7, 15, 41], iterations=100, sym=False)
__TEST_QUANTIZE_LC_B__(size=[5, 12, 130, 4], iterations=30, sym=True)
__TEST_QUANTIZE_LC_B__(size=[12, 74, 15, 41], iterations=30, sym=False)
__TEST_QUANTIZE_LC_B__(size=[50, 7, 13, 1], iterations=30, sym=True)
__TEST_QUANTIZE_LC_B__(size=[12, 4, 15, 3], iterations=30, sym=False)
__TEST_QUANTIZE_LC_B__(size=[501, 7, 7, 1], iterations=30, sym=True, c=0)
__TEST_QUANTIZE_LC_B__(size=[1222, 1, 10, 4], iterations=30, sym=False, c=0)
__TEST_QUANTIZE_LC_B__(size=[10, 10, 14, 47], iterations=30, sym=True, c=3)
__TEST_QUANTIZE_LC_B__(size=[19, 42, 10, 3], iterations=30, sym=False, c=3)

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
