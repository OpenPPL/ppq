# ---------------------------------------------------------------
# 这个脚本向你展示了如何使用 torch2trt 加速 pytorch 推理
# 截止目前为止 torch2trt 的适配能力有限，不要尝试运行特别奇怪的模型
# 你可以把模型分块来绕开那些不支持的算子。

# 使用之前你必须先装好 TensorRT, torch2trt等工具包
# https://github.com/NVIDIA-AI-IOT/torch2trt

# ---------------------------------------------------------------

import torch
import torch.utils.data
import torchvision
from torch2trt import torch2trt
from tqdm import tqdm


SAMPLES = [torch.zeros(1, 3, 224, 224) for _ in range(1024)]
MODEL = torchvision.models.resnet18()
FP16_MODE = True

# Model has to be the eval mode, and deploy to cuda.
MODEL.eval()
MODEL.cuda()

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

# Benckmark with pytorch
for sample in tqdm(SAMPLES, desc='Torch Executing'):
    MODEL.forward(sample.cuda())

# Convert torch.nn.Module to tensorrt
# 在转换过后，你模型中的执行函数将会被 trt 替换，同时进行图融合
model_trt = torch2trt(MODEL, [sample.cuda()], fp16_mode=FP16_MODE)
for sample in tqdm(SAMPLES, desc='TRT Executing'):
    model_trt.forward(sample.cuda())

# Test performance metrics using torch.profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=7),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('log')
    # used when outputting for tensorboard
    ) as p:
        for iter in range(10):
            model_trt.forward(sample.cuda())
            # send a signal to the profiler that the next iteration has started
            p.step()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=7),
    on_trace_ready=trace_handler
    # on_trace_ready=torch.profiler.tensorboard_trace_handler('log')
    # used when outputting for tensorboard
    ) as p:
        for iter in range(10):
            MODEL.forward(sample.cuda())
            # send a signal to the profiler that the next iteration has started
            p.step()
