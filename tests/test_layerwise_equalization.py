from ppq import *
from ppq.api import *
import torch

# 创建计算图
graph = BaseGraph(name='Created Graph', built_from=NetworkFramework.NATIVE)
op1 = graph.create_operation(
    op_type='Gemm', name='Gemm1',
    inputs=[graph.create_variable(), 
            graph.create_variable(is_parameter=True, value=torch.rand(size=[128, 1024]).cuda() * 100), 
            graph.create_variable(is_parameter=True, value=torch.rand(size=[1024]).cuda() * 100)],
    outputs=[graph.create_variable()])

op2 = graph.create_operation(
    op_type='Gemm', name='Gemm2',
    inputs=[op1.outputs[0], 
            graph.create_variable(is_parameter=True, value=torch.rand(size=[1024, 128]).cuda()), 
            graph.create_variable(is_parameter=True, value=torch.rand(size=[128]).cuda())],
    outputs=[graph.create_variable()])

op3 = graph.create_operation(
    op_type='Gemm', name='Gemm3', attributes={'transB': 1},
    inputs=[op2.outputs[0], 
            graph.create_variable(is_parameter=True, value=torch.rand(size=[1024, 128]).cuda() * 500), 
            graph.create_variable(is_parameter=True, value=torch.rand(size=[1024]).cuda() * 0)],
    outputs=[graph.create_variable()])

graph.mark_variable_as_graph_input(op1.inputs[0])
graph.mark_variable_as_graph_output(op3.outputs[0])

inputs   = [torch.rand(size=[8, 128]) for _ in range(32)]
executor = TorchExecutor(graph=graph)
b_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
b_outputs = torch.cat(b_outputs)

from ppq.quantization.optim import LayerwiseEqualizationPass
LayerwiseEqualizationPass(iterations=1000, including_bias=True, including_act=True).optimize(
    graph=graph, dataloader=inputs, executor=executor, collate_fn=lambda x: x.cuda())
p_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
p_outputs = torch.cat(p_outputs)
from ppq.quantization.measure import torch_snr_error
assert torch_snr_error(b_outputs, p_outputs).item() < 1e-7

import torch
class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.conv1 = torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, groups=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=8, groups=8, kernel_size=5, stride=1, padding=2)
            self.convtranspose1 = torch.nn.ConvTranspose2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
            self.convtranspose2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, groups=32, kernel_size=3, stride=2, bias=False)
            self.convtranspose3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=8, groups=1, kernel_size=1)
            
            self.conv1.bias.copy_(torch.rand_like(self.conv1.bias))
            self.conv3.bias.copy_(torch.rand_like(self.conv3.bias))
            self.convtranspose1.bias.copy_(torch.rand_like(self.convtranspose1.bias))
            self.convtranspose3.bias.copy_(torch.rand_like(self.convtranspose3.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.convtranspose1(x)
        x = self.convtranspose2(x)
        x = self.convtranspose3(x)
        return x

model = MyModel().cuda()
dump_torch_to_onnx(model=model, onnx_export_file='model.onnx', input_shape=[1, 8, 96, 96])
graph = load_onnx_graph(onnx_import_file='model.onnx')

inputs   = [torch.rand(size=[1, 8, 96, 96]) for _ in range(32)]
executor = TorchExecutor(graph=graph)
b_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
b_outputs = torch.cat(b_outputs)

from ppq.quantization.optim import LayerwiseEqualizationPass
LayerwiseEqualizationPass(iterations=10, including_bias=True, including_act=True).optimize(
    graph=graph, dataloader=inputs, executor=executor, collate_fn=lambda x: x.cuda())
p_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
p_outputs = torch.cat(p_outputs)
from ppq.quantization.measure import torch_snr_error
assert torch_snr_error(b_outputs, p_outputs).item() < 1e-7


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.conv1 = torch.nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv3d(in_channels=32, out_channels=32, groups=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv3d(in_channels=32, out_channels=8, groups=8, kernel_size=5, stride=1, padding=2)
            self.conv4 = torch.nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)

            self.conv1.bias.copy_(torch.rand_like(self.conv1.bias))
            self.conv3.bias.copy_(torch.rand_like(self.conv3.bias))
            self.conv4.bias.copy_(torch.rand_like(self.conv4.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

model = MyModel().cuda()
dump_torch_to_onnx(model=model, onnx_export_file='model.onnx', input_shape=[1, 8, 16, 96, 96])
graph = load_onnx_graph(onnx_import_file='model.onnx')

inputs   = [torch.rand(size=[1, 8, 16, 96, 96]) for _ in range(32)]
executor = TorchExecutor(graph=graph)
b_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
b_outputs = torch.cat(b_outputs)

from ppq.quantization.optim import LayerwiseEqualizationPass
LayerwiseEqualizationPass(iterations=10, including_bias=True, including_act=True).optimize(
    graph=graph, dataloader=inputs, executor=executor, collate_fn=lambda x: x.cuda())
p_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
p_outputs = torch.cat(p_outputs)
from ppq.quantization.measure import torch_snr_error
assert torch_snr_error(b_outputs, p_outputs).item() < 1e-7


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.conv1 = torch.nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, groups=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=8, groups=8, kernel_size=5, stride=1, padding=2)
            self.conv4 = torch.nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            
            self.conv1.bias.copy_(torch.rand_like(self.conv1.bias))
            self.conv3.bias.copy_(torch.rand_like(self.conv3.bias))
            self.conv4.bias.copy_(torch.rand_like(self.conv4.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

model = MyModel().cuda()
dump_torch_to_onnx(model=model, onnx_export_file='model.onnx', input_shape=[1, 8, 96])
graph = load_onnx_graph(onnx_import_file='model.onnx')

inputs   = [torch.rand(size=[1, 8, 96]) for _ in range(32)]
executor = TorchExecutor(graph=graph)
b_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
b_outputs = torch.cat(b_outputs)

from ppq.quantization.optim import LayerwiseEqualizationPass
LayerwiseEqualizationPass(iterations=10, including_bias=True, including_act=True).optimize(
    graph=graph, dataloader=inputs, executor=executor, collate_fn=lambda x: x.cuda())
p_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
p_outputs = torch.cat(p_outputs)
from ppq.quantization.measure import torch_snr_error
assert torch_snr_error(b_outputs, p_outputs).item() < 1e-7


class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.conv1 = torch.nn.ConvTranspose1d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=32, groups=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=8, groups=8, kernel_size=5, stride=1, padding=2)
            self.conv4 = torch.nn.ConvTranspose1d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)

            self.conv1.bias.copy_(torch.rand_like(self.conv1.bias))
            self.conv3.bias.copy_(torch.rand_like(self.conv3.bias))
            self.conv4.bias.copy_(torch.rand_like(self.conv4.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

model = MyModel().cuda()
dump_torch_to_onnx(model=model, onnx_export_file='model.onnx', input_shape=[1, 8, 96])
graph = load_onnx_graph(onnx_import_file='model.onnx')

inputs   = [torch.rand(size=[1, 8, 96]) for _ in range(32)]
executor = TorchExecutor(graph=graph)
b_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
b_outputs = torch.cat(b_outputs)

from ppq.quantization.optim import LayerwiseEqualizationPass
LayerwiseEqualizationPass(iterations=10, including_bias=True, including_act=True).optimize(
    graph=graph, dataloader=inputs, executor=executor, collate_fn=lambda x: x.cuda())
p_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
p_outputs = torch.cat(p_outputs)
from ppq.quantization.measure import torch_snr_error
assert torch_snr_error(b_outputs, p_outputs).item() < 1e-7

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.no_grad():
            self.conv1 = torch.nn.ConvTranspose3d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)
            self.conv2 = torch.nn.ConvTranspose3d(in_channels=32, out_channels=32, groups=16, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv3 = torch.nn.ConvTranspose3d(in_channels=32, out_channels=8, groups=8, kernel_size=5, stride=1, padding=2)
            self.conv4 = torch.nn.ConvTranspose3d(in_channels=8, out_channels=32, kernel_size=3, stride=2, padding=1)

            self.conv1.bias.copy_(torch.rand_like(self.conv1.bias))
            self.conv3.bias.copy_(torch.rand_like(self.conv3.bias))
            self.conv4.bias.copy_(torch.rand_like(self.conv4.bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

model = MyModel().cuda()
dump_torch_to_onnx(model=model, onnx_export_file='model.onnx', input_shape=[1, 8, 8, 8, 8])
graph = load_onnx_graph(onnx_import_file='model.onnx')

inputs   = [torch.rand(size=[1, 8, 8, 8, 8]) for _ in range(32)]
executor = TorchExecutor(graph=graph)
b_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
b_outputs = torch.cat(b_outputs)

from ppq.quantization.optim import LayerwiseEqualizationPass
LayerwiseEqualizationPass(iterations=10, including_bias=True, including_act=True).optimize(
    graph=graph, dataloader=inputs, executor=executor, collate_fn=lambda x: x.cuda())
p_outputs = [executor.forward(inputs=t.cuda())[0].unsqueeze(0) for t in inputs]
p_outputs = torch.cat(p_outputs)
from ppq.quantization.measure import torch_snr_error
assert torch_snr_error(b_outputs, p_outputs).item() < 1e-7