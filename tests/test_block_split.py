import torch

from ppq.core.quant import TargetPlatform

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gemm_1 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_2 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_3 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_4 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_5 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_6 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_7 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_8 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_9 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_10 = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_J = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_Q = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_K = torch.nn.Linear(in_features=10, out_features=10)
        self.gemm_A = torch.nn.Linear(in_features=10, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm_1(x)
        x = torch.relu(x)
        
        x2 = torch.relu(self.gemm_2(x))
        x3 = torch.relu(self.gemm_3(x))
        x4 = torch.relu(self.gemm_4(x))
        x5 = torch.relu(self.gemm_5(x))
        x6 = torch.relu(self.gemm_6(x))
        
        x2 = self.gemm_7(x2)
        x3 = self.gemm_8(x3)
        x4 = self.gemm_9(x4)
        x5 = self.gemm_10(x5)
        x6 = self.gemm_J(x6)
        
        x7 = torch.relu(self.gemm_Q(x))
        x7 = self.gemm_K(x7)
        
        x8 = torch.max_pool1d(x7, kernel_size=2)
        return torch.cat([x2, x3, x4, x5, x6, x7, x8], dim=-1)

model = MyModel().cuda()
model.forward(torch.zeros(size=[10, 10]).cuda())

from ppq.api import load_torch_model
from ppq.api import quantize_native_model
from ppq.api import QuantizationSettingFactory

graph = load_torch_model(model=model, sample=torch.zeros(size=[10, 10]).cuda())
s = QuantizationSettingFactory.default_setting()
s.lsq_optimization = True
s.lsq_optimization_setting.block_size = 4

quantize_native_model(
    model=graph, calib_dataloader=[torch.zeros(size=[10, 10])], input_shape=[10, 10],
    calib_steps=8, collate_fn=lambda x: x.cuda(), platform=TargetPlatform.TRT_INT8,
    setting=s)