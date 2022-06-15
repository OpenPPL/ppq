import torch

class TestBlock1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class TestBlock2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TestBlock3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu1(self.conv1(x))


class TestBlock4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu1(self.conv1(x)) + self.relu2(self.conv2(x))


class TestBlock5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU6()
        
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.relu2 = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu1(self.conv1(x)) + self.relu2(self.conv2(x))


class TestBlock6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.relu2 = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.relu1(self.conv1(x)) + self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class TestBlock7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.bn1   = torch.nn.BatchNorm2d(num_features=16)
        self.relu1 = torch.nn.ReLU()
        
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)
        self.bn2   = torch.nn.BatchNorm2d(num_features=16)
        self.relu2 = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.relu1(self.bn1(self.conv1(x))) + self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class TestBlock8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = x.mean(dim=1)
        return x


class TestBlock9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = x.sum(dim=1)
        return x


class TestBlock10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm = torch.nn.Linear(in_features=1000, out_features=1000)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        return x


class TestBlock11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm1(x)
        x = self.gemm2(x)
        x = self.gemm3(x)
        return x


class TestBlock12(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.relu1 = torch.nn.ReLU()
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.relu2 = torch.nn.ReLU()
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.gemm1(x))
        x = self.relu2(self.gemm2(x))
        x = self.relu3(self.gemm3(x))
        return x


class TestBlock13(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn1   = torch.nn.BatchNorm1d(num_features=1000)
        self.relu1 = torch.nn.ReLU()
        
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn2   = torch.nn.BatchNorm1d(num_features=1000)
        self.relu2 = torch.nn.ReLU()
        
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn3   = torch.nn.BatchNorm1d(num_features=1000)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.gemm1(x)))
        x = self.relu2(self.bn2(self.gemm2(x)))
        x = self.relu3(self.bn3(self.gemm3(x)))
        return x


class TestBlock14(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gemm1 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn1   = torch.nn.BatchNorm1d(num_features=1000)
        self.relu1 = torch.nn.ReLU()
        
        self.gemm2 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn2   = torch.nn.BatchNorm1d(num_features=1000)
        self.relu2 = torch.nn.ReLU()
        
        self.gemm3 = torch.nn.Linear(in_features=1000, out_features=1000)
        self.bn3   = torch.nn.BatchNorm1d(num_features=1000)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu1(self.bn1(self.gemm1(x)))
        x2 = self.relu2(self.bn2(self.gemm2(x)))
        x3 = self.relu3(self.bn3(self.gemm3(x)))
        return torch.cat([x1, x2, x3], dim=-1)


class TestBlock15(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
        self.conv5 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.conv6 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=6)
        self.conv7 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class TestBlock17(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=4)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=8)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=16)
        self.conv6 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class TestBlock18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=8, padding=[1, 2])
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=16, padding=[2, 0])
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=[0, 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class TestBlock19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, dilation=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
