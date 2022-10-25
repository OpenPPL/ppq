import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('input: ', x.shape)
        x = F.relu(self.conv1(x))
        print('conv1',x.shape)
        x = self.pool1(x)
        print('pool1: ', x.shape)
        x = F.relu(self.conv2(x))
        print('conv2',x.shape)
        x = self.pool1(x)
        print('pool2',x.shape)
        x = x.view(x.size(0), -1)
        print('view: ', x.shape)
        x = F.relu(self.fc1(x))
        print('fc1: ', x.shape)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def export_onnx(onnx_filename):
    points = torch.full((1, 1, 32, 32), 1.5, dtype=torch.float32).cuda()
    inputs = (points, )
    model = Lenet5().cuda()
    torch.onnx.export(model, inputs, onnx_filename, opset_version=11)
    print("The output of raw network: ", torch.mean(model(*inputs)).detach().cpu().numpy())

def print_onnx_model(onnx_filename):
    # Load the ONNX model
    import onnx
    model = onnx.load(onnx_filename)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))  


def main():
    # get onnx
    onnx_filename = "lenet.onnx"
    export_onnx(onnx_filename)
    print_onnx_model(onnx_filename)

if __name__ == '__main__':
    main()
