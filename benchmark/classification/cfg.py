import torchvision



MODELS = {
    'ResNet18': torchvision.models.resnet18,
    'MobileNetV2':torchvision.models.mobilenet_v2
    }

FP32_BASE_PATH = "./FP32_model"
OPENVINO_BASE_PATH = "./OpenVino_output"
TRT_BASE_PATH = "./TRT_output"


BATCHSIZE = 64     # 测试与calib时的 batchsize
INPUT_SHAPE = (BATCHSIZE,3,224,224)
DEVICE = "cuda"


VALIDATION_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Valid'   # 用来读取 validation dataset
TRAIN_DIR = '/home/geng/tinyml/ppq/benchmark/Assets/Imagenet_Train'        # 用来读取 train dataset，注意该集合将被用来 calibrate 你的模型

