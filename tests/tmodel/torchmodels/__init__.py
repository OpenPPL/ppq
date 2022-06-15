from functools import partial

import torchvision

from ..base import *

TORCH_TEST_CASES = [
    PPQTestCase(
        model_builder = torchvision.models.alexnet,
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'AlexNet(Imagenet)'
    ),
    PPQTestCase(
        model_builder = torchvision.models.inception_v3, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'InceptionV3(Imagenet)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
    PPQTestCase(
        model_builder = torchvision.models.mnasnet1_0, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'Mnasnet(Imagenet)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
    PPQTestCase(
        model_builder = torchvision.models.squeezenet1_0, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'Squeezenet(Imagenet)'
    ),
    PPQTestCase(
        model_builder = torchvision.models.shufflenet_v2_x1_0, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'Shufflenet(Imagenet)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
    PPQTestCase(
        model_builder = torchvision.models.resnet18, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'Resnet18(Imagenet)'
    ),    
    PPQTestCase(
        model_builder = torchvision.models.googlenet, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'GoogleNet(Imagenet)'
    ),
    PPQTestCase(
        model_builder = torchvision.models.vgg16, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'Vgg16(Imagenet)'
    ),    
    PPQTestCase(
        model_builder = torchvision.models.mobilenet_v2, 
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.CLASSIFY,
        model_name = 'MobilenetV2(Imagenet)'
    ),
]

TORCH_DETECTION_CASES = [
    PPQTestCase(
        model_builder = torchvision.models.detection.fasterrcnn_resnet50_fpn, 
        input_generator = partial(rand_tensor_generator, [1, 3, 480, 640]),
        model_type = ModelType.DETECTION,
        model_name = 'FasterRCNN(COCO)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
    PPQTestCase(
        model_builder = torchvision.models.detection.retinanet_resnet50_fpn, 
        input_generator = partial(rand_tensor_generator, [1, 3, 480, 640]),
        model_type = ModelType.DETECTION,
        model_name = 'RetinaNet(COCO)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
    PPQTestCase(
        model_builder = torchvision.models.detection.ssd300_vgg16, 
        input_generator = partial(rand_tensor_generator, [1, 3, 320, 240]),
        model_type = ModelType.DETECTION,
        model_name = 'SSD300(COCO)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
    PPQTestCase(
        model_builder = torchvision.models.detection.maskrcnn_resnet50_fpn, 
        input_generator = partial(rand_tensor_generator, [1, 3, 480, 640]),
        model_type = ModelType.DETECTION,
        model_name = 'MaskRCNN(COCO)',
        deploy_platforms = [TargetPlatform.PPL_CUDA_INT8]
    ),
]
