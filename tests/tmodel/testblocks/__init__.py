from functools import partial

import torchvision

from ..base import *
from .blocks import *

TORCH_TEST_BLOCKS = [
    PPQTestCase(
        model_builder = TestBlock1,
        input_generator = partial(rand_tensor_generator, [1, 1, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock1'
    ),
    PPQTestCase(
        model_builder = TestBlock2,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock2'
    ),
    PPQTestCase(
        model_builder = TestBlock3,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock3'
    ),
    PPQTestCase(
        model_builder = TestBlock4,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock4'
    ),
    PPQTestCase(
        model_builder = TestBlock5,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock5'
    ),
    PPQTestCase(
        model_builder = TestBlock6,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock6'
    ),
    PPQTestCase(
        model_builder = TestBlock7,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock7'
    ),
    PPQTestCase(
        model_builder = TestBlock8,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock8'
    ),
    PPQTestCase(
        model_builder = TestBlock9,
        input_generator = partial(rand_tensor_generator, [1, 3, 96, 96]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock9'
    ),
    PPQTestCase(
        model_builder = TestBlock10,
        input_generator = partial(rand_tensor_generator, [1, 1000]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock10'
    ),
    PPQTestCase(
        model_builder = TestBlock11,
        input_generator = partial(rand_tensor_generator, [1, 1000]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock11'
    ),
    PPQTestCase(
        model_builder = TestBlock12,
        input_generator = partial(rand_tensor_generator, [1, 1000]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock12'
    ),
    PPQTestCase(
        model_builder = TestBlock13,
        input_generator = partial(rand_tensor_generator, [1, 1000]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock13'
    ),
    PPQTestCase(
        model_builder = TestBlock14,
        input_generator = partial(rand_tensor_generator, [1, 1000]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock14'
    ),
    PPQTestCase(
        model_builder = TestBlock15,
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock15'
    ),
    PPQTestCase(
        model_builder = TestBlock17,
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock17'
    ),
    PPQTestCase(
        model_builder = TestBlock18,
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock18'
    ),
    PPQTestCase(
        model_builder = TestBlock19,
        input_generator = partial(rand_tensor_generator, [1, 3, 224, 224]),
        model_type = ModelType.BLOCK,
        model_name = 'TestBlock19'
    ),
]