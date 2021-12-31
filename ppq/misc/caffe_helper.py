import re
import logging
from google.protobuf import text_format
from ppq.parser.caffe.proto import ppl_caffe_pb2
from ppq.parser.caffe.caffe_layer import get_input_shape

logger = logging.getLogger(__name__)


def _convert(name):
    # Convert string 'InnerProduct'-like into 'inner_product_param' format
    if name in ['Deconvolution', 'ReflectionPadConvolution']:
        return 'convolution_param'
    if name == 'ReflectionPad':
        return 'pad_param'
    if name in ['ReLU', 'PReLU', 'ReLU6']:
        return name.lower() + '_param'

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower() + '_param'


def clean_caffemodel(prototxt, caffemodel, save_caffemodel):
    """
    Clean the caffemodel according to prototxt, which will delete unused layer and blobs
    :param prototxt: prototxt name
    :param caffemodel: caffemodel name
    :param save_caffemodel: caffemodel name after clean
    """
    # Load prototxt, caffemodel
    net_def = ppl_caffe_pb2.NetParameter()
    with open(prototxt) as f:
        text_format.Merge(f.read(), net_def)
    weight = ppl_caffe_pb2.NetParameter()
    with open(caffemodel, 'rb') as f:
        weight.ParseFromString(f.read())

    # merge net_def(prototxt file) with caffemodel
    for i in net_def.layer:
        for j in weight.layer:
            if i.name == j.name:
                i.ClearField('blobs')
                i.blobs.MergeFrom(j.blobs)
                break

    # Save prototxt, caffemodel
    with open(save_caffemodel, 'wb') as f:
        byte = f.write(net_def.SerializeToString())
        print(f'Save caffe model to {save_caffemodel} with size {byte / 1000000:.2f}MB')


def caffe_similarity(proto_1, proto_2):
    """
    Check the similarity of two caffe model, only compare the prototxt
    The layer orders must be the same
    :param proto_1:
    :param proto_2:
    :return: bool
    """
    net_1 = ppl_caffe_pb2.NetParameter()
    with open(proto_1) as f:
        text_format.Merge(f.read(), net_1)
    net_2 = ppl_caffe_pb2.NetParameter()
    with open(proto_2) as f:
        text_format.Merge(f.read(), net_2)

    # Compare inputs
    input_same = (len(net_1.input) == len(net_2.input))
    input_1 = get_input_shape(net_1)
    input_2 = get_input_shape(net_2)
    for name_1, dim_1 in input_1.items():
        if name_1 not in input_2:
            input_same = False
        else:
            input_same = input_same and (input_2[name_1] == dim_1)
        if not input_same:
            break

    check_flag = 'Pass!' if input_same else 'Fail!'
    logger.info(f'Check input ... {check_flag}')

    # Compare layers
    # The layers must in the same order
    layer_same = (len(net_1.layer) == len(net_2.layer))
    if layer_same:
        for layer_1, layer_2 in zip(net_1.layer, net_2.layer):
            layer_same = (layer_1.type == layer_2.type) and (len(layer_1.bottom) == len(layer_2.bottom)) and \
                         (len(layer_1.top) == len(layer_2.top))
            if not layer_same:
                logger.info('Check layer ... False')
                return False
            param_name = _convert(layer_1.type)
            layer_same = layer_same and (getattr(layer_1, param_name) == getattr(layer_2, param_name))

    check_flag = 'Pass!' if layer_same else 'Fail!'
    logger.info(f'Check layer ... {check_flag}')
