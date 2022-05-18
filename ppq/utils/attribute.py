from ppq.log import NaiveLogger
from ppq.core.defs import ppq_legacy

logger = NaiveLogger.get_logger('PPQ')

# attribute checker and preprocess
def process_attribute(attr, input_shape, kernel_shape=None, op_type=None):
    # ASSUME input is 2D
    # assert len(input_shape) == 2
    # Get default attr value
    auto_pad = attr.get('auto_pad', 'NOTSET')
    strides = attr.get('strides', [1, 1])
    dilations = attr.get('dilations', [1, 1])
    kernels = attr.get('kernel_shape', kernel_shape)
    pad_needed = None

    if op_type == 'ConvTranspose' and 'output_shape' in attr:
        output_shape = attr['output_shape']
        out_pad = [0, 1] if output_shape % 2 != 0 else [0, 0]
        pad_needed = [(input_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 + out_pad[i] -
                      output_shape[i] for i in range(len(input_shape))]

    if auto_pad != 'NOTSET':
        if 'pads' in attr:
            logger.warning('auto_pad is conflict with pads attribute. Use pads here.')
        elif auto_pad == 'VALID':
            attr['pads'] = [0, 0, 0, 0]
        elif auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
            if op_type == 'ConvTranspose':
                # `output_padding` is only used to find output shape, but does not actually add zero-padding to output
                out_pad = attr.get('output_padding', [0, 0])
                output_shape = [input_shape[i] * strides[i] for i in range(len(input_shape))]
                pad_needed = [(input_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 + out_pad[i] -
                              output_shape[i] for i in range(len(input_shape))]
            else:
                output_shape = [(input_shape[i] + strides[i] - 1) // strides[i] for i in range(len(input_shape))]
                pad_needed = [(output_shape[i] - 1) * strides[i] + dilations[i] * (kernels[i] - 1) + 1 - input_shape[i]
                              for i in range(len(input_shape))]
        else:
            raise ValueError(f'Invalid auto_pad value {auto_pad}')

    if pad_needed is not None:
        pads = []
        for item in pad_needed:
            pads.append((item if auto_pad == 'SAME_UPPER' else item + 1) // 2)
        # onnx pads format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...]
        pads = pads + [pad_needed[i] - p for i, p in enumerate(pads)]
        attr['pads'] = pads
        # onnx pads attribute cannot be used simultaneously with auto_pad attribute
        attr.pop('auto_pad')


def preprocess_attr(attr, op_type=None):
    processed_attribute = {}
    if 'kernel_shape' in attr and op_type == 'Pooling':
        processed_attribute['kernel_size'] = attr['kernel_shape']
    if 'group' in attr:
        processed_attribute['groups'] = attr['group']
    if 'pads' in attr:
        # Change pads from start-end to torch format
        pads = attr['pads']
        assert (len(pads) % 2 == 0)
        if len(pads) == 4:
            begin_pad = pads[:2]
            end_pad = pads[2:]
            if begin_pad == end_pad:
                processed_attribute['padding'] = begin_pad
            else:
                raise ValueError('Torch function only support begin_pad == end_pad in layer')
        else:
            processed_attribute['padding'] = pads

    if 'dilations' in attr:
        processed_attribute['dilation'] = attr['dilations']
    if 'strides' in attr:
        processed_attribute['stride'] = attr['strides']
    if 'ceil_mode' in attr:
        processed_attribute['ceil_mode'] = bool(attr['ceil_mode'])
    return processed_attribute
