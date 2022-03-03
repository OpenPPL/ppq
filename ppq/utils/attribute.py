import logging
logger = logging.getLogger('PPQ')

# attribute checker and preprocess
def checker(attr, input_shape, kernel_shape=None, op_type=None):
    # ASSUME input is 2D
    assert len(input_shape) == 2
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
            pads.append((item + 1) // 2)
        # onnx pads format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...]
        attr['pads'] = pads * 2
        # onnx pads attribute cannot be used simultaneously with auto_pad attribute
        attr.pop('auto_pad')


def preprocess_attr(attr, op_type=None):
    new_attr = {}
    if 'kernel_shape' in attr and op_type == 'Pooling':
        new_attr['kernel_size'] = attr['kernel_shape']
    if 'group' in attr:
        new_attr['groups'] = attr['group']
    if 'pads' in attr:
        # Change pads from start-end to torch format
        pads = attr['pads']
        assert (len(pads) % 2 == 0)
        if len(pads) == 4:
            begin_pad = pads[:2]
            end_pad = pads[2:]
            if begin_pad == end_pad:
                new_attr['padding'] = begin_pad
            else:
                # onnx pads format[top, left, bottom, right] to torch pads format[left, right, top, bottom]
                new_attr["external_padding"] = [pads[1], pads[3], pads[0], pads[2]]
                new_attr['padding'] = [0, 0]
                # raise ValueError('Torch function only support begin_pad == end_pad in layer')
        else:
            new_attr['padding'] = pads

    if 'dilations' in attr:
        new_attr['dilation'] = attr['dilations']
    if 'strides' in attr:
        new_attr['stride'] = attr['strides']
    if 'ceil_mode' in attr:
        new_attr['ceil_mode'] = bool(attr['ceil_mode'])
    return new_attr
