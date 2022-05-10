from .default import DEFAULT_BACKEND_TABLE
import torch
import torch.nn.functional as F


NXP_BACKEND_TABLE = DEFAULT_BACKEND_TABLE.copy()


def Resize_forward(op, input_value, device=None):
    """NXP Platform has a custimized resize operation implementation, which
    gives different result from torch.nn.Resize. To correctly simulate hardware
    beviour and have the same result with NXP, it is necessary to force resize
    to run with nearest mode. Any other mode of resize will be ignored by this
    function.

    Args:
        op ([type]): [description]
        input_value ([type]): [description]
        device ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    input_data = input_value[0]
    # Not used roi
    # roi  = input_value[1] if len(input_value) > 1 else None
    scales = input_value[2] if len(input_value) > 2 else None
    sizes = input_value[-1].tolist() if len(input_value) == 4 else None
    mode = 'nearest'

    # If 'size' is specified, then set scales to empty data (zero shape) in this operator's input list.
    if sizes is None or len(sizes) == 0:
        sizes = None
        if scales.numel() == 1:
            scales = scales.item()
        else:
            assert scales.numel() % 2 == 0
            scales = scales[-2].cpu().numpy().tolist()
    else:
        # the sizes in onnx is 4-D while in pytorch is 2-D
        # check the dim.0 & dim.1 is equal, then remain dim.2 and dim.3
        scales = None
        assert (sizes[:2] == list(input_data.shape[:2]))
        sizes = sizes[2:]

    trans_mode = op.attributes.get('coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        output = F.interpolate(input_data, sizes, scales, mode, align_corners=True)
    else:
        output = F.interpolate(input_data, sizes, scales, mode)
    return output


# When you trying to implement a custimized function for ppl_dsp platform
# Be aware that you can just overwrite part of DEFAULT_DISPATCHING_TABLE
# rather than rewrite all dispatching table.
# here an example was given: Sample_Forward
def Sample_Forward():
    return None


NXP_BACKEND_TABLE['Sample_Function'] = Sample_Forward
# NXP_DISPATCHING_TABLE['Resize'] = Resize_forward
