from abc import ABCMeta, abstractproperty
from typing import Any, Dict, List

from ppq.core import (CAFFE_DOMAIN, COMPUTING_OP, DEFAULT_OPSET_DOMAIN,
                      DEFAULT_OPSET_VERSION, ONNX_DOMAIN, OperationMeta,
                      STRICT_OPSET_CHECKING, TargetPlatform)


class Opset():
    def __init__(self, domain: str = DEFAULT_OPSET_DOMAIN, version: int = DEFAULT_OPSET_VERSION) -> None:
        """Open Neural Network Exchange (ONNX) is an open ecosystem that empowers AI developers 
            to choose the right tools as their project evolves. 
        
        ONNX provides an open source format for AI models, both deep learning and traditional ML. 
        It defines an extensible computation graph model, as well as definitions of
            built-in operators and standard data types. 
        Currently we focus on the capabilities needed for inferencing (scoring).
        
        PPQ IR is built based on ONNX defination.

        Args:
            domain (str, optional): _description_. Defaults to DEFAULT_OPSET_DOMAIN.
            version (int, optional): _description_. Defaults to DEFAULT_OPSET_VERSION.
        """
        self.domain  = domain
        self.version = version

    def is_onnx_v13(self):
        return self.domain == ONNX_DOMAIN and self.version == 13

    def is_onnx_v11(self):
        return self.domain == ONNX_DOMAIN and self.version == 11

    def onnx_opset_version(self) -> int:
        if self.domain == ONNX_DOMAIN: return self.version
        else: return -1   

    def is_onnx(self):
        return self.domain == ONNX_DOMAIN
    
    def is_caffe(self):
        return self.domain == CAFFE_DOMAIN


class OperationBase(metaclass=ABCMeta):
    def __init__(self,
                 name: str, op_type: str,
                 attributes: Dict[str, Any],
                 opset = None,
                 platform: TargetPlatform=TargetPlatform.UNSPECIFIED) -> None:
        self._name = name
        self._type = op_type
        self._attributes = attributes
        self._platform = platform
        self._meta = None
        self._input_vars  = []
        self._output_vars = []
        self._detail = {}
        if opset is None:
            self._opset = Opset()
        else: self._opset = opset

    @ abstractproperty
    def inputs(self) -> List[Any]: pass

    @ abstractproperty
    def outputs(self) -> List[Any]: pass

    @ abstractproperty
    def parameters(self) -> List[Any]: pass

    @ property
    def name(self) -> str:
        return self._name

    @ property
    def type(self) -> str:
        return self._type

    @ type.setter
    def type(self, type: str):
        self._type = type

    @ property
    def opset(self) -> Opset:
        return self._opset

    @ opset.setter
    def opset(self, opset: Opset):
        self._opset = opset

    @ property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    @ property
    def platform(self) -> TargetPlatform:
        return self._platform

    @ platform.setter
    def platform(self, platform: TargetPlatform):
        self._platform = platform

    @ property
    def num_of_input(self) -> int:
        return len(self._input_vars)

    @ property
    def num_of_output(self) -> int:
        return len(self._output_vars)

    @ property
    def meta_data(self) -> OperationMeta:
        raise NotImplementedError('This property has been removed since PPQ 0.6.6')
 
    @ property
    def inputs(self) -> List[object]:
        return self._input_vars

    @ property
    def outputs(self) -> List[object]:
        return self._output_vars

    @ property
    def is_computing_op(self) -> bool:
        return self.type in COMPUTING_OP

    def set_extension_attrib(self, attrib: str, value: Any):
        self._detail[attrib] = value

    @ property
    def extension_attrib(self) -> dict:
        return self._detail

    def __hash__(self) -> int:
        return self._name.__hash__()


class VLink:
    def __init__(self, in_idx: int, out_idx: int) -> None:
        if not isinstance(in_idx, int):
            raise TypeError(f'Can not create vlink with input_idx {in_idx}, '
                            'only int value is acceptable here.')
        if not isinstance(out_idx, int):
            raise TypeError(f'Can not create vlink with output_idx {out_idx}, '
                            'only int value is acceptable here.')
        self.in_idx  = in_idx
        self.out_idx = out_idx


class OpSocket:
    def __init__(
        self, op: OperationBase,
        in_plat: List[TargetPlatform] = None,
        out_plat: List[TargetPlatform] = None,
        links: List[VLink] = None) -> None:

        self.in_plat = in_plat
        if in_plat is None:
            self.in_plat  = [TargetPlatform.UNSPECIFIED for _ in range(op.num_of_input)]

        self.out_plat = out_plat
        if out_plat is None:
            self.out_plat = [TargetPlatform.UNSPECIFIED for _ in range(op.num_of_output)]

        self.links = links
        if self.links is None:
            self.links = []
            for i in range(op.num_of_input):
                for j in range(op.num_of_output):
                    self.links.append(VLink(i, j))


def CEHCK_TYPE(op: OperationBase, t: str):
    pass


def CHECK_OPSET(min_version_supported: int, 
                max_version_supported: int, 
                op: OperationBase, 
                strict_check: bool = STRICT_OPSET_CHECKING):
    if not strict_check: return
    opset = op.opset
    if opset.domain != ONNX_DOMAIN:
        raise TypeError(
            f'Unrecognizable opset was found, '
            f'can not generate dispatching scheme with op {op.name}, '
            'cause it might not be a standard onnx operation.')

    if opset.version > max_version_supported or opset.version < min_version_supported:
        raise TypeError(
            f'opset version is not supported, '
            f'can not generate dispatching scheme with op {op.name}({op.type}), '
            f'currently we support only [{min_version_supported, max_version_supported}], '
            f'however {opset.version} was given.')


def DEFAULT_SOCKET_CREATOR(op: OperationBase) -> OpSocket:
    return OpSocket(op=op)


def Reshape_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 5 - 13:
    
    Inputs
        data (differentiable) : T
            An input tensor.
    
        shape (non-differentiable) : tensor(int64)
            Specified shape for output.
    
    Outputs
        reshaped (differentiable) : T
            Reshaped data.
    """
    CEHCK_TYPE(op=op, t='Reshape')
    CHECK_OPSET(op=op, min_version_supported=5, max_version_supported=13)
    return OpSocket(op=op, in_plat=[TargetPlatform.UNSPECIFIED, TargetPlatform.SOI], links=[VLink(in_idx=0, out_idx=0)])


def Pad_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs (2 - 3)
        data (differentiable) : T
            Input tensor.
            
        pads (non-differentiable) : tensor(int64)
            Tensor of integers indicating the number of padding elements to add or remove 
            (if negative) at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. 
            `pads` should be a 1D tensor of shape [2 * input_rank]. 
            `pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], 
            where xi_begin is the number of pad values added at the beginning of axis `i` and xi_end, 
            the number of pad values added at the end of axis `i`.
            
        constant_value (optional, non-differentiable) : T
            (Optional) A scalar value to be used if the mode chosen is `constant`
            (by default it is 0, empty string or False).
        
    Outputs
        output (differentiable) : T
            Tensor after padding.
    """
    CEHCK_TYPE(op=op, t='Pad')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Gather_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
        
        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of any rank q. 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.
    
    Outputs
        output (differentiable) : T
            Tensor of rank q + (r - 1).
    """
    CEHCK_TYPE(op=op, t='Gather')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    return OpSocket(
        op=op, in_plat=[TargetPlatform.UNSPECIFIED, TargetPlatform.FP32], 
        links=[VLink(in_idx=0, out_idx=0)])


def Resize_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 11 - 13:
    
    Inputs (1 - 4)
        X (differentiable) : T1
            N-D tensor
   
        roi (optional, non-differentiable) : T2
            1-D tensor given as [start1, ..., startN, end1, ..., endN], 
            where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. 
            It only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"
    
        scales (optional, non-differentiable) : tensor(float)
            The scale array along each dimension. It takes value greater than 0. 
            If it's less than 1, it's sampling down, otherwise, it's upsampling. 
            The number of elements of 'scales' should be the same as the rank of input 'X'.
            One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified.
            If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.
    
        sizes (optional, non-differentiable) : tensor(int64)
            The size of the output tensor. The number of elements of 'sizes' should be the same as the rank of input 'X'.
            Only one of 'scales' and 'sizes' can be specified.
    
    Outputs
        Y (differentiable) : T1
            N-D tensor after resizing
    """
    CEHCK_TYPE(op=op, t='Resize')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=13, strict_check=True)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Split_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:
    
    Inputs (1 - 2)
        input (differentiable) : T
            The tensor to split
    
        split (optional, non-differentiable) : tensor(int64)
            Optional length of each output. Values should be >= 0.Sum of the values 
            must be equal to the dim value at 'axis' specified.
    
    Outputs (1 - ∞)
        outputs (variadic, differentiable) : T
            One or more outputs forming list of tensors after splitting
    """
    CEHCK_TYPE(op=op, t='Split')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Topk_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 11:
    
    Inputs
        X (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_n, r]
    
        K (non-differentiable) : tensor(int64)
            A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
    
    Outputs
        Values (differentiable) : T
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
            containing top K values from the input tensor
    
        Indices (non-differentiable) : I
            Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] 
            containing the corresponding input tensor indices for the top K values.
    """
    CEHCK_TYPE(op=op, t='TopK')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=11)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    out_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        out_plat = out_plat, links=[VLink(in_idx=0, out_idx=0)])


def Tile_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:
        
    Inputs
        input (differentiable) : T
            Input tensor of any shape.
    
    repeats (non-differentiable) : T1
        1D int64 tensor of the same length as input's dimension number, 
        includes numbers of repeated copies along input's dimensions.

    Outputs
        output (differentiable) : T
            Output tensor of the same dimensions and type as tensor input. 
            output_dim[i] = input_dim[i] * repeats[i]
    """
    CEHCK_TYPE(op=op, t='Tile')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def Expand_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 8 - 13:
        
    Inputs
        input (differentiable) : T
    
    Input tensor
        shape (non-differentiable) : tensor(int64)
            A 1-D tensor indicates the shape you want to expand to, following the broadcast rule
    
    Outputs
        output (differentiable) : T
            Output tensor
    """
    CEHCK_TYPE(op=op, t='Expand')
    CHECK_OPSET(op=op, min_version_supported=8, max_version_supported=13)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def RoiAlign_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 10 - 16:
        
    Inputs
        X : T1
            Input data tensor from the previous operator; 4-D feature map of shape (N, C, H, W), 
            where N is the batch size, C is the number of channels, 
            and H and W are the height and the width of the data.

        rois : T1
            RoIs (Regions of Interest) to pool over; rois is 2-D input of shape (num_rois, 4) 
            given as [[x1, y1, x2, y2], ...]. The RoIs' coordinates are in the coordinate system of the input image. 
            Each coordinate set has a 1:1 correspondence with the 'batch_indices' input.

        batch_indices : T2
            1-D tensor of shape (num_rois,) with each element denoting the index of the corresponding image in the batch.

    Outputs
        Y : T1
            RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). 
            The r-th batch element Y[r-1] is a pooled feature map corresponding to the r-th RoI X[r-1].
    """
    CEHCK_TYPE(op=op, t='RoiAlign')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=16)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def GridSampler_Socket(op: OperationBase) -> OpSocket:
    """
    From MMCV
    """
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def Clip_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:
        
    Inputs (1 - 3)
        input (differentiable) : T
            Input tensor whose elements to be clipped
    
        min (optional, non-differentiable) : T
            Minimum value, under which element is replaced by min. 
            It must be a scalar(tensor of empty shape).
    
        max (optional, non-differentiable) : T
            Maximum value, above which element is replaced by max. 
            It must be a scalar(tensor of empty shape).
    
    Outputs
        output (differentiable) : T
            Output tensor with clipped input elements
    """
    CEHCK_TYPE(op=op, t='Clip')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def ConstantOfShape_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 9 - 9:
        
    Inputs
        input : T1
            1D tensor. The shape of the expected output tensor. 
            If empty tensor is given, the output would be a scalar. All values must be >= 0.
    
    Outputs
        output : T2
            Output tensor of shape specified by 'input'.If attribute 'value' is specified, 
            the value and datatype of the output tensor is taken from 'value'.If attribute 'value' is not specified, 
            the value in the output defaults to 0, and the datatype defaults to float32.
    """
    CEHCK_TYPE(op=op, t='ConstantOfShape')
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat = [TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[])


def GatherElements_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, with the same rank r as the input. 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of the same shape as indices.
    """
    CEHCK_TYPE(op=op, t='GatherElements')
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def GatherND_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] along axis of size s.
            It is an error if any of the index values are out of bounds.

    Outputs
        output (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.
    """
    CEHCK_TYPE(op=op, t='GatherND')
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[VLink(in_idx=0, out_idx=0)])


def NonMaxSuppression_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 10 - 13:
        
    Inputs (2 - 5)
        boxes : tensor(float)
            An input tensor with shape [num_batches, spatial_dimension, 4]. 
            The single box data format is indicated by center_point_box.
    
        scores : tensor(float)
            An input tensor with shape [num_batches, num_classes, spatial_dimension]
    
        max_output_boxes_per_class (optional) : tensor(int64)
            Integer representing the maximum number of boxes to be selected per batch per class. 
            It is a scalar. Default to 0, which means no output.
    
        iou_threshold (optional) : tensor(float)
            Float representing the threshold for deciding whether boxes overlap too much with respect to IOU. 
            It is scalar. Value range [0, 1]. Default to 0.
    
        score_threshold (optional) : tensor(float)
            Float representing the threshold for deciding when to remove boxes based on score. It is a scalar.

    Outputs
        selected_indices : tensor(int64)
            selected indices from the boxes tensor. [num_selected_indices, 3], 
            the selected index format is [batch_index, class_index, box_index].
    """
    CEHCK_TYPE(op=op, t='NonMaxSuppression')
    CHECK_OPSET(op=op, min_version_supported=10, max_version_supported=13)
    in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input],
        links=[])


def NonZero_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 9 - 13:
        
    Inputs
        X (non-differentiable) : T
            input

    Outputs
        Y (non-differentiable) : tensor(int64)
            output
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=13)
    in_plat  = [TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        out_plat=out_plat, links=[])


def Range_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 11 - 13:
        
    Inputs
        start : T
            Scalar. First entry for the range of output values.

        limit : T
            Scalar. Exclusive upper limit for the range of output values.

        delta : T
            Scalar. Value to step by.

    Outputs
        output : T
            A 1-D tensor with same type as the inputs containing generated range of values.
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=13)
    in_plat  = [TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[])


def ScatterElements_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 11 - 16:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.

        indices (non-differentiable) : Tind
            Tensor of int32/int64 indices, of r >= 1 (same rank as input). 
            All index values are expected to be within bounds [-s, s-1] along axis of size s. 
            It is an error if any of the index values are out of bounds.

        updates (differentiable) : T
            Tensor of rank r >=1 (same rank and shape as indices)

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1 (same rank as input).
    """
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=16)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.UNSPECIFIED]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)])


def ScatterND_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 11 - 16:
        
    Inputs
        data (differentiable) : T
            Tensor of rank r >= 1.
        
        indices (non-differentiable) : tensor(int64)
            Tensor of rank q >= 1.
        
        updates (differentiable) : T
            Tensor of rank q + r - indices_shape[-1] - 1.

    Outputs
        output (differentiable) : T
            Tensor of rank r >= 1.
    """
    CHECK_OPSET(op=op, min_version_supported=11, max_version_supported=16)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.UNSPECIFIED]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0), VLink(in_idx=2, out_idx=0)])


def Shape_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 16:
        
    Inputs
        data (non-differentiable) : T
            An input tensor.
    
    Outputs
        shape (non-differentiable) : T1
            Shape of the input tensor
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=16)
    in_plat  = [TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        out_plat=out_plat, links=[])


def Slice_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 16:

    Inputs (3 - 5)
        data : T
            Tensor of data to extract slices from.

        starts : Tind
            1-D tensor of starting indices of corresponding axis in `axes`

        ends : Tind
            1-D tensor of ending indices (exclusive) of corresponding axis in `axes`

        axes (optional) : Tind
            1-D tensor of axes that `starts` and `ends` apply to.
            Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).

        steps (optional) : Tind
            1-D tensor of slice step of corresponding axis in `axes`.
            Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1.

    Outputs
        output : T
            Sliced data tensor.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Squeeze_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs (1 - 2)
        data (differentiable) : T
            Tensors with at least max(dims) dimensions.

        axes (optional, non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to squeeze. 
            Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(data).
    
    Outputs
        squeezed (differentiable) : T
            Reshaped tensor with same data as input.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Unsqueeze_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 13:

    Inputs
        data (differentiable) : T
            Original tensor

        axes (non-differentiable) : tensor(int64)
            List of integers indicating the dimensions to be inserted. 
            Negative value means counting dimensions from the back. 
            Accepted range is [-r, r-1] where r = rank(expanded).

    Outputs
        expanded (differentiable) : T
            Reshaped tensor with same data as input.
    """
    CHECK_OPSET(op=op, min_version_supported=1, max_version_supported=13)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Where_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 9 - 16:

    Inputs
        condition (non-differentiable) : B
            When True (nonzero), yield X, otherwise yield Y
    
        X (differentiable) : T
            values selected at indices where condition is True
        
        Y (differentiable) : T
            values selected at indices where condition is False
    
    Outputs
        output (differentiable) : T
            Tensor of shape equal to the broadcasted shape of condition, X, and Y.
    """
    CHECK_OPSET(op=op, min_version_supported=9, max_version_supported=16)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.FP32, TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        links=[VLink(in_idx=0, out_idx=0)])


def Logical_Socket(op: OperationBase) -> OpSocket:
    """
    From Opset 1 - 16:

    Inputs
        A (non-differentiable) : T
            First input operand for the logical operator.
    
        B (non-differentiable) : T
            Second input operand for the logical operator.
    
    Outputs
        C (non-differentiable) : T1
            Result tensor.
    """
    CHECK_OPSET(op=op, min_version_supported=12, max_version_supported=16)
    in_plat  = [TargetPlatform.UNSPECIFIED, TargetPlatform.UNSPECIFIED]
    out_plat = [TargetPlatform.FP32]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        out_plat=out_plat, links=[])


def Onehot_Socket(op: OperationBase) -> OpSocket:
    """
    Inputs
        indices (non-differentiable) : T1

        depth (non-differentiable) : T2
            
        values (non-differentiable) : T3
    
    Outputs
        output (non-differentiable) : T3

    Args:
        op (Operation): _description_

    Returns:
        OpSocket: _description_
    """
    CHECK_OPSET(op=op, min_version_supported=12, max_version_supported=16)
    in_plat  = [TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
    out_plat = [TargetPlatform.SOI]
    return OpSocket(
        op=op, in_plat=in_plat[: op.num_of_input], 
        out_plat=out_plat, links=[])


DEFAULT_SOCKET_TABLE = {
    'AdaptiveAvgPool2d': DEFAULT_SOCKET_CREATOR,
    'Add': DEFAULT_SOCKET_CREATOR,
    'ArgMax': DEFAULT_SOCKET_CREATOR,
    'AveragePool': DEFAULT_SOCKET_CREATOR,
    'BatchNormalization': DEFAULT_SOCKET_CREATOR,
    'Cast': DEFAULT_SOCKET_CREATOR,
    'Clip': Clip_Socket,
    'Concat': DEFAULT_SOCKET_CREATOR,
    'Constant': DEFAULT_SOCKET_CREATOR,
    'ConstantOfShape': ConstantOfShape_Socket,
    'Conv': DEFAULT_SOCKET_CREATOR,
    'ConvTranspose': DEFAULT_SOCKET_CREATOR,
    'Div': DEFAULT_SOCKET_CREATOR,
    'Equal': Logical_Socket,
    'Exp': DEFAULT_SOCKET_CREATOR,
    'Expand': Expand_Socket,
    'Flatten': DEFAULT_SOCKET_CREATOR,
    'Gather': Gather_Socket,
    'GatherElements': GatherElements_Socket,
    'GatherND': GatherND_Socket,
    'Gelu': DEFAULT_SOCKET_CREATOR,
    'Gemm': DEFAULT_SOCKET_CREATOR,
    'grid_sampler': GridSampler_Socket,
    'GlobalAveragePool': DEFAULT_SOCKET_CREATOR,
    'GlobalMaxPool': DEFAULT_SOCKET_CREATOR,
    'Greater': Logical_Socket,
    'LayerNorm': DEFAULT_SOCKET_CREATOR,
    'LeakyRelu': DEFAULT_SOCKET_CREATOR,
    'Less': Logical_Socket,
    'LogSoftmax': DEFAULT_SOCKET_CREATOR,
    'MatMul': DEFAULT_SOCKET_CREATOR,
    'Max': DEFAULT_SOCKET_CREATOR,
    'MaxPool': DEFAULT_SOCKET_CREATOR,
    'Min': DEFAULT_SOCKET_CREATOR,
    'Mul': DEFAULT_SOCKET_CREATOR,
    'MultiHeadAttention': DEFAULT_SOCKET_CREATOR,
    'NonMaxSuppression': NonMaxSuppression_Socket,
    'NonZero': NonZero_Socket,
    'Not': DEFAULT_SOCKET_CREATOR,
    'Pad': Pad_Socket,
    'PRelu': DEFAULT_SOCKET_CREATOR,
    'Range': Range_Socket,
    'ReduceL2': DEFAULT_SOCKET_CREATOR,
    'ReduceMax': DEFAULT_SOCKET_CREATOR,
    'ReduceMean': DEFAULT_SOCKET_CREATOR,
    'ReduceSum': DEFAULT_SOCKET_CREATOR,
    'Relu': DEFAULT_SOCKET_CREATOR,
    'Reshape': Reshape_Socket,
    'Resize': Resize_Socket,
    'ScatterElements': ScatterElements_Socket,
    'ScatterND': ScatterND_Socket,
    'Shape': Shape_Socket,
    'Sigmoid': DEFAULT_SOCKET_CREATOR,
    'Slice': Slice_Socket,
    'Softmax': DEFAULT_SOCKET_CREATOR,
    'Softplus': DEFAULT_SOCKET_CREATOR,
    'Split': Split_Socket,
    'Squeeze': Squeeze_Socket,
    'Sub': DEFAULT_SOCKET_CREATOR,
    'Tile': Tile_Socket,
    'TopK': Topk_Socket,
    'Transpose': DEFAULT_SOCKET_CREATOR,
    'Unsqueeze': Unsqueeze_Socket,
    'Where': Where_Socket,
    'Sqrt': DEFAULT_SOCKET_CREATOR,
    'Log': DEFAULT_SOCKET_CREATOR,
    'Floor': DEFAULT_SOCKET_CREATOR,
    'RoiAlign': RoiAlign_Socket,
    'MMCVRoiAlign': RoiAlign_Socket,
    'SpaceToDepth': DEFAULT_SOCKET_CREATOR,
    'DepthToSpace': DEFAULT_SOCKET_CREATOR,
    'Scale': DEFAULT_SOCKET_CREATOR,  # caffe op
    'Tanh': DEFAULT_SOCKET_CREATOR,
    'Pow': DEFAULT_SOCKET_CREATOR,
    'Crop': DEFAULT_SOCKET_CREATOR,  # caffe op
    'ChannelShuffle': DEFAULT_SOCKET_CREATOR,  # caffe op
    'InstanceNormalization': DEFAULT_SOCKET_CREATOR,
    'Parameter': DEFAULT_SOCKET_CREATOR,  # caffe op
    'Interp': DEFAULT_SOCKET_CREATOR,  # caffe op
    'CaffeArgMax': DEFAULT_SOCKET_CREATOR,  # caffe op
    'HardSigmoid': DEFAULT_SOCKET_CREATOR,
    'HardSwish': DEFAULT_SOCKET_CREATOR,
    'Neg': DEFAULT_SOCKET_CREATOR,
    'GRU': DEFAULT_SOCKET_CREATOR,
    'PPQDeviceSwitch': DEFAULT_SOCKET_CREATOR,
    'Identity': DEFAULT_SOCKET_CREATOR,
    'OneHot': Onehot_Socket,
    'Reciprocal': DEFAULT_SOCKET_CREATOR,
    'GreaterOrEqual': Logical_Socket,
    'LessOrEqual': Logical_Socket,
    'Xor': Logical_Socket,
    'Or': Logical_Socket,
    'And': Logical_Socket,
    'Erf': DEFAULT_SOCKET_CREATOR
}
