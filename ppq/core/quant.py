"""PPQ Core Data Structure Abstraction PPQ 核心量化结构抽象.

You are not allowed to modify this 请勿修改此文件
"""

import time  # for hash generation
from enum import Enum
from typing import Any, Iterable, List

import torch

from .common import EXPORT_OVERLAPPED_CONFIG
from .defs import ppq_warning
from .storage import Serializable


class QuantizationVisibility(Enum):
    FORCE_EXPORT       = 1
    EXPOET_WHEN_ACTIVE = 2
    INTERNAL           = 3


class NetworkFramework(Enum):
    PPL     = 1
    ONNX    = 2
    CAFFE   = 3
    NXP     = 4
    NATIVE  = 5


class TargetPlatform(Enum):
    """TargetPlatform is a core abstraction of PPQ framework, it defines
    "platform" as an attribute of an operation. Platform attribute of an
    operation indicates where this operation is going to be deployed. This
    feature enables PPQ to simulate inter-device computing.

    Platform attribute also tells PPQ how to quantize an operation, and how to execute it.
        ATTENTION: Different platform might bring different behaviour of a same operation.
        ATTENTION: Operation which is assigned to an non-quantizible platform will never be quantized.

    There are several supported platforms for PPQ now,
        however you are supposed to be aware of some particular platforms here:

    SHAPE_OR_INDEX is a virtual platform, however it is an EXTREMELY IMPORTANT components in PPQ.
        Dispatch an operation to platform SHAPE_OR_INDEX means this operation is SOI-related,
        it processes a SOI tensor and gives a processed SOI, all calculation of this operation must be sent to CPU
            (or any platform capable for calculating this.) when deploy.

        An operation with SHAPE_OR_INDEX platform assigned will never be quantized regardless of its type.
        It is a crucial feature for quantizing network that contains SOI-related operation. (Shufflenet etc.)

        By default, PPQ automatically detects all SOI-related operations, and dispatch them to SHAPE_OR_INDEX platform.
        To understand how this feature works, see also: ppq.sche

    UNSPECIFIED is a virtual platform, all operations are sent to this platform once they were created.
        Quantizer then dispatches them towards desired platform through its quantization logic.
    """
    TRT_INT8      = 101
    TRT_FP8       = 105
    NCNN_INT8     = 102
    OPENVINO_INT8 = 103
    TENGINE_INT8  = 104
    
    PPL_CUDA_INT8 = 201
    PPL_CUDA_INT4 = 202
    PPL_CUDA_FP16 = 203
    PPL_CUDA_MIX  = 204

    PPL_DSP_INT8  = 301
    SNPE_INT8     = 302
    PPL_DSP_TI_INT8 = 303
    QNN_DSP_INT8  = 304

    HOST_INT8 = 401

    NXP_INT8  = 501
    FPGA_INT8 = 502

    RKNN_INT8 = 601

    METAX_INT8_C = 701 # channel wise
    METAX_INT8_T = 702 # tensor wise
    
    HEXAGON_INT8  = 801
    GRAPHCORE_FP8 = 901

    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8  = 3
    # SHAPE-OR-INDEX
    SOI = -1
    # initial state
    UNSPECIFIED   = -2
    # boundary op
    BOUNDARY      = -3
    # just used for calling exporter
    ONNX          = -4
    CAFFE         = -5
    NATIVE        = -6
    ONNXRUNTIME   = -7
    # THIS IS A DUUMY PLATFORM JUST FOR CREATING YOUR OWN EXTENSION.
    EXTENSION     = -10086

    @ classmethod
    def is_quantized_platform(cls, platform) -> bool:
        return platform in {
            cls.PPL_DSP_INT8, cls.PPL_DSP_TI_INT8, cls.QNN_DSP_INT8, cls.TRT_INT8, cls.NCNN_INT8, cls.NXP_INT8,
            cls.SNPE_INT8, cls.PPL_CUDA_INT8, cls.PPL_CUDA_INT4, cls.EXTENSION, cls.PPL_CUDA_MIX, cls.RKNN_INT8,
            cls.METAX_INT8_C, cls.METAX_INT8_T, cls.OPENVINO_INT8, cls.FPGA_INT8, cls.TENGINE_INT8, 
            cls.FP8, cls.GRAPHCORE_FP8, cls.TRT_FP8}


class RoundingPolicy(Enum):
    """RoundingPolicy is a core setting for PPQ quantization calculation. It
    defines rounding behaviour inside quantization calculation.

    Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

    PPQ Supports 7 different rounding policies now.
    Take a look at https://en.wikipedia.org/wiki/Rounding

    ATTENTION: RoundingPolicy greatly affects PPQ executor behaviour in some cases,
        to get a correct result from PPQ executor,
        make sure your RoundingPolicy is the same as your hardware.
    """
    ROUND_HALF_EVEN            = 0
    ROUND_HALF_UP              = 1
    ROUND_HALF_DOWN            = 2
    ROUND_HALF_TOWARDS_ZERO    = 3
    ROUND_HALF_FAR_FORM_ZERO   = 4
    ROUND_TO_NEAR_INT          = 5
    ROUND_UP                   = 6


class QuantizationProperty(Enum):
    """QuantizationProperty is a core abstraction for PPQ quantization
    calculation. QuantizationProperty and QuantizationPolicy together build a
    bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 8 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Linear quantization, follow formula: quant(x) = clip(round(x / scale))

        FLOATING: Low precision float quantization, FP8, BF16, FP16.

        SYMMETRICAL: Symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Power-of-2 quantization, scale must be pow(2, k) in this mode.

        DYNAMIC: Dynamic Activation Quantization, scale is computed on the fly.

    ATTENTION: Not all combinations of all 8 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    PER_TENSOR   = 0x00000001
    PER_CHANNEL  = 0x00000002
    LINEAR       = 0x00000004
    FLOATING     = 0x00000008
    SYMMETRICAL  = 0x00000010
    ASYMMETRICAL = 0x00000020
    POWER_OF_2   = 0x00000040
    DYNAMIC      = 0x00000080

    def __or__(self, other: int) -> int:
        return self.value + other

    def __ror__(self, other: int) -> int:
        return self.value + other

    def __and__(self, other: int) -> int:
        return self.value & other

    def __rand__(self, other: int) -> int:
        return self.value & other

    def __radd__(self, other: int) -> int:
        return self.value + other

    def __add__(self, other: int) -> int:
        return self.value + other

    def __sub__(self, other: int) -> int:
        return self - (self.value & other)

    def __rsub__(self, other: int) -> int:
        return other - (self.value & other)


class QuantizationPolicy:
    """QuantizationPolicy is a core abstraction for PPQ quantization
    calculation. QuantizationProperty and QuantizationPolicy together build a
    bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 8 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Linear quantization, follow formula: quant(x) = clip(round(x / scale))

        EXPONENTIAL: Exponential quantization, not yet used.

        SYMMETRICAL: Symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Power-of-2 quantization, scale must be pow(2, k) in this mode.

        DYNAMIC: Dynamic Activation Quantization, scale is computed on the fly.

    ATTENTION: Not all combinations of all 8 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    """
    def __init__(self, policy: int) -> None:
        if not QuantizationPolicy.__check_valid(policy):
            raise ValueError(
                'invalid quantization pattern, valid partterns are listed in '
                'ppq.core.OperationQuantizationPolicy.__check_valid'
            )
        self._policy = policy

    def has_property(self, property: QuantizationProperty) -> bool:
        return (self._policy & property.value) != 0

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, QuantizationPolicy):
            raise TypeError('Can only compare QuantizationPolicy object '
                            'with another QuantizationPolicy object.')
        return self._policy == o._policy

    @ classmethod
    def __check_valid(cls, policy):
        return policy in {
            # Standard Int Quantization
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            
            # Low Precision Float Quantization
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_CHANNEL | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.FLOATING | QuantizationProperty.PER_TENSOR | QuantizationProperty.POWER_OF_2,

            # Dynamic Activation Quantization
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_TENSOR | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.SYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
            QuantizationProperty.ASYMMETRICAL | QuantizationProperty.LINEAR | QuantizationProperty.PER_CHANNEL | QuantizationProperty.DYNAMIC | QuantizationProperty.POWER_OF_2,
        }

    def to_dict(self) -> dict:
        """return a dictionary to describe this policy.

        nothing funny.
        """
        return {
            property.name: self.has_property(property)
            for property in QuantizationProperty
        }


class QuantizationStates(Enum):
    """QuantizationStates is a core data structure for PPQ quantization.
    QuantizationStates tells whether a quantization configuration is activated.

    ATTENTION: Changes of QuantizationState will greatly affect execution result.

    For a TensorQuantizationConfig instance, there are 11 available quantization states now.
    Only when state is ACTIVATED or NEGATIVE, corresponding tensor will be quantized during the execution.

    Here we give a brief description of each quantization state:

        INITIAL: given when TensorQuantizationConfig is created, is an initial state of all quantization configuration.

        PASSIVE_INIT: for particular parameter like bias of GEMM(Convolution) and padding value of Pad. Usually it
        does not have an independent quantization scale and offset, while gets quantized with other tensor's configuration.
            For GEMM and Convolution, there bias will be quantized with input scale * weight scale.
            For padding value and clip value, it shares the same scale with its input.
        Those parameters will have a PASSIVE_INIT state when created.

        ATTENTION: if there is any quantization configuration with INITIAL or PASSIVE_INIT state, PPQ will refuse
            to deploy your model and an error will be thrown.
            This inspection will be ignored when PPQ.core.config.DEBUG set as True.

        OVERLAPPED: state OVERLAPPED means there is someone else takes control of current tensor,
        and overlapped tensor quantization configuration will be ignored by optimization algorithms and executor.

        Graph fusion always generate overlapped quantization, for a typical conv - relu fusion,
        the output quantization of convolution will be overlapped by the output tensor of relu.
        State OVERLAPPED cares only about quantization behaviour that cross layers.

        DEACTIVATED: state DEACTIVATED is related with "dequantize" function, once an operation is dequantized,
        all related tensor configurations will be replaced as DEACTIVATED, so that skipping all quantization during
        execution.

        SOI: whenever a tensor quantization configuration holds SOI state,
            it will be never quantized and will not be included into any optimization algorithm.
        it means underlying tensor is SOI-related tensor, and it can not be quantized.

        ACTIVATE: means corresponding tensor is ready to be quantized with its configuration.

        PASSIVE: means corresponding tensor is ready to be quantized with its configuration.
            (however its configuration is not stand alone, its scale and offset depends on someone else.)

        BAKED: means corresponding tensor has been pre-quantized, its value can directly
            go forward without quantization.
    """
    INITIAL     = 1   # 量化参数刚刚被初始化，当前 config 不生效，数据不能被使用
    BAKED       = 2   # 只针对参数量化，表示参数已经被静态量化，当前 config 不生效，数据可以直接使用
    OVERLAPPED  = 3   # 只针对activation量化，表示数据流的量化由其他 config 管理，当前 config 不生效
    DEACTIVATED = 5   # 表示当前 config 不生效
    ACTIVATED   = 6   # 表示当前 config 生效
    DEQUANTIZED = 7   # 表示当前 config 处于解量化状态，解量化是 PPQ 的一个系统操作
    SOI         = 8   # 表示这一路输入与 Shape or index 相关，不量化
    PASSIVE     = 9   # 表示这一路输入被动量化，如 bias, clip value 等，被动量化参数使用其他 TQC 的量化信息完成量化
    PASSIVE_INIT = 10  # 表示这一路输入被动量化，并且刚刚初始化不能被使用
    PASSIVE_BAKED = 11 # 被动量化且静态量化，当前config不生效，数据可以直接使用
    FP32          = 12 # 表示这一路输入直接为FP32浮点数

    @ classmethod
    def is_activated(cls, state)->bool:
        return state in {QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE}

    @ classmethod
    def can_export(cls, state) -> bool:
        return state not in {QuantizationStates.INITIAL, QuantizationStates.DEACTIVATED,
                             QuantizationStates.DEQUANTIZED, QuantizationStates.PASSIVE_INIT}


class TensorQuantizationConfig(Serializable):
    """TensorQuantizationConfig, as known as tensor quantization configuration,
    is the most important data structure in PPQ system.

    PPQ generates quantization configuration for all tensors that need to be quantized, and control their
        quantization logic via this abstraction. As a basic building block of PPQ quantization system, tensor
        quantization is designed to store and manage all quantization related information like:

        Quantization policy, rounding policy, quantization bits, scale, offset, quantization state, etc.

    ATTENTION: tensor(or variable in PPQ) might have more than one quantization configuration, since
        PPQ is designed as an operation-oriented quantization system, so to say tensor quantization configurations
        are created operation by operation. Considering a pattern conv - conv, both the upstream convolution layer
        and the downstream convolution layer will hold a tensor quantization configuration of the middle variable.
        Duplicated quantization configuration will be disabled by optimization pass later.

    PPQ is designed as an operation-oriented quantization system, literally all tensor quantization configurations
        are managed by operations, through you can access their image by variable instance.
        (see the definition of PPQ.IR.quant.QuantableVariable for more information)

    You are supposed to change tensor quantization configuration during optimization passes, this abstraction
        is widely tested among various platforms, it shall satisfy most of your quantization demands.
    """
    def __init__(
        self,
        policy: QuantizationPolicy,
        rounding: RoundingPolicy  = RoundingPolicy.ROUND_HALF_EVEN,
        num_of_bits: int          = 8,
        quant_min: int            = -127,
        quant_max: int            = 128,
        exponent_bits: int        = 0,
        scale: Any                = None,
        offset: Any               = None,
        observer_algorithm: str   = None,
        detail: Any               = None,
        channel_axis: int         = None,
        visiblity: QuantizationVisibility = QuantizationVisibility.EXPOET_WHEN_ACTIVE,
        state: QuantizationStates = QuantizationStates.INITIAL
    ):
        """Create a PPQ Tensor Quantization Configuration Instance.

        Args:
            policy (QuantizationPolicy):
                Quantization policy instance which defines the quantization behaviour from marco view.

            rounding (RoundingPolicy): Rounding policy used in quantization.

            num_of_bits (int): Quantization fraction bits. (2 < num_of_bits < 32)
            
            exponent_bits (int): Quantization exponent bits. (0 < num_of_bits < 8)
                For Int8 Quantization, num_of_bits = 8 and exponent_bits = 0
                For FP8 Quantization, num_of_bits = 4 and exponent_bits = 4

            quant_min (int): An integer value represents the upper bound(inclusive) of quantized value.

            quant_max (int): An integer value represents the lower bound(inclusive) of quantized value.

            scale (Any):
                Scale of quantized value, for per-tensor quantization policy, we use a single float as its scale,
                while for per-channel quantization policy, it will be an array that contains scales for each channel.

            offset (Any): Quantization offset for ASYMMETRICAL quantization policy,
                it will be set as 0 in SYMMETRICAL quantization schema.

            observer_algorithm (str): A string represents an observing algorithm for this tensor.
                PPQ support 'kl', 'minmax' observer now.

            detail (Any, optional): Only used by PPQ internal logic, detail is used to store some internal data,
                you are not supposed to use it.

            channel_axis (int, optional): Only used in PER_CHANNEL quantization, channel index.
        
            visiblity (Visiblity): visiblity is the attribute that controls export logic.

            Currently, there are 3 Visiblity level in PPQ:
            if Visiblity == FORCE_EXPORT, ppq exporter will export this TQC 
                ignoring state check(even if current TQC has been overrlapped).
            if Visiblity == EXPORT_WHEN_ACTIVD, ppq exporter will export this TQC only when it has been actived.
            if Visiblity == INTERNAL, This TQC will not be exported.

            state (QuantizationStates, optional):
                Defaults to QuantizationStates.INITIAL, see QuantizationStates for more detail.
        """

        assert num_of_bits <= 32, 'Cannot quantize a tensor with more than 32 bits.'
        assert num_of_bits >= 2, 'Cannot quantize a tensor with less than 2 bits.'
        assert exponent_bits <= 8, 'Cannot quantize a tensor with more than 8 bits exponent(fp32 overflow).'
        assert exponent_bits >= 0, 'Cannot quantize a tensor with less than 0 bits exponent.'
        
        self._policy = policy
        self._exponent_bits = exponent_bits
        self._num_of_bits = num_of_bits
        self._scale = scale
        self._offset = offset
        self.state = state
        self._rounding = rounding
        self._quant_min = quant_min
        self._quant_max = quant_max
        self._channel_axis = channel_axis
        self.observer_algorithm = observer_algorithm
        self.detail = {} if detail is None else detail
        self._dominator = self # union-find
        self._hash = self.__create_hash()
        self._visiblity = visiblity
        super().__init__()

    def can_export(self) -> bool:
        if self.visiblity == QuantizationVisibility.INTERNAL: return False
        type_check  = isinstance(self.scale, torch.Tensor) and isinstance(self.offset, torch.Tensor)
        valid_states = {QuantizationStates.BAKED, QuantizationStates.PASSIVE_BAKED}

        if EXPORT_OVERLAPPED_CONFIG: valid_states.add(QuantizationStates.OVERLAPPED)
        state_check = QuantizationStates.is_activated(self.state) or self.state in valid_states

        if (state_check or self.visiblity == QuantizationVisibility.FORCE_EXPORT):
            if type_check: return True
        return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError('Can only compare TensorQuantizationConfig object '
                            'with another TensorQuantizationConfig object.')
        return self._hash == o._hash

    def __str__(self) -> str:
        return f'PPQ TensorQuantizationConfig({self.__hash__()})'

    _hash_seed = int(time.time())
    @ staticmethod
    def __create_hash():
        TensorQuantizationConfig._hash_seed = (
            0x343FD * TensorQuantizationConfig._hash_seed + 0x269EC3) % (2 << 31)
        return TensorQuantizationConfig._hash_seed

    def __hash__(self) -> int:
        return self._hash

    def is_same_scheme(self, o: object) -> bool:
        if not isinstance(o, TensorQuantizationConfig):
            raise TypeError('Can only compare TensorQuantizationConfig object '
                            'with another TensorQuantizationConfig object.')
        return (self.quant_max == o.quant_max and 
                self.quant_min == o.quant_min and 
                self.policy == o.policy and 
                self.num_of_bits == o.num_of_bits and
                self.exponent_bits == o.exponent_bits and
                self.channel_axis == o.channel_axis and
                self.rounding == o.rounding)

    @ property
    def dominated_by(self):
        """dominated_by is a crucial feature for tensor quantization
        configuration in PPQ. This property is actually maintained by union-
        find set data structure.

        Every tensor quantization configuration(A) is created with dominated_by = self, and only when
            it is overlapped by other configuration(B), it shall set A.dominated_by = B.
            Setting A.dominated_by = B also makes A, B as a quantization group.
            (quantization state of A is always set as OVERLAPPED here)

        So to say every tensor quantization configuration with dominated_by != self is overrlaped by
            other quantization configuration. When a tensor quantization configuration is overlapped,
            it means this tensor is already been quantized with another quantization configuration,
            and there is no need to be quantized with this configuration anymore.

        PPQ use this property to find root configuration for each configuration group,

        Returns:
            [TensorQuantizationConfig]: root configuration of this quantization group.

        ATTENTION: This configuration is invalid when self.dominated_by != self.
        """
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @ dominated_by.setter
    def dominated_by(self, o):
        assert isinstance(o, TensorQuantizationConfig), (
            'Can only set this attribute with another tensor config.')
        if o._hash == self._hash:
            raise ValueError('Error with TQC.dominated_by = o: o must not equal to TQC its self.')
        root, dominator = self.dominated_by, o.dominated_by
        assert isinstance(root, TensorQuantizationConfig)
        if dominator != root:
            root._dominator = dominator
            self._dominator = dominator
            root.state = QuantizationStates.OVERLAPPED
            self.state = QuantizationStates.OVERLAPPED

    @ property
    def master_by(self):
        if self._dominator == self:
            return self
        else:
            root = self._dominator.dominated_by
            self._dominator = root
            return root

    @ master_by.setter
    def master_by(self, master):
        if not isinstance(master, TensorQuantizationConfig):
            raise TypeError(f'Error with TQC.master_by(o): o must be another Tensor Quantization Config, '
                            f'however {type(master)} was given.')
        if master._hash == self._hash:
            raise ValueError('Error with TQC.dominated_by = o: o must not equal to TQC its self.')
        self._dominator = master
        if master.scale is not None and master.offset is not None:
            self.state   = QuantizationStates.PASSIVE
        else: self.state = QuantizationStates.PASSIVE_INIT

    def is_revisable(self):
        return (self.dominated_by == self and self.state in {
            QuantizationStates.ACTIVATED,
            QuantizationStates.DEQUANTIZED,
            QuantizationStates.DEACTIVATED,
            QuantizationStates.INITIAL,
            QuantizationStates.FP32,
            QuantizationStates.PASSIVE,
            QuantizationStates.PASSIVE_INIT
        })

    @ property
    def visiblity(self) -> bool:
        return self._visiblity

    @ visiblity.setter
    def visiblity(self, visiblity: bool):
        self._visiblity = visiblity

    @ property
    def scale(self) -> torch.Tensor:
        if self.dominated_by == self: return self._scale
        else: return self.dominated_by.scale

    @ property
    def offset(self) -> torch.Tensor:
        if self.dominated_by == self: return self._offset
        else: return self.dominated_by.offset

    @ property
    def policy(self) -> QuantizationPolicy:
        return self._policy

    @ property
    def num_of_bits(self) -> int:
        return self._num_of_bits

    @ property
    def rounding(self) -> RoundingPolicy:
        return self._rounding

    @ property
    def quant_min(self) -> int:
        return self._quant_min

    @ property
    def quant_max(self) -> int:
        return self._quant_max

    @ property
    def exponent_bits(self) -> int:
        return self._exponent_bits

    @ property
    def mantissa_bits(self) -> int:
        # there is one bit for sign.
        return self.num_of_bits - self._exponent_bits - 1
    
    @ property
    def channel_axis(self) -> int:
        return self._channel_axis

    @ scale.setter
    def scale(self, value: Any):
        if not self.is_revisable():
            raise PermissionError(
                'Can not change scale of this tensor quantization configuration now. '
                'It has been overlapped or has an inactive state. '
                'Due to it is not a active config, any change of this configuration is not allowed.'
            )
        else:
            self._scale = value

    @ offset.setter
    def offset(self, value: Any):
        if not self.is_revisable():
            raise PermissionError(
                'Can not change offset of this tensor quantization configuration now. '
                'It has been overlapped or has an inactive state. '
                'Due to it is not a active config, any change of this configuration is not allowed.'
            )
        else:
            self._offset = value

    @ policy.setter
    def policy(self, policy: QuantizationPolicy):
        self._policy = policy

    @ num_of_bits.setter
    def num_of_bits(self, bits: int):
        self._num_of_bits = bits

    @ rounding.setter
    def rounding(self, policy: RoundingPolicy):
        self._rounding = policy

    @ quant_min.setter
    def quant_min(self, min: int):
        self._quant_min = min

    @ quant_max.setter
    def quant_max(self, max: int):
        self._quant_max = max

    @ exponent_bits.setter
    def exponent_bits(self, bits: int):
        if not self.policy.has_property(QuantizationProperty.FLOATING):
            raise PermissionError(
                'Can not change property: exponent bits for this TQC. '
                'self.policy.has_property(QuantizationProperty.FLOATING) == False.')
        self._exponent_bits = bits

    @ channel_axis.setter
    def channel_axis(self, channel_axis: int):
        if not self.policy.has_property(QuantizationProperty.PER_CHANNEL):
            raise PermissionError(
                'Can not change property: quantization channel axis for this TQC. '
                'self.policy.has_property(QuantizationProperty.PER_CHANNEL) == False.')
        self._channel_axis = channel_axis

    def copy(self):
        """Create a tensor config from this one, keep policy and state
        unchanged.

        if there is an non-empty scale and offset, they will be cloned too.
        """
        scale, offset = None, None
        if self.scale is not None:
            if isinstance(self.scale, torch.Tensor):
                scale = self.scale.clone()
            else: scale = self.scale
        if self.offset is not None:
            if isinstance(self.offset, torch.Tensor):
                offset = self.offset.clone()
            else: offset = self.offset
        config = TensorQuantizationConfig(
            policy=self.policy,
            rounding=self.rounding,
            num_of_bits=self.num_of_bits,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            scale=scale, offset=offset,
            observer_algorithm=self.observer_algorithm,
            detail=self.detail.copy(),
            state=self.state,
            exponent_bits=self.exponent_bits,
            channel_axis=self.channel_axis
        )
        if self.state == QuantizationStates.OVERLAPPED:
            config._dominator = self._dominator
        return config


class ChannelwiseTensorQuantizationConfig(TensorQuantizationConfig):
    """ Legacy Class Since PPQ 0.6.6, Use TensorQuantizationConfig Instead. """
    def __init__(self,
        policy: QuantizationPolicy, rounding:RoundingPolicy,
        num_of_bits: int, exponent_bits: int, 
        quant_min: int, quant_max: int,
        scale: Any, offset: Any, observer_algorithm: str,
        state: QuantizationStates, channel_axis: int, detail: dict = {}
    ):
        ppq_warning('ChannelwiseTensorQuantizationConfig is now obsolescent(Since PPQ 0.6.6), '
                    'use TensorQuantizationConfig Instead.')
        if policy.has_property(QuantizationProperty.PER_TENSOR):
            raise TypeError('Can not assign QuantizationProperty.PER_TENSOR policy '
                'to a Channel-wise Tensor Quantization Config instance.')
        super().__init__(
            policy=policy, num_of_bits=num_of_bits,
            quant_min=quant_min, quant_max=quant_max, scale=scale, offset=offset,
            observer_algorithm=observer_algorithm, detail=detail, state=state,
            rounding=rounding, exponent_bits=exponent_bits
        )
        self.channel_axis = channel_axis

    @ classmethod
    def convert_from_tensor_config(cls,
        convert_from: TensorQuantizationConfig,
        scale: Iterable = None,
        offset: Iterable = None,
        channel_axis: int = 1,
    ):
        if scale is None: scale = convert_from.scale
        if offset is None: offset = convert_from.offset
        this = ChannelwiseTensorQuantizationConfig(
            policy=convert_from.policy,
            num_of_bits=convert_from.num_of_bits,
            quant_min=convert_from.quant_min,
            quant_max=convert_from.quant_max,
            scale=scale, offset=offset,
            observer_algorithm=convert_from.observer_algorithm,
            detail=convert_from.detail.copy(),
            state=convert_from.state,
            channel_axis=channel_axis,
            rounding=convert_from.rounding,
            exponent_bits=convert_from.exponent_bits
        )
        return this

    def copy(self):
        config = super().copy()
        return self.convert_from_tensor_config(
            config, scale=config.scale, offset=config.offset,
            channel_axis=self.channel_axis)


class OperationQuantizationConfig(Iterable):
    """OperationQuantizationConfig serves as a collection of tensor
    quantization configuration.

    See TensorQuantizationConfig for more information.
    """
    def __init__(
        self,
        input_quantization_configs: List[TensorQuantizationConfig] = None,
        output_quantization_configs: List[TensorQuantizationConfig] = None,
        is_positive_quant_op: bool = True
    ):
        """Create an operation quantization configuration.

        Args:
            input_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all input variables.

            output_quantization_configs (List[TensorQuantizationConfig], optional):
                a list contains all configuration of all output variables.

            ATTENTION: whether a variable is gonna to be quantized or not, it must have a quantization configuration.

            is_positive_quant_op (bool, optional): [description]. Defaults to True.
                some operations are passively quantized, such as Maxpooling, Padding.
                For those operations, set this property as False, PPQ will use this property to optimize your graph.
        """
        self.input_quantization_config     = self.__check_famliy_config(input_quantization_configs)
        self.output_quantization_config    = self.__check_famliy_config(output_quantization_configs)
        self.is_active_quant_op = is_positive_quant_op

    def export(self) -> str:
        raise Exception('Implement this first')

    def __check_famliy_config(self, famliy_configs):
        for famliy_config in famliy_configs:
            if not isinstance(famliy_config, TensorQuantizationConfig):
                raise TypeError(
                    f'You are trying to set famliy quantization config of {str(self)}, ' \
                    f'However your input is invalid, except one TensorQuantizationConfig object, ' \
                    f'while a {type(famliy_config)} was given.'
                )
        return famliy_configs

    def __str__(self) -> str:
        return f'Inputs config: {self.input_quantization_config}, '\
            f'Outputs config {self.output_quantization_config}'

    def __iter__(self) -> TensorQuantizationConfig:
        return (self.input_quantization_config + self.output_quantization_config).__iter__()

    def copy(self):
        """Create an operation config from this one, keep policy and state
        unchanged.

        if this one has an non-empty scale or offset, they will be cloned too.
        """
        return OperationQuantizationConfig(
            input_quantization_configs=[_.copy() for _ in self.input_quantization_config],
            output_quantization_configs=[_.copy() for _ in self.output_quantization_config],
            is_positive_quant_op=self.is_active_quant_op
        )
