from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Phase(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    TRAIN: _ClassVar[Phase]
    TEST: _ClassVar[Phase]

class InterpType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
    INTERP_BILINEAR: _ClassVar[InterpType]
    INTERP_NEAREST: _ClassVar[InterpType]
TRAIN: Phase
TEST: Phase
INTERP_BILINEAR: InterpType
INTERP_NEAREST: InterpType

class BlobShape(_message.Message):
    __slots__ = ["dim"]
    DIM_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dim: _Optional[_Iterable[int]] = ...) -> None: ...

class BlobProto(_message.Message):
    __slots__ = ["shape", "data", "diff", "num", "channels", "height", "width", "depth"]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    DIFF_FIELD_NUMBER: _ClassVar[int]
    NUM_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    shape: BlobShape
    data: _containers.RepeatedScalarFieldContainer[float]
    diff: _containers.RepeatedScalarFieldContainer[float]
    num: int
    channels: int
    height: int
    width: int
    depth: int
    def __init__(self, shape: _Optional[_Union[BlobShape, _Mapping]] = ..., data: _Optional[_Iterable[float]] = ..., diff: _Optional[_Iterable[float]] = ..., num: _Optional[int] = ..., channels: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., depth: _Optional[int] = ...) -> None: ...

class BlobProtoVector(_message.Message):
    __slots__ = ["blobs"]
    BLOBS_FIELD_NUMBER: _ClassVar[int]
    blobs: _containers.RepeatedCompositeFieldContainer[BlobProto]
    def __init__(self, blobs: _Optional[_Iterable[_Union[BlobProto, _Mapping]]] = ...) -> None: ...

class QuantizeParameter(_message.Message):
    __slots__ = ["type", "step", "range_min", "range_max", "zero_point", "adjust_range"]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    RANGE_MIN_FIELD_NUMBER: _ClassVar[int]
    RANGE_MAX_FIELD_NUMBER: _ClassVar[int]
    ZERO_POINT_FIELD_NUMBER: _ClassVar[int]
    ADJUST_RANGE_FIELD_NUMBER: _ClassVar[int]
    type: str
    step: float
    range_min: float
    range_max: float
    zero_point: int
    adjust_range: bool
    def __init__(self, type: _Optional[str] = ..., step: _Optional[float] = ..., range_min: _Optional[float] = ..., range_max: _Optional[float] = ..., zero_point: _Optional[int] = ..., adjust_range: bool = ...) -> None: ...

class Datum(_message.Message):
    __slots__ = ["channels", "height", "width", "data", "label", "float_data", "encoded", "depth"]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    ENCODED_FIELD_NUMBER: _ClassVar[int]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    channels: int
    height: int
    width: int
    data: bytes
    label: int
    float_data: _containers.RepeatedScalarFieldContainer[float]
    encoded: bool
    depth: int
    def __init__(self, channels: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., data: _Optional[bytes] = ..., label: _Optional[int] = ..., float_data: _Optional[_Iterable[float]] = ..., encoded: bool = ..., depth: _Optional[int] = ...) -> None: ...

class FillerParameter(_message.Message):
    __slots__ = ["type", "value", "min", "max", "mean", "std", "sparse"]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MEAN_FIELD_NUMBER: _ClassVar[int]
    STD_FIELD_NUMBER: _ClassVar[int]
    SPARSE_FIELD_NUMBER: _ClassVar[int]
    type: str
    value: float
    min: float
    max: float
    mean: float
    std: float
    sparse: int
    def __init__(self, type: _Optional[str] = ..., value: _Optional[float] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., mean: _Optional[float] = ..., std: _Optional[float] = ..., sparse: _Optional[int] = ...) -> None: ...

class NetParameter(_message.Message):
    __slots__ = ["name", "input", "input_shape", "input_dim", "force_backward", "state", "debug_info", "layer", "layers"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    INPUT_SHAPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_DIM_FIELD_NUMBER: _ClassVar[int]
    FORCE_BACKWARD_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    DEBUG_INFO_FIELD_NUMBER: _ClassVar[int]
    LAYER_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    input: _containers.RepeatedScalarFieldContainer[str]
    input_shape: _containers.RepeatedCompositeFieldContainer[BlobShape]
    input_dim: _containers.RepeatedScalarFieldContainer[int]
    force_backward: bool
    state: NetState
    debug_info: bool
    layer: _containers.RepeatedCompositeFieldContainer[LayerParameter]
    layers: _containers.RepeatedCompositeFieldContainer[V1LayerParameter]
    def __init__(self, name: _Optional[str] = ..., input: _Optional[_Iterable[str]] = ..., input_shape: _Optional[_Iterable[_Union[BlobShape, _Mapping]]] = ..., input_dim: _Optional[_Iterable[int]] = ..., force_backward: bool = ..., state: _Optional[_Union[NetState, _Mapping]] = ..., debug_info: bool = ..., layer: _Optional[_Iterable[_Union[LayerParameter, _Mapping]]] = ..., layers: _Optional[_Iterable[_Union[V1LayerParameter, _Mapping]]] = ...) -> None: ...

class SolverParameter(_message.Message):
    __slots__ = ["net", "net_param", "train_net", "test_net", "train_net_param", "test_net_param", "train_state", "test_state", "test_iter", "test_interval", "test_compute_loss", "test_initialization", "base_lr", "display", "average_loss", "max_iter", "lr_policy", "gamma", "power", "momentum", "weight_decay", "regularization_type", "stepsize", "stepvalue", "clip_gradients", "snapshot", "snapshot_prefix", "snapshot_diff", "solver_mode", "device_id", "random_seed", "solver_type", "delta", "debug_info", "snapshot_after_train"]
    class SolverMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        CPU: _ClassVar[SolverParameter.SolverMode]
        GPU: _ClassVar[SolverParameter.SolverMode]
    CPU: SolverParameter.SolverMode
    GPU: SolverParameter.SolverMode
    class SolverType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        SGD: _ClassVar[SolverParameter.SolverType]
        NESTEROV: _ClassVar[SolverParameter.SolverType]
        ADAGRAD: _ClassVar[SolverParameter.SolverType]
    SGD: SolverParameter.SolverType
    NESTEROV: SolverParameter.SolverType
    ADAGRAD: SolverParameter.SolverType
    NET_FIELD_NUMBER: _ClassVar[int]
    NET_PARAM_FIELD_NUMBER: _ClassVar[int]
    TRAIN_NET_FIELD_NUMBER: _ClassVar[int]
    TEST_NET_FIELD_NUMBER: _ClassVar[int]
    TRAIN_NET_PARAM_FIELD_NUMBER: _ClassVar[int]
    TEST_NET_PARAM_FIELD_NUMBER: _ClassVar[int]
    TRAIN_STATE_FIELD_NUMBER: _ClassVar[int]
    TEST_STATE_FIELD_NUMBER: _ClassVar[int]
    TEST_ITER_FIELD_NUMBER: _ClassVar[int]
    TEST_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    TEST_COMPUTE_LOSS_FIELD_NUMBER: _ClassVar[int]
    TEST_INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    BASE_LR_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_LOSS_FIELD_NUMBER: _ClassVar[int]
    MAX_ITER_FIELD_NUMBER: _ClassVar[int]
    LR_POLICY_FIELD_NUMBER: _ClassVar[int]
    GAMMA_FIELD_NUMBER: _ClassVar[int]
    POWER_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    REGULARIZATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    STEPSIZE_FIELD_NUMBER: _ClassVar[int]
    STEPVALUE_FIELD_NUMBER: _ClassVar[int]
    CLIP_GRADIENTS_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_PREFIX_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_DIFF_FIELD_NUMBER: _ClassVar[int]
    SOLVER_MODE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    RANDOM_SEED_FIELD_NUMBER: _ClassVar[int]
    SOLVER_TYPE_FIELD_NUMBER: _ClassVar[int]
    DELTA_FIELD_NUMBER: _ClassVar[int]
    DEBUG_INFO_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_AFTER_TRAIN_FIELD_NUMBER: _ClassVar[int]
    net: str
    net_param: NetParameter
    train_net: str
    test_net: _containers.RepeatedScalarFieldContainer[str]
    train_net_param: NetParameter
    test_net_param: _containers.RepeatedCompositeFieldContainer[NetParameter]
    train_state: NetState
    test_state: _containers.RepeatedCompositeFieldContainer[NetState]
    test_iter: _containers.RepeatedScalarFieldContainer[int]
    test_interval: int
    test_compute_loss: bool
    test_initialization: bool
    base_lr: float
    display: int
    average_loss: int
    max_iter: int
    lr_policy: str
    gamma: float
    power: float
    momentum: float
    weight_decay: float
    regularization_type: str
    stepsize: int
    stepvalue: _containers.RepeatedScalarFieldContainer[int]
    clip_gradients: float
    snapshot: int
    snapshot_prefix: str
    snapshot_diff: bool
    solver_mode: SolverParameter.SolverMode
    device_id: int
    random_seed: int
    solver_type: SolverParameter.SolverType
    delta: float
    debug_info: bool
    snapshot_after_train: bool
    def __init__(self, net: _Optional[str] = ..., net_param: _Optional[_Union[NetParameter, _Mapping]] = ..., train_net: _Optional[str] = ..., test_net: _Optional[_Iterable[str]] = ..., train_net_param: _Optional[_Union[NetParameter, _Mapping]] = ..., test_net_param: _Optional[_Iterable[_Union[NetParameter, _Mapping]]] = ..., train_state: _Optional[_Union[NetState, _Mapping]] = ..., test_state: _Optional[_Iterable[_Union[NetState, _Mapping]]] = ..., test_iter: _Optional[_Iterable[int]] = ..., test_interval: _Optional[int] = ..., test_compute_loss: bool = ..., test_initialization: bool = ..., base_lr: _Optional[float] = ..., display: _Optional[int] = ..., average_loss: _Optional[int] = ..., max_iter: _Optional[int] = ..., lr_policy: _Optional[str] = ..., gamma: _Optional[float] = ..., power: _Optional[float] = ..., momentum: _Optional[float] = ..., weight_decay: _Optional[float] = ..., regularization_type: _Optional[str] = ..., stepsize: _Optional[int] = ..., stepvalue: _Optional[_Iterable[int]] = ..., clip_gradients: _Optional[float] = ..., snapshot: _Optional[int] = ..., snapshot_prefix: _Optional[str] = ..., snapshot_diff: bool = ..., solver_mode: _Optional[_Union[SolverParameter.SolverMode, str]] = ..., device_id: _Optional[int] = ..., random_seed: _Optional[int] = ..., solver_type: _Optional[_Union[SolverParameter.SolverType, str]] = ..., delta: _Optional[float] = ..., debug_info: bool = ..., snapshot_after_train: bool = ...) -> None: ...

class SolverState(_message.Message):
    __slots__ = ["iter", "learned_net", "history", "current_step"]
    ITER_FIELD_NUMBER: _ClassVar[int]
    LEARNED_NET_FIELD_NUMBER: _ClassVar[int]
    HISTORY_FIELD_NUMBER: _ClassVar[int]
    CURRENT_STEP_FIELD_NUMBER: _ClassVar[int]
    iter: int
    learned_net: str
    history: _containers.RepeatedCompositeFieldContainer[BlobProto]
    current_step: int
    def __init__(self, iter: _Optional[int] = ..., learned_net: _Optional[str] = ..., history: _Optional[_Iterable[_Union[BlobProto, _Mapping]]] = ..., current_step: _Optional[int] = ...) -> None: ...

class NetState(_message.Message):
    __slots__ = ["phase", "level", "stage"]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    phase: Phase
    level: int
    stage: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, phase: _Optional[_Union[Phase, str]] = ..., level: _Optional[int] = ..., stage: _Optional[_Iterable[str]] = ...) -> None: ...

class NetStateRule(_message.Message):
    __slots__ = ["phase", "min_level", "max_level", "stage", "not_stage"]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    MIN_LEVEL_FIELD_NUMBER: _ClassVar[int]
    MAX_LEVEL_FIELD_NUMBER: _ClassVar[int]
    STAGE_FIELD_NUMBER: _ClassVar[int]
    NOT_STAGE_FIELD_NUMBER: _ClassVar[int]
    phase: Phase
    min_level: int
    max_level: int
    stage: _containers.RepeatedScalarFieldContainer[str]
    not_stage: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, phase: _Optional[_Union[Phase, str]] = ..., min_level: _Optional[int] = ..., max_level: _Optional[int] = ..., stage: _Optional[_Iterable[str]] = ..., not_stage: _Optional[_Iterable[str]] = ...) -> None: ...

class ParamSpec(_message.Message):
    __slots__ = ["name", "share_mode", "lr_mult", "decay_mult"]
    class DimCheckMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        STRICT: _ClassVar[ParamSpec.DimCheckMode]
        PERMISSIVE: _ClassVar[ParamSpec.DimCheckMode]
    STRICT: ParamSpec.DimCheckMode
    PERMISSIVE: ParamSpec.DimCheckMode
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHARE_MODE_FIELD_NUMBER: _ClassVar[int]
    LR_MULT_FIELD_NUMBER: _ClassVar[int]
    DECAY_MULT_FIELD_NUMBER: _ClassVar[int]
    name: str
    share_mode: ParamSpec.DimCheckMode
    lr_mult: float
    decay_mult: float
    def __init__(self, name: _Optional[str] = ..., share_mode: _Optional[_Union[ParamSpec.DimCheckMode, str]] = ..., lr_mult: _Optional[float] = ..., decay_mult: _Optional[float] = ...) -> None: ...

class LayerParameter(_message.Message):
    __slots__ = ["name", "type", "bottom", "top", "phase", "loss_weight", "param", "blobs", "include", "exclude", "top_data_type", "forward_precision", "transform_param", "loss_param", "accuracy_param", "argmax_param", "concat_param", "contrastive_loss_param", "convolution_param", "data_param", "dropout_param", "dummy_data_param", "eltwise_param", "exp_param", "hdf5_data_param", "hdf5_output_param", "hinge_loss_param", "image_data_param", "infogain_loss_param", "inner_product_param", "lrn_param", "memory_data_param", "mvn_param", "pooling_param", "pooling_specific_param", "power_param", "relu_param", "sigmoid_param", "softmax_param", "slice_param", "tanh_param", "threshold_param", "window_data_param", "python_param", "ctc_param", "prelu_param", "reshape_param", "roi_pooling_param", "affine_trans_param", "roi_param", "affine_trans_point_param", "permute_param", "prior_box_param", "calc_affine_mat_param", "interp_param", "detection_output_param", "relu6_param", "bias_param", "zxybn_param", "correlation_param", "psroi_pooling_param", "nn_upsample_param", "roi_align_param", "rpn_proposal_param", "roi_mask_pooling_param", "channel_shuffle_param", "bn_param", "roi_align_pooling_pod_param", "roi_align_pooling_param", "psroi_align_pooling_param", "alignment_to_roi_param", "unpooling_param", "log_param", "reverse_param", "lstm_param", "prior_vbox_param", "btcostvolume_param", "moving_avg_param", "slicing_param", "bilateral_slicing_param", "bilateral_slice_apply_param", "adaptive_bilateral_slice_apply_param", "pointscurve_param", "subpixel_down_param", "subpixel_up_param", "clip_param", "eltwise_affine_transform_param", "tile_param", "pad_param", "reduce_param", "convolution3d_param", "deconvolution3d_param", "relu3d_param", "prelu3d_param", "leaky_relu3d_param", "elu3d_param", "groupnorm3d_param", "batchnorm3d_param", "maxpooling3d_param", "avepooling3d_param", "dropout3d_param", "interp3d_param", "pooling3d_param", "reshape3d_param", "softmax3d_param", "argmax3d_param", "inner_product3d_param", "slice3d_param", "concat3d_param", "bn3d_param", "matmul3d_param", "transpose3d_param", "scale_param", "crop_param", "recurrent_param", "batch_norm_param", "flatten_param", "scales_param", "transpose_param", "heatmap_param", "normalize_param", "correlation2d_param", "matmul_param", "parameter_param", "pixelshuffle_param", "roi_transform_param", "instance_norm_param", "grid_sample_param", "reducel2_param", "mean_param", "variance_param", "grid_sample_3d_param", "gru_param", "correlationmig_param", "argsort_param", "cumprod_param", "main_transform_param", "quantize_param", "quantize_bits"]
    class PrecisionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[LayerParameter.PrecisionType]
        INT8: _ClassVar[LayerParameter.PrecisionType]
        FLOAT16: _ClassVar[LayerParameter.PrecisionType]
        FLOAT32: _ClassVar[LayerParameter.PrecisionType]
        INT4B: _ClassVar[LayerParameter.PrecisionType]
        UINT8: _ClassVar[LayerParameter.PrecisionType]
        UINT16: _ClassVar[LayerParameter.PrecisionType]
    DEFAULT: LayerParameter.PrecisionType
    INT8: LayerParameter.PrecisionType
    FLOAT16: LayerParameter.PrecisionType
    FLOAT32: LayerParameter.PrecisionType
    INT4B: LayerParameter.PrecisionType
    UINT8: LayerParameter.PrecisionType
    UINT16: LayerParameter.PrecisionType
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    LOSS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    PARAM_FIELD_NUMBER: _ClassVar[int]
    BLOBS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_FIELD_NUMBER: _ClassVar[int]
    TOP_DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    FORWARD_PRECISION_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_PARAM_FIELD_NUMBER: _ClassVar[int]
    ARGMAX_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONCAT_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONTRASTIVE_LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONVOLUTION_PARAM_FIELD_NUMBER: _ClassVar[int]
    DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    DROPOUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    DUMMY_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    ELTWISE_PARAM_FIELD_NUMBER: _ClassVar[int]
    EXP_PARAM_FIELD_NUMBER: _ClassVar[int]
    HDF5_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    HDF5_OUTPUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    HINGE_LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    INFOGAIN_LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    INNER_PRODUCT_PARAM_FIELD_NUMBER: _ClassVar[int]
    LRN_PARAM_FIELD_NUMBER: _ClassVar[int]
    MEMORY_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    MVN_PARAM_FIELD_NUMBER: _ClassVar[int]
    POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    POOLING_SPECIFIC_PARAM_FIELD_NUMBER: _ClassVar[int]
    POWER_PARAM_FIELD_NUMBER: _ClassVar[int]
    RELU_PARAM_FIELD_NUMBER: _ClassVar[int]
    SIGMOID_PARAM_FIELD_NUMBER: _ClassVar[int]
    SOFTMAX_PARAM_FIELD_NUMBER: _ClassVar[int]
    SLICE_PARAM_FIELD_NUMBER: _ClassVar[int]
    TANH_PARAM_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_PARAM_FIELD_NUMBER: _ClassVar[int]
    WINDOW_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    PYTHON_PARAM_FIELD_NUMBER: _ClassVar[int]
    CTC_PARAM_FIELD_NUMBER: _ClassVar[int]
    PRELU_PARAM_FIELD_NUMBER: _ClassVar[int]
    RESHAPE_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    AFFINE_TRANS_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_PARAM_FIELD_NUMBER: _ClassVar[int]
    AFFINE_TRANS_POINT_PARAM_FIELD_NUMBER: _ClassVar[int]
    PERMUTE_PARAM_FIELD_NUMBER: _ClassVar[int]
    PRIOR_BOX_PARAM_FIELD_NUMBER: _ClassVar[int]
    CALC_AFFINE_MAT_PARAM_FIELD_NUMBER: _ClassVar[int]
    INTERP_PARAM_FIELD_NUMBER: _ClassVar[int]
    DETECTION_OUTPUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    RELU6_PARAM_FIELD_NUMBER: _ClassVar[int]
    BIAS_PARAM_FIELD_NUMBER: _ClassVar[int]
    ZXYBN_PARAM_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_PARAM_FIELD_NUMBER: _ClassVar[int]
    PSROI_POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    NN_UPSAMPLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_ALIGN_PARAM_FIELD_NUMBER: _ClassVar[int]
    RPN_PROPOSAL_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_MASK_POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_SHUFFLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    BN_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_ALIGN_POOLING_POD_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_ALIGN_POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    PSROI_ALIGN_POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    ALIGNMENT_TO_ROI_PARAM_FIELD_NUMBER: _ClassVar[int]
    UNPOOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    LOG_PARAM_FIELD_NUMBER: _ClassVar[int]
    REVERSE_PARAM_FIELD_NUMBER: _ClassVar[int]
    LSTM_PARAM_FIELD_NUMBER: _ClassVar[int]
    PRIOR_VBOX_PARAM_FIELD_NUMBER: _ClassVar[int]
    BTCOSTVOLUME_PARAM_FIELD_NUMBER: _ClassVar[int]
    MOVING_AVG_PARAM_FIELD_NUMBER: _ClassVar[int]
    SLICING_PARAM_FIELD_NUMBER: _ClassVar[int]
    BILATERAL_SLICING_PARAM_FIELD_NUMBER: _ClassVar[int]
    BILATERAL_SLICE_APPLY_PARAM_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_BILATERAL_SLICE_APPLY_PARAM_FIELD_NUMBER: _ClassVar[int]
    POINTSCURVE_PARAM_FIELD_NUMBER: _ClassVar[int]
    SUBPIXEL_DOWN_PARAM_FIELD_NUMBER: _ClassVar[int]
    SUBPIXEL_UP_PARAM_FIELD_NUMBER: _ClassVar[int]
    CLIP_PARAM_FIELD_NUMBER: _ClassVar[int]
    ELTWISE_AFFINE_TRANSFORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    TILE_PARAM_FIELD_NUMBER: _ClassVar[int]
    PAD_PARAM_FIELD_NUMBER: _ClassVar[int]
    REDUCE_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONVOLUTION3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    DECONVOLUTION3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    RELU3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    PRELU3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    LEAKY_RELU3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    ELU3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    GROUPNORM3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    BATCHNORM3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    MAXPOOLING3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    AVEPOOLING3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    DROPOUT3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    INTERP3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    POOLING3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    RESHAPE3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    SOFTMAX3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    ARGMAX3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    INNER_PRODUCT3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    SLICE3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONCAT3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    BN3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    MATMUL3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    TRANSPOSE3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    SCALE_PARAM_FIELD_NUMBER: _ClassVar[int]
    CROP_PARAM_FIELD_NUMBER: _ClassVar[int]
    RECURRENT_PARAM_FIELD_NUMBER: _ClassVar[int]
    BATCH_NORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    FLATTEN_PARAM_FIELD_NUMBER: _ClassVar[int]
    SCALES_PARAM_FIELD_NUMBER: _ClassVar[int]
    TRANSPOSE_PARAM_FIELD_NUMBER: _ClassVar[int]
    HEATMAP_PARAM_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    CORRELATION2D_PARAM_FIELD_NUMBER: _ClassVar[int]
    MATMUL_PARAM_FIELD_NUMBER: _ClassVar[int]
    PARAMETER_PARAM_FIELD_NUMBER: _ClassVar[int]
    PIXELSHUFFLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    ROI_TRANSFORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_NORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    GRID_SAMPLE_PARAM_FIELD_NUMBER: _ClassVar[int]
    REDUCEL2_PARAM_FIELD_NUMBER: _ClassVar[int]
    MEAN_PARAM_FIELD_NUMBER: _ClassVar[int]
    VARIANCE_PARAM_FIELD_NUMBER: _ClassVar[int]
    GRID_SAMPLE_3D_PARAM_FIELD_NUMBER: _ClassVar[int]
    GRU_PARAM_FIELD_NUMBER: _ClassVar[int]
    CORRELATIONMIG_PARAM_FIELD_NUMBER: _ClassVar[int]
    ARGSORT_PARAM_FIELD_NUMBER: _ClassVar[int]
    CUMPROD_PARAM_FIELD_NUMBER: _ClassVar[int]
    MAIN_TRANSFORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    QUANTIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    QUANTIZE_BITS_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    bottom: _containers.RepeatedScalarFieldContainer[str]
    top: _containers.RepeatedScalarFieldContainer[str]
    phase: Phase
    loss_weight: _containers.RepeatedScalarFieldContainer[float]
    param: _containers.RepeatedCompositeFieldContainer[ParamSpec]
    blobs: _containers.RepeatedCompositeFieldContainer[BlobProto]
    include: _containers.RepeatedCompositeFieldContainer[NetStateRule]
    exclude: _containers.RepeatedCompositeFieldContainer[NetStateRule]
    top_data_type: str
    forward_precision: LayerParameter.PrecisionType
    transform_param: TransformationParameter
    loss_param: LossParameter
    accuracy_param: AccuracyParameter
    argmax_param: ArgMaxParameter
    concat_param: ConcatParameter
    contrastive_loss_param: ContrastiveLossParameter
    convolution_param: ConvolutionParameter
    data_param: DataParameter
    dropout_param: DropoutParameter
    dummy_data_param: DummyDataParameter
    eltwise_param: EltwiseParameter
    exp_param: ExpParameter
    hdf5_data_param: HDF5DataParameter
    hdf5_output_param: HDF5OutputParameter
    hinge_loss_param: HingeLossParameter
    image_data_param: ImageDataParameter
    infogain_loss_param: InfogainLossParameter
    inner_product_param: InnerProductParameter
    lrn_param: LRNParameter
    memory_data_param: MemoryDataParameter
    mvn_param: MVNParameter
    pooling_param: PoolingParameter
    pooling_specific_param: PoolingSpecificParameter
    power_param: PowerParameter
    relu_param: ReLUParameter
    sigmoid_param: SigmoidParameter
    softmax_param: SoftmaxParameter
    slice_param: SliceParameter
    tanh_param: TanHParameter
    threshold_param: ThresholdParameter
    window_data_param: WindowDataParameter
    python_param: PythonParameter
    ctc_param: CTCParameter
    prelu_param: PReLUParameter
    reshape_param: ReshapeParameter
    roi_pooling_param: ROIPoolingParameter
    affine_trans_param: AffineTransParameter
    roi_param: ROIParameter
    affine_trans_point_param: AffineTransPointParameter
    permute_param: PermuteParameter
    prior_box_param: PriorBoxParameter
    calc_affine_mat_param: CalcAffineMatParameter
    interp_param: InterpParameter
    detection_output_param: DetectionOutputParameter
    relu6_param: ReLU6Parameter
    bias_param: BiasParameter
    zxybn_param: ZXYBNParameter
    correlation_param: CorrelationParameter
    psroi_pooling_param: PSROIPoolingParameter
    nn_upsample_param: NNUpsampleParameter
    roi_align_param: ROIAlignParameter
    rpn_proposal_param: RpnProposalLayerParameter
    roi_mask_pooling_param: ROIMaskPoolingParameter
    channel_shuffle_param: ChannelShuffleParameter
    bn_param: BNParameter
    roi_align_pooling_pod_param: ROIAlignPoolingPodParameter
    roi_align_pooling_param: ROIAlignPoolingParameter
    psroi_align_pooling_param: PSROIAlignPoolingParameter
    alignment_to_roi_param: AlignmentToROIParameter
    unpooling_param: UnpoolingParameter
    log_param: LogParameter
    reverse_param: ReverseParameter
    lstm_param: LSTMParameter
    prior_vbox_param: PriorVBoxParameter
    btcostvolume_param: BTCostVolumeParameter
    moving_avg_param: MovingAvgParameter
    slicing_param: SlicingParameter
    bilateral_slicing_param: BilateralSlicingParameter
    bilateral_slice_apply_param: BilateralSliceApplyParameter
    adaptive_bilateral_slice_apply_param: AdaptiveBilateralSliceApplyParameter
    pointscurve_param: PointsCurveParameter
    subpixel_down_param: SubpixelDownParameter
    subpixel_up_param: SubpixelUpParameter
    clip_param: ClipParameter
    eltwise_affine_transform_param: EltwiseAffineTransformParameter
    tile_param: TileParameter
    pad_param: PadParameter
    reduce_param: ReduceParameter
    convolution3d_param: Convolution3dParameter
    deconvolution3d_param: Deconvolution3dParameter
    relu3d_param: ReLU3dParameter
    prelu3d_param: PReLUParameter
    leaky_relu3d_param: ReLU3dParameter
    elu3d_param: ReLU3dParameter
    groupnorm3d_param: GroupNorm3dParameter
    batchnorm3d_param: BatchNorm3dParameter
    maxpooling3d_param: Pooling3dParameter
    avepooling3d_param: Pooling3dParameter
    dropout3d_param: DropoutParameter
    interp3d_param: Interp3dParameter
    pooling3d_param: Pooling3dParameter
    reshape3d_param: ReshapeParameter
    softmax3d_param: SoftmaxParameter
    argmax3d_param: ArgMaxParameter
    inner_product3d_param: InnerProductParameter
    slice3d_param: SliceParameter
    concat3d_param: ConcatParameter
    bn3d_param: BNParameter
    matmul3d_param: MatMulParameter
    transpose3d_param: TransposeParameter
    scale_param: ScaleParameter
    crop_param: CropParameter
    recurrent_param: RecurrentParameter
    batch_norm_param: BatchNormParameter
    flatten_param: FlattenParameter
    scales_param: ScalesParameter
    transpose_param: TransposeParameter
    heatmap_param: HeatMap2CoordParameter
    normalize_param: NormalizeParameter
    correlation2d_param: Correlation2DParameter
    matmul_param: MatMulParameter
    parameter_param: ParameterParameter
    pixelshuffle_param: PixelShuffleParameter
    roi_transform_param: ROITransformParameter
    instance_norm_param: InstanceNormParameter
    grid_sample_param: GridSampleParameter
    reducel2_param: ReduceL2Parameter
    mean_param: MeanParameter
    variance_param: VarianceParameter
    grid_sample_3d_param: GridSample3DParameter
    gru_param: GRUParameter
    correlationmig_param: CorrelationMigParameter
    argsort_param: ArgSortParameter
    cumprod_param: CumProdParameter
    main_transform_param: MainTransformParameter
    quantize_param: _containers.RepeatedCompositeFieldContainer[QuantizeParameter]
    quantize_bits: int
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., bottom: _Optional[_Iterable[str]] = ..., top: _Optional[_Iterable[str]] = ..., phase: _Optional[_Union[Phase, str]] = ..., loss_weight: _Optional[_Iterable[float]] = ..., param: _Optional[_Iterable[_Union[ParamSpec, _Mapping]]] = ..., blobs: _Optional[_Iterable[_Union[BlobProto, _Mapping]]] = ..., include: _Optional[_Iterable[_Union[NetStateRule, _Mapping]]] = ..., exclude: _Optional[_Iterable[_Union[NetStateRule, _Mapping]]] = ..., top_data_type: _Optional[str] = ..., forward_precision: _Optional[_Union[LayerParameter.PrecisionType, str]] = ..., transform_param: _Optional[_Union[TransformationParameter, _Mapping]] = ..., loss_param: _Optional[_Union[LossParameter, _Mapping]] = ..., accuracy_param: _Optional[_Union[AccuracyParameter, _Mapping]] = ..., argmax_param: _Optional[_Union[ArgMaxParameter, _Mapping]] = ..., concat_param: _Optional[_Union[ConcatParameter, _Mapping]] = ..., contrastive_loss_param: _Optional[_Union[ContrastiveLossParameter, _Mapping]] = ..., convolution_param: _Optional[_Union[ConvolutionParameter, _Mapping]] = ..., data_param: _Optional[_Union[DataParameter, _Mapping]] = ..., dropout_param: _Optional[_Union[DropoutParameter, _Mapping]] = ..., dummy_data_param: _Optional[_Union[DummyDataParameter, _Mapping]] = ..., eltwise_param: _Optional[_Union[EltwiseParameter, _Mapping]] = ..., exp_param: _Optional[_Union[ExpParameter, _Mapping]] = ..., hdf5_data_param: _Optional[_Union[HDF5DataParameter, _Mapping]] = ..., hdf5_output_param: _Optional[_Union[HDF5OutputParameter, _Mapping]] = ..., hinge_loss_param: _Optional[_Union[HingeLossParameter, _Mapping]] = ..., image_data_param: _Optional[_Union[ImageDataParameter, _Mapping]] = ..., infogain_loss_param: _Optional[_Union[InfogainLossParameter, _Mapping]] = ..., inner_product_param: _Optional[_Union[InnerProductParameter, _Mapping]] = ..., lrn_param: _Optional[_Union[LRNParameter, _Mapping]] = ..., memory_data_param: _Optional[_Union[MemoryDataParameter, _Mapping]] = ..., mvn_param: _Optional[_Union[MVNParameter, _Mapping]] = ..., pooling_param: _Optional[_Union[PoolingParameter, _Mapping]] = ..., pooling_specific_param: _Optional[_Union[PoolingSpecificParameter, _Mapping]] = ..., power_param: _Optional[_Union[PowerParameter, _Mapping]] = ..., relu_param: _Optional[_Union[ReLUParameter, _Mapping]] = ..., sigmoid_param: _Optional[_Union[SigmoidParameter, _Mapping]] = ..., softmax_param: _Optional[_Union[SoftmaxParameter, _Mapping]] = ..., slice_param: _Optional[_Union[SliceParameter, _Mapping]] = ..., tanh_param: _Optional[_Union[TanHParameter, _Mapping]] = ..., threshold_param: _Optional[_Union[ThresholdParameter, _Mapping]] = ..., window_data_param: _Optional[_Union[WindowDataParameter, _Mapping]] = ..., python_param: _Optional[_Union[PythonParameter, _Mapping]] = ..., ctc_param: _Optional[_Union[CTCParameter, _Mapping]] = ..., prelu_param: _Optional[_Union[PReLUParameter, _Mapping]] = ..., reshape_param: _Optional[_Union[ReshapeParameter, _Mapping]] = ..., roi_pooling_param: _Optional[_Union[ROIPoolingParameter, _Mapping]] = ..., affine_trans_param: _Optional[_Union[AffineTransParameter, _Mapping]] = ..., roi_param: _Optional[_Union[ROIParameter, _Mapping]] = ..., affine_trans_point_param: _Optional[_Union[AffineTransPointParameter, _Mapping]] = ..., permute_param: _Optional[_Union[PermuteParameter, _Mapping]] = ..., prior_box_param: _Optional[_Union[PriorBoxParameter, _Mapping]] = ..., calc_affine_mat_param: _Optional[_Union[CalcAffineMatParameter, _Mapping]] = ..., interp_param: _Optional[_Union[InterpParameter, _Mapping]] = ..., detection_output_param: _Optional[_Union[DetectionOutputParameter, _Mapping]] = ..., relu6_param: _Optional[_Union[ReLU6Parameter, _Mapping]] = ..., bias_param: _Optional[_Union[BiasParameter, _Mapping]] = ..., zxybn_param: _Optional[_Union[ZXYBNParameter, _Mapping]] = ..., correlation_param: _Optional[_Union[CorrelationParameter, _Mapping]] = ..., psroi_pooling_param: _Optional[_Union[PSROIPoolingParameter, _Mapping]] = ..., nn_upsample_param: _Optional[_Union[NNUpsampleParameter, _Mapping]] = ..., roi_align_param: _Optional[_Union[ROIAlignParameter, _Mapping]] = ..., rpn_proposal_param: _Optional[_Union[RpnProposalLayerParameter, _Mapping]] = ..., roi_mask_pooling_param: _Optional[_Union[ROIMaskPoolingParameter, _Mapping]] = ..., channel_shuffle_param: _Optional[_Union[ChannelShuffleParameter, _Mapping]] = ..., bn_param: _Optional[_Union[BNParameter, _Mapping]] = ..., roi_align_pooling_pod_param: _Optional[_Union[ROIAlignPoolingPodParameter, _Mapping]] = ..., roi_align_pooling_param: _Optional[_Union[ROIAlignPoolingParameter, _Mapping]] = ..., psroi_align_pooling_param: _Optional[_Union[PSROIAlignPoolingParameter, _Mapping]] = ..., alignment_to_roi_param: _Optional[_Union[AlignmentToROIParameter, _Mapping]] = ..., unpooling_param: _Optional[_Union[UnpoolingParameter, _Mapping]] = ..., log_param: _Optional[_Union[LogParameter, _Mapping]] = ..., reverse_param: _Optional[_Union[ReverseParameter, _Mapping]] = ..., lstm_param: _Optional[_Union[LSTMParameter, _Mapping]] = ..., prior_vbox_param: _Optional[_Union[PriorVBoxParameter, _Mapping]] = ..., btcostvolume_param: _Optional[_Union[BTCostVolumeParameter, _Mapping]] = ..., moving_avg_param: _Optional[_Union[MovingAvgParameter, _Mapping]] = ..., slicing_param: _Optional[_Union[SlicingParameter, _Mapping]] = ..., bilateral_slicing_param: _Optional[_Union[BilateralSlicingParameter, _Mapping]] = ..., bilateral_slice_apply_param: _Optional[_Union[BilateralSliceApplyParameter, _Mapping]] = ..., adaptive_bilateral_slice_apply_param: _Optional[_Union[AdaptiveBilateralSliceApplyParameter, _Mapping]] = ..., pointscurve_param: _Optional[_Union[PointsCurveParameter, _Mapping]] = ..., subpixel_down_param: _Optional[_Union[SubpixelDownParameter, _Mapping]] = ..., subpixel_up_param: _Optional[_Union[SubpixelUpParameter, _Mapping]] = ..., clip_param: _Optional[_Union[ClipParameter, _Mapping]] = ..., eltwise_affine_transform_param: _Optional[_Union[EltwiseAffineTransformParameter, _Mapping]] = ..., tile_param: _Optional[_Union[TileParameter, _Mapping]] = ..., pad_param: _Optional[_Union[PadParameter, _Mapping]] = ..., reduce_param: _Optional[_Union[ReduceParameter, _Mapping]] = ..., convolution3d_param: _Optional[_Union[Convolution3dParameter, _Mapping]] = ..., deconvolution3d_param: _Optional[_Union[Deconvolution3dParameter, _Mapping]] = ..., relu3d_param: _Optional[_Union[ReLU3dParameter, _Mapping]] = ..., prelu3d_param: _Optional[_Union[PReLUParameter, _Mapping]] = ..., leaky_relu3d_param: _Optional[_Union[ReLU3dParameter, _Mapping]] = ..., elu3d_param: _Optional[_Union[ReLU3dParameter, _Mapping]] = ..., groupnorm3d_param: _Optional[_Union[GroupNorm3dParameter, _Mapping]] = ..., batchnorm3d_param: _Optional[_Union[BatchNorm3dParameter, _Mapping]] = ..., maxpooling3d_param: _Optional[_Union[Pooling3dParameter, _Mapping]] = ..., avepooling3d_param: _Optional[_Union[Pooling3dParameter, _Mapping]] = ..., dropout3d_param: _Optional[_Union[DropoutParameter, _Mapping]] = ..., interp3d_param: _Optional[_Union[Interp3dParameter, _Mapping]] = ..., pooling3d_param: _Optional[_Union[Pooling3dParameter, _Mapping]] = ..., reshape3d_param: _Optional[_Union[ReshapeParameter, _Mapping]] = ..., softmax3d_param: _Optional[_Union[SoftmaxParameter, _Mapping]] = ..., argmax3d_param: _Optional[_Union[ArgMaxParameter, _Mapping]] = ..., inner_product3d_param: _Optional[_Union[InnerProductParameter, _Mapping]] = ..., slice3d_param: _Optional[_Union[SliceParameter, _Mapping]] = ..., concat3d_param: _Optional[_Union[ConcatParameter, _Mapping]] = ..., bn3d_param: _Optional[_Union[BNParameter, _Mapping]] = ..., matmul3d_param: _Optional[_Union[MatMulParameter, _Mapping]] = ..., transpose3d_param: _Optional[_Union[TransposeParameter, _Mapping]] = ..., scale_param: _Optional[_Union[ScaleParameter, _Mapping]] = ..., crop_param: _Optional[_Union[CropParameter, _Mapping]] = ..., recurrent_param: _Optional[_Union[RecurrentParameter, _Mapping]] = ..., batch_norm_param: _Optional[_Union[BatchNormParameter, _Mapping]] = ..., flatten_param: _Optional[_Union[FlattenParameter, _Mapping]] = ..., scales_param: _Optional[_Union[ScalesParameter, _Mapping]] = ..., transpose_param: _Optional[_Union[TransposeParameter, _Mapping]] = ..., heatmap_param: _Optional[_Union[HeatMap2CoordParameter, _Mapping]] = ..., normalize_param: _Optional[_Union[NormalizeParameter, _Mapping]] = ..., correlation2d_param: _Optional[_Union[Correlation2DParameter, _Mapping]] = ..., matmul_param: _Optional[_Union[MatMulParameter, _Mapping]] = ..., parameter_param: _Optional[_Union[ParameterParameter, _Mapping]] = ..., pixelshuffle_param: _Optional[_Union[PixelShuffleParameter, _Mapping]] = ..., roi_transform_param: _Optional[_Union[ROITransformParameter, _Mapping]] = ..., instance_norm_param: _Optional[_Union[InstanceNormParameter, _Mapping]] = ..., grid_sample_param: _Optional[_Union[GridSampleParameter, _Mapping]] = ..., reducel2_param: _Optional[_Union[ReduceL2Parameter, _Mapping]] = ..., mean_param: _Optional[_Union[MeanParameter, _Mapping]] = ..., variance_param: _Optional[_Union[VarianceParameter, _Mapping]] = ..., grid_sample_3d_param: _Optional[_Union[GridSample3DParameter, _Mapping]] = ..., gru_param: _Optional[_Union[GRUParameter, _Mapping]] = ..., correlationmig_param: _Optional[_Union[CorrelationMigParameter, _Mapping]] = ..., argsort_param: _Optional[_Union[ArgSortParameter, _Mapping]] = ..., cumprod_param: _Optional[_Union[CumProdParameter, _Mapping]] = ..., main_transform_param: _Optional[_Union[MainTransformParameter, _Mapping]] = ..., quantize_param: _Optional[_Iterable[_Union[QuantizeParameter, _Mapping]]] = ..., quantize_bits: _Optional[int] = ...) -> None: ...

class Convolution3dParameter(_message.Message):
    __slots__ = ["num_output", "bias_term", "pad", "pad_d", "pad_h", "pad_w", "kernel_size", "kernel_d", "kernel_h", "kernel_w", "group", "stride", "stride_d", "stride_h", "stride_w", "hole", "hole_d", "hole_h", "hole_w"]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BIAS_TERM_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_D_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_D_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_D_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    HOLE_FIELD_NUMBER: _ClassVar[int]
    HOLE_D_FIELD_NUMBER: _ClassVar[int]
    HOLE_H_FIELD_NUMBER: _ClassVar[int]
    HOLE_W_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    bias_term: bool
    pad: int
    pad_d: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_d: int
    kernel_h: int
    kernel_w: int
    group: int
    stride: int
    stride_d: int
    stride_h: int
    stride_w: int
    hole: int
    hole_d: int
    hole_h: int
    hole_w: int
    def __init__(self, num_output: _Optional[int] = ..., bias_term: bool = ..., pad: _Optional[int] = ..., pad_d: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_d: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., group: _Optional[int] = ..., stride: _Optional[int] = ..., stride_d: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., hole: _Optional[int] = ..., hole_d: _Optional[int] = ..., hole_h: _Optional[int] = ..., hole_w: _Optional[int] = ...) -> None: ...

class ConvolutionTranspose3dParameter(_message.Message):
    __slots__ = ["num_output", "bias_term", "pad", "pad_h", "pad_w", "kernel_size", "kernel_h", "kernel_w", "group", "stride", "stride_h", "stride_w", "hole", "hole_h"]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BIAS_TERM_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    HOLE_FIELD_NUMBER: _ClassVar[int]
    HOLE_H_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    bias_term: bool
    pad: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_h: int
    kernel_w: int
    group: int
    stride: int
    stride_h: int
    stride_w: int
    hole: int
    hole_h: int
    def __init__(self, num_output: _Optional[int] = ..., bias_term: bool = ..., pad: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., group: _Optional[int] = ..., stride: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., hole: _Optional[int] = ..., hole_h: _Optional[int] = ...) -> None: ...

class Deconvolution3dParameter(_message.Message):
    __slots__ = ["num_output", "bias_term", "pad", "pad_d", "pad_h", "pad_w", "kernel_size", "kernel_d", "kernel_h", "kernel_w", "group", "stride", "stride_d", "stride_h", "stride_w", "hole", "hole_d", "hole_h", "hole_w", "out_pad", "out_pad_d", "out_pad_h", "out_pad_w"]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BIAS_TERM_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_D_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_D_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_D_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    HOLE_FIELD_NUMBER: _ClassVar[int]
    HOLE_D_FIELD_NUMBER: _ClassVar[int]
    HOLE_H_FIELD_NUMBER: _ClassVar[int]
    HOLE_W_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_D_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_H_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_W_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    bias_term: bool
    pad: int
    pad_d: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_d: int
    kernel_h: int
    kernel_w: int
    group: int
    stride: int
    stride_d: int
    stride_h: int
    stride_w: int
    hole: int
    hole_d: int
    hole_h: int
    hole_w: int
    out_pad: int
    out_pad_d: int
    out_pad_h: int
    out_pad_w: int
    def __init__(self, num_output: _Optional[int] = ..., bias_term: bool = ..., pad: _Optional[int] = ..., pad_d: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_d: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., group: _Optional[int] = ..., stride: _Optional[int] = ..., stride_d: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., hole: _Optional[int] = ..., hole_d: _Optional[int] = ..., hole_h: _Optional[int] = ..., hole_w: _Optional[int] = ..., out_pad: _Optional[int] = ..., out_pad_d: _Optional[int] = ..., out_pad_h: _Optional[int] = ..., out_pad_w: _Optional[int] = ...) -> None: ...

class ReLU3dParameter(_message.Message):
    __slots__ = ["negative_slope", "channel_shared"]
    NEGATIVE_SLOPE_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_SHARED_FIELD_NUMBER: _ClassVar[int]
    negative_slope: float
    channel_shared: bool
    def __init__(self, negative_slope: _Optional[float] = ..., channel_shared: bool = ...) -> None: ...

class GroupNorm3dParameter(_message.Message):
    __slots__ = ["group", "moving_average_fraction", "use_global_stats", "eps"]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    MOVING_AVERAGE_FRACTION_FIELD_NUMBER: _ClassVar[int]
    USE_GLOBAL_STATS_FIELD_NUMBER: _ClassVar[int]
    EPS_FIELD_NUMBER: _ClassVar[int]
    group: int
    moving_average_fraction: float
    use_global_stats: bool
    eps: float
    def __init__(self, group: _Optional[int] = ..., moving_average_fraction: _Optional[float] = ..., use_global_stats: bool = ..., eps: _Optional[float] = ...) -> None: ...

class BatchNorm3dParameter(_message.Message):
    __slots__ = ["moving_average_fraction", "use_global_stats", "eps"]
    MOVING_AVERAGE_FRACTION_FIELD_NUMBER: _ClassVar[int]
    USE_GLOBAL_STATS_FIELD_NUMBER: _ClassVar[int]
    EPS_FIELD_NUMBER: _ClassVar[int]
    moving_average_fraction: float
    use_global_stats: bool
    eps: float
    def __init__(self, moving_average_fraction: _Optional[float] = ..., use_global_stats: bool = ..., eps: _Optional[float] = ...) -> None: ...

class Pooling3dParameter(_message.Message):
    __slots__ = ["pool", "pad", "pad_d", "pad_h", "pad_w", "kernel_size", "kernel_d", "kernel_h", "kernel_w", "stride", "stride_d", "stride_h", "stride_w", "global_pooling", "ceil_mode"]
    class PoolMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MAX: _ClassVar[Pooling3dParameter.PoolMethod]
        AVE: _ClassVar[Pooling3dParameter.PoolMethod]
    MAX: Pooling3dParameter.PoolMethod
    AVE: Pooling3dParameter.PoolMethod
    POOL_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_D_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_D_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_D_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_POOLING_FIELD_NUMBER: _ClassVar[int]
    CEIL_MODE_FIELD_NUMBER: _ClassVar[int]
    pool: Pooling3dParameter.PoolMethod
    pad: int
    pad_d: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_d: int
    kernel_h: int
    kernel_w: int
    stride: int
    stride_d: int
    stride_h: int
    stride_w: int
    global_pooling: bool
    ceil_mode: bool
    def __init__(self, pool: _Optional[_Union[Pooling3dParameter.PoolMethod, str]] = ..., pad: _Optional[int] = ..., pad_d: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_d: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., stride: _Optional[int] = ..., stride_d: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., global_pooling: bool = ..., ceil_mode: bool = ...) -> None: ...

class NormalizeParameter(_message.Message):
    __slots__ = ["coeff", "mode"]
    class NormalizeMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        L1: _ClassVar[NormalizeParameter.NormalizeMethod]
        L2: _ClassVar[NormalizeParameter.NormalizeMethod]
    L1: NormalizeParameter.NormalizeMethod
    L2: NormalizeParameter.NormalizeMethod
    COEFF_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    coeff: float
    mode: NormalizeParameter.NormalizeMethod
    def __init__(self, coeff: _Optional[float] = ..., mode: _Optional[_Union[NormalizeParameter.NormalizeMethod, str]] = ...) -> None: ...

class HeatMap2CoordParameter(_message.Message):
    __slots__ = ["coord_h", "coord_w", "coord_reposition"]
    COORD_H_FIELD_NUMBER: _ClassVar[int]
    COORD_W_FIELD_NUMBER: _ClassVar[int]
    COORD_REPOSITION_FIELD_NUMBER: _ClassVar[int]
    coord_h: int
    coord_w: int
    coord_reposition: bool
    def __init__(self, coord_h: _Optional[int] = ..., coord_w: _Optional[int] = ..., coord_reposition: bool = ...) -> None: ...

class BiasParameter(_message.Message):
    __slots__ = ["axis", "num_axes", "filler"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    NUM_AXES_FIELD_NUMBER: _ClassVar[int]
    FILLER_FIELD_NUMBER: _ClassVar[int]
    axis: int
    num_axes: int
    filler: FillerParameter
    def __init__(self, axis: _Optional[int] = ..., num_axes: _Optional[int] = ..., filler: _Optional[_Union[FillerParameter, _Mapping]] = ...) -> None: ...

class AlignmentToROIParameter(_message.Message):
    __slots__ = ["mouth_scale", "eye_scale", "max_width", "max_height"]
    MOUTH_SCALE_FIELD_NUMBER: _ClassVar[int]
    EYE_SCALE_FIELD_NUMBER: _ClassVar[int]
    MAX_WIDTH_FIELD_NUMBER: _ClassVar[int]
    MAX_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    mouth_scale: float
    eye_scale: float
    max_width: int
    max_height: int
    def __init__(self, mouth_scale: _Optional[float] = ..., eye_scale: _Optional[float] = ..., max_width: _Optional[int] = ..., max_height: _Optional[int] = ...) -> None: ...

class ROIAlignPoolingParameter(_message.Message):
    __slots__ = ["pooled_h", "pooled_w", "spatial_scale", "sample_num"]
    POOLED_H_FIELD_NUMBER: _ClassVar[int]
    POOLED_W_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_NUM_FIELD_NUMBER: _ClassVar[int]
    pooled_h: int
    pooled_w: int
    spatial_scale: float
    sample_num: int
    def __init__(self, pooled_h: _Optional[int] = ..., pooled_w: _Optional[int] = ..., spatial_scale: _Optional[float] = ..., sample_num: _Optional[int] = ...) -> None: ...

class ROIAlignPoolingPodParameter(_message.Message):
    __slots__ = ["pooled_h", "pooled_w", "spatial_scale", "sample_num"]
    POOLED_H_FIELD_NUMBER: _ClassVar[int]
    POOLED_W_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_NUM_FIELD_NUMBER: _ClassVar[int]
    pooled_h: int
    pooled_w: int
    spatial_scale: float
    sample_num: int
    def __init__(self, pooled_h: _Optional[int] = ..., pooled_w: _Optional[int] = ..., spatial_scale: _Optional[float] = ..., sample_num: _Optional[int] = ...) -> None: ...

class PSROIAlignPoolingParameter(_message.Message):
    __slots__ = ["spatial_scale", "output_dim", "group_size", "sample_num"]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIM_FIELD_NUMBER: _ClassVar[int]
    GROUP_SIZE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_NUM_FIELD_NUMBER: _ClassVar[int]
    spatial_scale: float
    output_dim: int
    group_size: int
    sample_num: int
    def __init__(self, spatial_scale: _Optional[float] = ..., output_dim: _Optional[int] = ..., group_size: _Optional[int] = ..., sample_num: _Optional[int] = ...) -> None: ...

class UnpoolingParameter(_message.Message):
    __slots__ = ["unpool", "pad", "pad_h", "pad_w", "kernel_size", "kernel_h", "kernel_w", "stride", "stride_h", "stride_w", "unpool_size", "unpool_h", "unpool_w", "engine"]
    class UnpoolMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MAX: _ClassVar[UnpoolingParameter.UnpoolMethod]
        AVE: _ClassVar[UnpoolingParameter.UnpoolMethod]
        TILE: _ClassVar[UnpoolingParameter.UnpoolMethod]
    MAX: UnpoolingParameter.UnpoolMethod
    AVE: UnpoolingParameter.UnpoolMethod
    TILE: UnpoolingParameter.UnpoolMethod
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[UnpoolingParameter.Engine]
        CAFFE: _ClassVar[UnpoolingParameter.Engine]
    DEFAULT: UnpoolingParameter.Engine
    CAFFE: UnpoolingParameter.Engine
    UNPOOL_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    UNPOOL_SIZE_FIELD_NUMBER: _ClassVar[int]
    UNPOOL_H_FIELD_NUMBER: _ClassVar[int]
    UNPOOL_W_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    unpool: UnpoolingParameter.UnpoolMethod
    pad: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_h: int
    kernel_w: int
    stride: int
    stride_h: int
    stride_w: int
    unpool_size: int
    unpool_h: int
    unpool_w: int
    engine: UnpoolingParameter.Engine
    def __init__(self, unpool: _Optional[_Union[UnpoolingParameter.UnpoolMethod, str]] = ..., pad: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., stride: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., unpool_size: _Optional[int] = ..., unpool_h: _Optional[int] = ..., unpool_w: _Optional[int] = ..., engine: _Optional[_Union[UnpoolingParameter.Engine, str]] = ...) -> None: ...

class ReverseParameter(_message.Message):
    __slots__ = ["axis"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    axis: int
    def __init__(self, axis: _Optional[int] = ...) -> None: ...

class LSTMParameter(_message.Message):
    __slots__ = ["num_output", "clipping_threshold", "weight_filler", "bias_filler", "batch_size"]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    CLIPPING_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    clipping_threshold: float
    weight_filler: FillerParameter
    bias_filler: FillerParameter
    batch_size: int
    def __init__(self, num_output: _Optional[int] = ..., clipping_threshold: _Optional[float] = ..., weight_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., batch_size: _Optional[int] = ...) -> None: ...

class PriorVBoxParameter(_message.Message):
    __slots__ = ["height", "width", "clip", "variance", "img_size", "img_h", "img_w", "step", "step_h", "step_w", "offset"]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    CLIP_FIELD_NUMBER: _ClassVar[int]
    VARIANCE_FIELD_NUMBER: _ClassVar[int]
    IMG_SIZE_FIELD_NUMBER: _ClassVar[int]
    IMG_H_FIELD_NUMBER: _ClassVar[int]
    IMG_W_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    STEP_H_FIELD_NUMBER: _ClassVar[int]
    STEP_W_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    height: _containers.RepeatedScalarFieldContainer[float]
    width: float
    clip: bool
    variance: _containers.RepeatedScalarFieldContainer[float]
    img_size: int
    img_h: int
    img_w: int
    step: float
    step_h: float
    step_w: float
    offset: float
    def __init__(self, height: _Optional[_Iterable[float]] = ..., width: _Optional[float] = ..., clip: bool = ..., variance: _Optional[_Iterable[float]] = ..., img_size: _Optional[int] = ..., img_h: _Optional[int] = ..., img_w: _Optional[int] = ..., step: _Optional[float] = ..., step_h: _Optional[float] = ..., step_w: _Optional[float] = ..., offset: _Optional[float] = ...) -> None: ...

class NNUpsampleParameter(_message.Message):
    __slots__ = ["resize"]
    RESIZE_FIELD_NUMBER: _ClassVar[int]
    resize: int
    def __init__(self, resize: _Optional[int] = ...) -> None: ...

class BTCostVolumeParameter(_message.Message):
    __slots__ = ["min_disparity", "disparity_num", "save_disparity", "step_w", "step_h", "step_mode", "cost_domain_type", "implement_mode"]
    class CostDomainType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        SOBEL_ONLY: _ClassVar[BTCostVolumeParameter.CostDomainType]
        IMAGE_ONLY: _ClassVar[BTCostVolumeParameter.CostDomainType]
        SOBEL_AND_IMAGE: _ClassVar[BTCostVolumeParameter.CostDomainType]
    SOBEL_ONLY: BTCostVolumeParameter.CostDomainType
    IMAGE_ONLY: BTCostVolumeParameter.CostDomainType
    SOBEL_AND_IMAGE: BTCostVolumeParameter.CostDomainType
    class ImplementMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        GPU_MODE: _ClassVar[BTCostVolumeParameter.ImplementMode]
        CPU_MODE: _ClassVar[BTCostVolumeParameter.ImplementMode]
    GPU_MODE: BTCostVolumeParameter.ImplementMode
    CPU_MODE: BTCostVolumeParameter.ImplementMode
    MIN_DISPARITY_FIELD_NUMBER: _ClassVar[int]
    DISPARITY_NUM_FIELD_NUMBER: _ClassVar[int]
    SAVE_DISPARITY_FIELD_NUMBER: _ClassVar[int]
    STEP_W_FIELD_NUMBER: _ClassVar[int]
    STEP_H_FIELD_NUMBER: _ClassVar[int]
    STEP_MODE_FIELD_NUMBER: _ClassVar[int]
    COST_DOMAIN_TYPE_FIELD_NUMBER: _ClassVar[int]
    IMPLEMENT_MODE_FIELD_NUMBER: _ClassVar[int]
    min_disparity: int
    disparity_num: int
    save_disparity: bool
    step_w: int
    step_h: int
    step_mode: int
    cost_domain_type: BTCostVolumeParameter.CostDomainType
    implement_mode: BTCostVolumeParameter.ImplementMode
    def __init__(self, min_disparity: _Optional[int] = ..., disparity_num: _Optional[int] = ..., save_disparity: bool = ..., step_w: _Optional[int] = ..., step_h: _Optional[int] = ..., step_mode: _Optional[int] = ..., cost_domain_type: _Optional[_Union[BTCostVolumeParameter.CostDomainType, str]] = ..., implement_mode: _Optional[_Union[BTCostVolumeParameter.ImplementMode, str]] = ...) -> None: ...

class CorrelationParameter(_message.Message):
    __slots__ = ["pad", "kernel_size", "max_displacement", "stride_1", "stride_2", "single_direction", "do_abs", "correlation_type", "pad_shift", "mvn"]
    class CorrelationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MULTIPLY: _ClassVar[CorrelationParameter.CorrelationType]
        SUBTRACT: _ClassVar[CorrelationParameter.CorrelationType]
    MULTIPLY: CorrelationParameter.CorrelationType
    SUBTRACT: CorrelationParameter.CorrelationType
    PAD_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_DISPLACEMENT_FIELD_NUMBER: _ClassVar[int]
    STRIDE_1_FIELD_NUMBER: _ClassVar[int]
    STRIDE_2_FIELD_NUMBER: _ClassVar[int]
    SINGLE_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    DO_ABS_FIELD_NUMBER: _ClassVar[int]
    CORRELATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAD_SHIFT_FIELD_NUMBER: _ClassVar[int]
    MVN_FIELD_NUMBER: _ClassVar[int]
    pad: int
    kernel_size: int
    max_displacement: int
    stride_1: int
    stride_2: int
    single_direction: int
    do_abs: bool
    correlation_type: CorrelationParameter.CorrelationType
    pad_shift: int
    mvn: bool
    def __init__(self, pad: _Optional[int] = ..., kernel_size: _Optional[int] = ..., max_displacement: _Optional[int] = ..., stride_1: _Optional[int] = ..., stride_2: _Optional[int] = ..., single_direction: _Optional[int] = ..., do_abs: bool = ..., correlation_type: _Optional[_Union[CorrelationParameter.CorrelationType, str]] = ..., pad_shift: _Optional[int] = ..., mvn: bool = ...) -> None: ...

class PSROIPoolingParameter(_message.Message):
    __slots__ = ["spatial_scale", "output_dim", "group_size", "roi_scale"]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DIM_FIELD_NUMBER: _ClassVar[int]
    GROUP_SIZE_FIELD_NUMBER: _ClassVar[int]
    ROI_SCALE_FIELD_NUMBER: _ClassVar[int]
    spatial_scale: float
    output_dim: int
    group_size: int
    roi_scale: float
    def __init__(self, spatial_scale: _Optional[float] = ..., output_dim: _Optional[int] = ..., group_size: _Optional[int] = ..., roi_scale: _Optional[float] = ...) -> None: ...

class ScaleParameter(_message.Message):
    __slots__ = ["axis", "num_axes", "filler", "bias_term", "bias_filler"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    NUM_AXES_FIELD_NUMBER: _ClassVar[int]
    FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_TERM_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    axis: int
    num_axes: int
    filler: FillerParameter
    bias_term: bool
    bias_filler: FillerParameter
    def __init__(self, axis: _Optional[int] = ..., num_axes: _Optional[int] = ..., filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_term: bool = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ...) -> None: ...

class ScalesParameter(_message.Message):
    __slots__ = ["alpha", "beta"]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    BETA_FIELD_NUMBER: _ClassVar[int]
    alpha: float
    beta: float
    def __init__(self, alpha: _Optional[float] = ..., beta: _Optional[float] = ...) -> None: ...

class TransposeParameter(_message.Message):
    __slots__ = ["dim"]
    DIM_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, dim: _Optional[_Iterable[int]] = ...) -> None: ...

class ROIAlignParameter(_message.Message):
    __slots__ = ["pooled_h", "pooled_w", "spatial_scale"]
    POOLED_H_FIELD_NUMBER: _ClassVar[int]
    POOLED_W_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    pooled_h: int
    pooled_w: int
    spatial_scale: float
    def __init__(self, pooled_h: _Optional[int] = ..., pooled_w: _Optional[int] = ..., spatial_scale: _Optional[float] = ...) -> None: ...

class RpnProposalLayerParameter(_message.Message):
    __slots__ = ["feat_stride", "allowed_border", "test_desc_param", "train_desc_param", "anchor_scales", "anchor_ratios", "score", "withOutSoftmax", "bbox_reg_xy_limit", "bbox_reg_hw_temp", "bbox_reg_hw_ratio_thresh_hi", "bbox_reg_hw_ratio_thresh_lo"]
    class TestDescParam(_message.Message):
        __slots__ = ["rpn_pre_nms_top_n", "rpn_post_nms_top_n", "rpn_min_size", "rpn_nms_thresh"]
        RPN_PRE_NMS_TOP_N_FIELD_NUMBER: _ClassVar[int]
        RPN_POST_NMS_TOP_N_FIELD_NUMBER: _ClassVar[int]
        RPN_MIN_SIZE_FIELD_NUMBER: _ClassVar[int]
        RPN_NMS_THRESH_FIELD_NUMBER: _ClassVar[int]
        rpn_pre_nms_top_n: int
        rpn_post_nms_top_n: int
        rpn_min_size: int
        rpn_nms_thresh: float
        def __init__(self, rpn_pre_nms_top_n: _Optional[int] = ..., rpn_post_nms_top_n: _Optional[int] = ..., rpn_min_size: _Optional[int] = ..., rpn_nms_thresh: _Optional[float] = ...) -> None: ...
    class TrainDescParam(_message.Message):
        __slots__ = ["rpn_pre_nms_top_n", "rpn_post_nms_top_n", "rpn_min_size", "rpn_nms_thresh"]
        RPN_PRE_NMS_TOP_N_FIELD_NUMBER: _ClassVar[int]
        RPN_POST_NMS_TOP_N_FIELD_NUMBER: _ClassVar[int]
        RPN_MIN_SIZE_FIELD_NUMBER: _ClassVar[int]
        RPN_NMS_THRESH_FIELD_NUMBER: _ClassVar[int]
        rpn_pre_nms_top_n: int
        rpn_post_nms_top_n: int
        rpn_min_size: int
        rpn_nms_thresh: float
        def __init__(self, rpn_pre_nms_top_n: _Optional[int] = ..., rpn_post_nms_top_n: _Optional[int] = ..., rpn_min_size: _Optional[int] = ..., rpn_nms_thresh: _Optional[float] = ...) -> None: ...
    FEAT_STRIDE_FIELD_NUMBER: _ClassVar[int]
    ALLOWED_BORDER_FIELD_NUMBER: _ClassVar[int]
    TEST_DESC_PARAM_FIELD_NUMBER: _ClassVar[int]
    TRAIN_DESC_PARAM_FIELD_NUMBER: _ClassVar[int]
    ANCHOR_SCALES_FIELD_NUMBER: _ClassVar[int]
    ANCHOR_RATIOS_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    WITHOUTSOFTMAX_FIELD_NUMBER: _ClassVar[int]
    BBOX_REG_XY_LIMIT_FIELD_NUMBER: _ClassVar[int]
    BBOX_REG_HW_TEMP_FIELD_NUMBER: _ClassVar[int]
    BBOX_REG_HW_RATIO_THRESH_HI_FIELD_NUMBER: _ClassVar[int]
    BBOX_REG_HW_RATIO_THRESH_LO_FIELD_NUMBER: _ClassVar[int]
    feat_stride: int
    allowed_border: int
    test_desc_param: RpnProposalLayerParameter.TestDescParam
    train_desc_param: RpnProposalLayerParameter.TrainDescParam
    anchor_scales: _containers.RepeatedScalarFieldContainer[int]
    anchor_ratios: _containers.RepeatedScalarFieldContainer[float]
    score: float
    withOutSoftmax: bool
    bbox_reg_xy_limit: float
    bbox_reg_hw_temp: float
    bbox_reg_hw_ratio_thresh_hi: float
    bbox_reg_hw_ratio_thresh_lo: float
    def __init__(self, feat_stride: _Optional[int] = ..., allowed_border: _Optional[int] = ..., test_desc_param: _Optional[_Union[RpnProposalLayerParameter.TestDescParam, _Mapping]] = ..., train_desc_param: _Optional[_Union[RpnProposalLayerParameter.TrainDescParam, _Mapping]] = ..., anchor_scales: _Optional[_Iterable[int]] = ..., anchor_ratios: _Optional[_Iterable[float]] = ..., score: _Optional[float] = ..., withOutSoftmax: bool = ..., bbox_reg_xy_limit: _Optional[float] = ..., bbox_reg_hw_temp: _Optional[float] = ..., bbox_reg_hw_ratio_thresh_hi: _Optional[float] = ..., bbox_reg_hw_ratio_thresh_lo: _Optional[float] = ...) -> None: ...

class ROIMaskPoolingParameter(_message.Message):
    __slots__ = ["pooled_h", "pooled_w", "spatial_scale", "half_part", "roi_scale", "mask_scale"]
    POOLED_H_FIELD_NUMBER: _ClassVar[int]
    POOLED_W_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    HALF_PART_FIELD_NUMBER: _ClassVar[int]
    ROI_SCALE_FIELD_NUMBER: _ClassVar[int]
    MASK_SCALE_FIELD_NUMBER: _ClassVar[int]
    pooled_h: int
    pooled_w: int
    spatial_scale: float
    half_part: int
    roi_scale: float
    mask_scale: float
    def __init__(self, pooled_h: _Optional[int] = ..., pooled_w: _Optional[int] = ..., spatial_scale: _Optional[float] = ..., half_part: _Optional[int] = ..., roi_scale: _Optional[float] = ..., mask_scale: _Optional[float] = ...) -> None: ...

class ZXYBNParameter(_message.Message):
    __slots__ = ["slope_filler", "bias_filler", "momentum", "eps", "frozen", "engine"]
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[ZXYBNParameter.Engine]
        CAFFE: _ClassVar[ZXYBNParameter.Engine]
        CUDNN: _ClassVar[ZXYBNParameter.Engine]
    DEFAULT: ZXYBNParameter.Engine
    CAFFE: ZXYBNParameter.Engine
    CUDNN: ZXYBNParameter.Engine
    SLOPE_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    EPS_FIELD_NUMBER: _ClassVar[int]
    FROZEN_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    slope_filler: FillerParameter
    bias_filler: FillerParameter
    momentum: float
    eps: float
    frozen: bool
    engine: ZXYBNParameter.Engine
    def __init__(self, slope_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., momentum: _Optional[float] = ..., eps: _Optional[float] = ..., frozen: bool = ..., engine: _Optional[_Union[ZXYBNParameter.Engine, str]] = ...) -> None: ...

class TransformationParameter(_message.Message):
    __slots__ = ["scale", "mirror", "crop_size", "mean_file", "mean_value"]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    MIRROR_FIELD_NUMBER: _ClassVar[int]
    CROP_SIZE_FIELD_NUMBER: _ClassVar[int]
    MEAN_FILE_FIELD_NUMBER: _ClassVar[int]
    MEAN_VALUE_FIELD_NUMBER: _ClassVar[int]
    scale: float
    mirror: bool
    crop_size: int
    mean_file: str
    mean_value: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, scale: _Optional[float] = ..., mirror: bool = ..., crop_size: _Optional[int] = ..., mean_file: _Optional[str] = ..., mean_value: _Optional[_Iterable[float]] = ...) -> None: ...

class LossParameter(_message.Message):
    __slots__ = ["ignore_label", "normalize"]
    IGNORE_LABEL_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    ignore_label: int
    normalize: bool
    def __init__(self, ignore_label: _Optional[int] = ..., normalize: bool = ...) -> None: ...

class AccuracyParameter(_message.Message):
    __slots__ = ["top_k"]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    top_k: int
    def __init__(self, top_k: _Optional[int] = ...) -> None: ...

class ArgMaxParameter(_message.Message):
    __slots__ = ["out_max_val", "top_k", "axis"]
    OUT_MAX_VAL_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    out_max_val: bool
    top_k: int
    axis: int
    def __init__(self, out_max_val: bool = ..., top_k: _Optional[int] = ..., axis: _Optional[int] = ...) -> None: ...

class ConcatParameter(_message.Message):
    __slots__ = ["axis", "concat_dim"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    CONCAT_DIM_FIELD_NUMBER: _ClassVar[int]
    axis: int
    concat_dim: int
    def __init__(self, axis: _Optional[int] = ..., concat_dim: _Optional[int] = ...) -> None: ...

class ContrastiveLossParameter(_message.Message):
    __slots__ = ["margin"]
    MARGIN_FIELD_NUMBER: _ClassVar[int]
    margin: float
    def __init__(self, margin: _Optional[float] = ...) -> None: ...

class ConvolutionParameter(_message.Message):
    __slots__ = ["num_output", "bias_term", "pad", "pad_h", "pad_w", "kernel_size", "kernel_h", "kernel_w", "group", "stride", "stride_h", "stride_w", "hole", "hole_h", "hole_w", "weight_filler", "bias_filler", "engine", "ForwardAlgo", "ntile_width", "ntile_height", "quantize_param", "is_relu", "perchannel_quantize_param", "out_pad", "out_pad_h", "out_pad_w"]
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[ConvolutionParameter.Engine]
        CAFFE: _ClassVar[ConvolutionParameter.Engine]
        CUDNN: _ClassVar[ConvolutionParameter.Engine]
    DEFAULT: ConvolutionParameter.Engine
    CAFFE: ConvolutionParameter.Engine
    CUDNN: ConvolutionParameter.Engine
    class PPLConvolutionForwardAlgo_t(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        PPL_CONVOLUTION_FORWARD_ALGO_NONE: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_IMPLICIT_GEMM: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_IMPLICIT_PRECOMP_GEMM: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_GEMM: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_DIRECT: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_FFT: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_FFT_TILING: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK2X2: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK4X4: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK6X6: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK4x4_FLT3x3: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK4x4_FLT5X5: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
        PPL_CONVOLUTION_FORWARD_ALGO_INVGEMM: _ClassVar[ConvolutionParameter.PPLConvolutionForwardAlgo_t]
    PPL_CONVOLUTION_FORWARD_ALGO_NONE: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_IMPLICIT_GEMM: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_IMPLICIT_PRECOMP_GEMM: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_GEMM: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_DIRECT: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_FFT: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_FFT_TILING: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK2X2: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK4X4: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK6X6: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK4x4_FLT3x3: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_WINOGRAD_BLK4x4_FLT5X5: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    PPL_CONVOLUTION_FORWARD_ALGO_INVGEMM: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BIAS_TERM_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    HOLE_FIELD_NUMBER: _ClassVar[int]
    HOLE_H_FIELD_NUMBER: _ClassVar[int]
    HOLE_W_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    FORWARDALGO_FIELD_NUMBER: _ClassVar[int]
    NTILE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NTILE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    QUANTIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    IS_RELU_FIELD_NUMBER: _ClassVar[int]
    PERCHANNEL_QUANTIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_H_FIELD_NUMBER: _ClassVar[int]
    OUT_PAD_W_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    bias_term: bool
    pad: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_h: int
    kernel_w: int
    group: int
    stride: int
    stride_h: int
    stride_w: int
    hole: int
    hole_h: int
    hole_w: int
    weight_filler: FillerParameter
    bias_filler: FillerParameter
    engine: ConvolutionParameter.Engine
    ForwardAlgo: ConvolutionParameter.PPLConvolutionForwardAlgo_t
    ntile_width: int
    ntile_height: int
    quantize_param: QuantizeParameter
    is_relu: bool
    perchannel_quantize_param: _containers.RepeatedCompositeFieldContainer[QuantizeParameter]
    out_pad: int
    out_pad_h: int
    out_pad_w: int
    def __init__(self, num_output: _Optional[int] = ..., bias_term: bool = ..., pad: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., group: _Optional[int] = ..., stride: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., hole: _Optional[int] = ..., hole_h: _Optional[int] = ..., hole_w: _Optional[int] = ..., weight_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., engine: _Optional[_Union[ConvolutionParameter.Engine, str]] = ..., ForwardAlgo: _Optional[_Union[ConvolutionParameter.PPLConvolutionForwardAlgo_t, str]] = ..., ntile_width: _Optional[int] = ..., ntile_height: _Optional[int] = ..., quantize_param: _Optional[_Union[QuantizeParameter, _Mapping]] = ..., is_relu: bool = ..., perchannel_quantize_param: _Optional[_Iterable[_Union[QuantizeParameter, _Mapping]]] = ..., out_pad: _Optional[int] = ..., out_pad_h: _Optional[int] = ..., out_pad_w: _Optional[int] = ...) -> None: ...

class DataParameter(_message.Message):
    __slots__ = ["source", "batch_size", "rand_skip", "backend", "scale", "mean_file", "crop_size", "mirror", "force_encoded_color"]
    class DB(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        LEVELDB: _ClassVar[DataParameter.DB]
        LMDB: _ClassVar[DataParameter.DB]
    LEVELDB: DataParameter.DB
    LMDB: DataParameter.DB
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    RAND_SKIP_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    MEAN_FILE_FIELD_NUMBER: _ClassVar[int]
    CROP_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIRROR_FIELD_NUMBER: _ClassVar[int]
    FORCE_ENCODED_COLOR_FIELD_NUMBER: _ClassVar[int]
    source: str
    batch_size: int
    rand_skip: int
    backend: DataParameter.DB
    scale: float
    mean_file: str
    crop_size: int
    mirror: bool
    force_encoded_color: bool
    def __init__(self, source: _Optional[str] = ..., batch_size: _Optional[int] = ..., rand_skip: _Optional[int] = ..., backend: _Optional[_Union[DataParameter.DB, str]] = ..., scale: _Optional[float] = ..., mean_file: _Optional[str] = ..., crop_size: _Optional[int] = ..., mirror: bool = ..., force_encoded_color: bool = ...) -> None: ...

class NonMaximumSuppressionParameter(_message.Message):
    __slots__ = ["nms_threshold", "top_k"]
    NMS_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    nms_threshold: float
    top_k: int
    def __init__(self, nms_threshold: _Optional[float] = ..., top_k: _Optional[int] = ...) -> None: ...

class SaveOutputParameter(_message.Message):
    __slots__ = ["output_directory", "output_name_prefix", "output_format", "label_map_file", "name_size_file", "num_test_image"]
    OUTPUT_DIRECTORY_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FORMAT_FIELD_NUMBER: _ClassVar[int]
    LABEL_MAP_FILE_FIELD_NUMBER: _ClassVar[int]
    NAME_SIZE_FILE_FIELD_NUMBER: _ClassVar[int]
    NUM_TEST_IMAGE_FIELD_NUMBER: _ClassVar[int]
    output_directory: str
    output_name_prefix: str
    output_format: str
    label_map_file: str
    name_size_file: str
    num_test_image: int
    def __init__(self, output_directory: _Optional[str] = ..., output_name_prefix: _Optional[str] = ..., output_format: _Optional[str] = ..., label_map_file: _Optional[str] = ..., name_size_file: _Optional[str] = ..., num_test_image: _Optional[int] = ...) -> None: ...

class DetectionOutputParameter(_message.Message):
    __slots__ = ["num_classes", "share_location", "background_label_id", "nms_param", "save_output_param", "code_type", "variance_encoded_in_target", "keep_top_k", "confidence_threshold", "visualize", "visualize_threshold"]
    NUM_CLASSES_FIELD_NUMBER: _ClassVar[int]
    SHARE_LOCATION_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_LABEL_ID_FIELD_NUMBER: _ClassVar[int]
    NMS_PARAM_FIELD_NUMBER: _ClassVar[int]
    SAVE_OUTPUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    CODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    VARIANCE_ENCODED_IN_TARGET_FIELD_NUMBER: _ClassVar[int]
    KEEP_TOP_K_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    VISUALIZE_FIELD_NUMBER: _ClassVar[int]
    VISUALIZE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    num_classes: int
    share_location: bool
    background_label_id: int
    nms_param: NonMaximumSuppressionParameter
    save_output_param: SaveOutputParameter
    code_type: PriorBoxParameter.CodeType
    variance_encoded_in_target: bool
    keep_top_k: int
    confidence_threshold: float
    visualize: bool
    visualize_threshold: float
    def __init__(self, num_classes: _Optional[int] = ..., share_location: bool = ..., background_label_id: _Optional[int] = ..., nms_param: _Optional[_Union[NonMaximumSuppressionParameter, _Mapping]] = ..., save_output_param: _Optional[_Union[SaveOutputParameter, _Mapping]] = ..., code_type: _Optional[_Union[PriorBoxParameter.CodeType, str]] = ..., variance_encoded_in_target: bool = ..., keep_top_k: _Optional[int] = ..., confidence_threshold: _Optional[float] = ..., visualize: bool = ..., visualize_threshold: _Optional[float] = ...) -> None: ...

class DropoutParameter(_message.Message):
    __slots__ = ["dropout_ratio"]
    DROPOUT_RATIO_FIELD_NUMBER: _ClassVar[int]
    dropout_ratio: float
    def __init__(self, dropout_ratio: _Optional[float] = ...) -> None: ...

class DummyDataParameter(_message.Message):
    __slots__ = ["data_filler", "shape", "num", "channels", "height", "width"]
    DATA_FILLER_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    NUM_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    data_filler: _containers.RepeatedCompositeFieldContainer[FillerParameter]
    shape: _containers.RepeatedCompositeFieldContainer[BlobShape]
    num: _containers.RepeatedScalarFieldContainer[int]
    channels: _containers.RepeatedScalarFieldContainer[int]
    height: _containers.RepeatedScalarFieldContainer[int]
    width: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data_filler: _Optional[_Iterable[_Union[FillerParameter, _Mapping]]] = ..., shape: _Optional[_Iterable[_Union[BlobShape, _Mapping]]] = ..., num: _Optional[_Iterable[int]] = ..., channels: _Optional[_Iterable[int]] = ..., height: _Optional[_Iterable[int]] = ..., width: _Optional[_Iterable[int]] = ...) -> None: ...

class EltwiseParameter(_message.Message):
    __slots__ = ["operation", "coeff", "stable_prod_grad"]
    class EltwiseOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        PROD: _ClassVar[EltwiseParameter.EltwiseOp]
        SUM: _ClassVar[EltwiseParameter.EltwiseOp]
        MAX: _ClassVar[EltwiseParameter.EltwiseOp]
    PROD: EltwiseParameter.EltwiseOp
    SUM: EltwiseParameter.EltwiseOp
    MAX: EltwiseParameter.EltwiseOp
    OPERATION_FIELD_NUMBER: _ClassVar[int]
    COEFF_FIELD_NUMBER: _ClassVar[int]
    STABLE_PROD_GRAD_FIELD_NUMBER: _ClassVar[int]
    operation: EltwiseParameter.EltwiseOp
    coeff: _containers.RepeatedScalarFieldContainer[float]
    stable_prod_grad: bool
    def __init__(self, operation: _Optional[_Union[EltwiseParameter.EltwiseOp, str]] = ..., coeff: _Optional[_Iterable[float]] = ..., stable_prod_grad: bool = ...) -> None: ...

class ExpParameter(_message.Message):
    __slots__ = ["base", "scale", "shift"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SHIFT_FIELD_NUMBER: _ClassVar[int]
    base: float
    scale: float
    shift: float
    def __init__(self, base: _Optional[float] = ..., scale: _Optional[float] = ..., shift: _Optional[float] = ...) -> None: ...

class LogParameter(_message.Message):
    __slots__ = ["base", "scale", "shift"]
    BASE_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SHIFT_FIELD_NUMBER: _ClassVar[int]
    base: float
    scale: float
    shift: float
    def __init__(self, base: _Optional[float] = ..., scale: _Optional[float] = ..., shift: _Optional[float] = ...) -> None: ...

class HDF5DataParameter(_message.Message):
    __slots__ = ["source", "batch_size"]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    source: str
    batch_size: int
    def __init__(self, source: _Optional[str] = ..., batch_size: _Optional[int] = ...) -> None: ...

class HDF5OutputParameter(_message.Message):
    __slots__ = ["file_name"]
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    file_name: str
    def __init__(self, file_name: _Optional[str] = ...) -> None: ...

class HingeLossParameter(_message.Message):
    __slots__ = ["norm"]
    class Norm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        L1: _ClassVar[HingeLossParameter.Norm]
        L2: _ClassVar[HingeLossParameter.Norm]
    L1: HingeLossParameter.Norm
    L2: HingeLossParameter.Norm
    NORM_FIELD_NUMBER: _ClassVar[int]
    norm: HingeLossParameter.Norm
    def __init__(self, norm: _Optional[_Union[HingeLossParameter.Norm, str]] = ...) -> None: ...

class ImageDataParameter(_message.Message):
    __slots__ = ["source", "batch_size", "rand_skip", "shuffle", "new_height", "new_width", "is_color", "scale", "mean_file", "crop_size", "mirror", "root_folder"]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    RAND_SKIP_FIELD_NUMBER: _ClassVar[int]
    SHUFFLE_FIELD_NUMBER: _ClassVar[int]
    NEW_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    NEW_WIDTH_FIELD_NUMBER: _ClassVar[int]
    IS_COLOR_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    MEAN_FILE_FIELD_NUMBER: _ClassVar[int]
    CROP_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIRROR_FIELD_NUMBER: _ClassVar[int]
    ROOT_FOLDER_FIELD_NUMBER: _ClassVar[int]
    source: str
    batch_size: int
    rand_skip: int
    shuffle: bool
    new_height: int
    new_width: int
    is_color: bool
    scale: float
    mean_file: str
    crop_size: int
    mirror: bool
    root_folder: str
    def __init__(self, source: _Optional[str] = ..., batch_size: _Optional[int] = ..., rand_skip: _Optional[int] = ..., shuffle: bool = ..., new_height: _Optional[int] = ..., new_width: _Optional[int] = ..., is_color: bool = ..., scale: _Optional[float] = ..., mean_file: _Optional[str] = ..., crop_size: _Optional[int] = ..., mirror: bool = ..., root_folder: _Optional[str] = ...) -> None: ...

class InfogainLossParameter(_message.Message):
    __slots__ = ["source"]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    source: str
    def __init__(self, source: _Optional[str] = ...) -> None: ...

class InnerProductParameter(_message.Message):
    __slots__ = ["num_output", "bias_term", "weight_filler", "bias_filler", "axis", "quantize_param", "perchannel_quantize_param"]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BIAS_TERM_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    PERCHANNEL_QUANTIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    bias_term: bool
    weight_filler: FillerParameter
    bias_filler: FillerParameter
    axis: int
    quantize_param: QuantizeParameter
    perchannel_quantize_param: _containers.RepeatedCompositeFieldContainer[QuantizeParameter]
    def __init__(self, num_output: _Optional[int] = ..., bias_term: bool = ..., weight_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., axis: _Optional[int] = ..., quantize_param: _Optional[_Union[QuantizeParameter, _Mapping]] = ..., perchannel_quantize_param: _Optional[_Iterable[_Union[QuantizeParameter, _Mapping]]] = ...) -> None: ...

class LRNParameter(_message.Message):
    __slots__ = ["local_size", "alpha", "beta", "norm_region", "k"]
    class NormRegion(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        ACROSS_CHANNELS: _ClassVar[LRNParameter.NormRegion]
        WITHIN_CHANNEL: _ClassVar[LRNParameter.NormRegion]
    ACROSS_CHANNELS: LRNParameter.NormRegion
    WITHIN_CHANNEL: LRNParameter.NormRegion
    LOCAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    BETA_FIELD_NUMBER: _ClassVar[int]
    NORM_REGION_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    local_size: int
    alpha: float
    beta: float
    norm_region: LRNParameter.NormRegion
    k: float
    def __init__(self, local_size: _Optional[int] = ..., alpha: _Optional[float] = ..., beta: _Optional[float] = ..., norm_region: _Optional[_Union[LRNParameter.NormRegion, str]] = ..., k: _Optional[float] = ...) -> None: ...

class MemoryDataParameter(_message.Message):
    __slots__ = ["batch_size", "channels", "height", "width"]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    channels: int
    height: int
    width: int
    def __init__(self, batch_size: _Optional[int] = ..., channels: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ...) -> None: ...

class MVNParameter(_message.Message):
    __slots__ = ["normalize_variance", "across_channels"]
    NORMALIZE_VARIANCE_FIELD_NUMBER: _ClassVar[int]
    ACROSS_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    normalize_variance: bool
    across_channels: bool
    def __init__(self, normalize_variance: bool = ..., across_channels: bool = ...) -> None: ...

class PermuteParameter(_message.Message):
    __slots__ = ["order"]
    ORDER_FIELD_NUMBER: _ClassVar[int]
    order: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, order: _Optional[_Iterable[int]] = ...) -> None: ...

class PoolingParameter(_message.Message):
    __slots__ = ["pool", "pad", "pad_h", "pad_w", "kernel_size", "kernel_h", "kernel_w", "stride", "stride_h", "stride_w", "engine", "global_pooling", "ceil_mode"]
    class PoolMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MAX: _ClassVar[PoolingParameter.PoolMethod]
        AVE: _ClassVar[PoolingParameter.PoolMethod]
        STOCHASTIC: _ClassVar[PoolingParameter.PoolMethod]
        AVE_COUNT_INCLUDE_PADDING: _ClassVar[PoolingParameter.PoolMethod]
        AVE_COUNT_EXCLUDE_PADDING: _ClassVar[PoolingParameter.PoolMethod]
    MAX: PoolingParameter.PoolMethod
    AVE: PoolingParameter.PoolMethod
    STOCHASTIC: PoolingParameter.PoolMethod
    AVE_COUNT_INCLUDE_PADDING: PoolingParameter.PoolMethod
    AVE_COUNT_EXCLUDE_PADDING: PoolingParameter.PoolMethod
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[PoolingParameter.Engine]
        CAFFE: _ClassVar[PoolingParameter.Engine]
        CUDNN: _ClassVar[PoolingParameter.Engine]
    DEFAULT: PoolingParameter.Engine
    CAFFE: PoolingParameter.Engine
    CUDNN: PoolingParameter.Engine
    POOL_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    KERNEL_H_FIELD_NUMBER: _ClassVar[int]
    KERNEL_W_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    STRIDE_H_FIELD_NUMBER: _ClassVar[int]
    STRIDE_W_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_POOLING_FIELD_NUMBER: _ClassVar[int]
    CEIL_MODE_FIELD_NUMBER: _ClassVar[int]
    pool: PoolingParameter.PoolMethod
    pad: int
    pad_h: int
    pad_w: int
    kernel_size: int
    kernel_h: int
    kernel_w: int
    stride: int
    stride_h: int
    stride_w: int
    engine: PoolingParameter.Engine
    global_pooling: bool
    ceil_mode: bool
    def __init__(self, pool: _Optional[_Union[PoolingParameter.PoolMethod, str]] = ..., pad: _Optional[int] = ..., pad_h: _Optional[int] = ..., pad_w: _Optional[int] = ..., kernel_size: _Optional[int] = ..., kernel_h: _Optional[int] = ..., kernel_w: _Optional[int] = ..., stride: _Optional[int] = ..., stride_h: _Optional[int] = ..., stride_w: _Optional[int] = ..., engine: _Optional[_Union[PoolingParameter.Engine, str]] = ..., global_pooling: bool = ..., ceil_mode: bool = ...) -> None: ...

class PoolingSpecificParameter(_message.Message):
    __slots__ = ["pool", "spe_w", "spe_h"]
    class PoolSpecificMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MAX: _ClassVar[PoolingSpecificParameter.PoolSpecificMethod]
        AVE: _ClassVar[PoolingSpecificParameter.PoolSpecificMethod]
    MAX: PoolingSpecificParameter.PoolSpecificMethod
    AVE: PoolingSpecificParameter.PoolSpecificMethod
    POOL_FIELD_NUMBER: _ClassVar[int]
    SPE_W_FIELD_NUMBER: _ClassVar[int]
    SPE_H_FIELD_NUMBER: _ClassVar[int]
    pool: PoolingSpecificParameter.PoolSpecificMethod
    spe_w: int
    spe_h: int
    def __init__(self, pool: _Optional[_Union[PoolingSpecificParameter.PoolSpecificMethod, str]] = ..., spe_w: _Optional[int] = ..., spe_h: _Optional[int] = ...) -> None: ...

class PowerParameter(_message.Message):
    __slots__ = ["power", "scale", "shift"]
    POWER_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SHIFT_FIELD_NUMBER: _ClassVar[int]
    power: float
    scale: float
    shift: float
    def __init__(self, power: _Optional[float] = ..., scale: _Optional[float] = ..., shift: _Optional[float] = ...) -> None: ...

class PriorBoxParameter(_message.Message):
    __slots__ = ["min_size", "max_size", "aspect_ratio", "flip", "clip", "variance"]
    class CodeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        CORNER: _ClassVar[PriorBoxParameter.CodeType]
        CENTER_SIZE: _ClassVar[PriorBoxParameter.CodeType]
    CORNER: PriorBoxParameter.CodeType
    CENTER_SIZE: PriorBoxParameter.CodeType
    MIN_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_SIZE_FIELD_NUMBER: _ClassVar[int]
    ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    FLIP_FIELD_NUMBER: _ClassVar[int]
    CLIP_FIELD_NUMBER: _ClassVar[int]
    VARIANCE_FIELD_NUMBER: _ClassVar[int]
    min_size: float
    max_size: float
    aspect_ratio: _containers.RepeatedScalarFieldContainer[float]
    flip: bool
    clip: bool
    variance: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, min_size: _Optional[float] = ..., max_size: _Optional[float] = ..., aspect_ratio: _Optional[_Iterable[float]] = ..., flip: bool = ..., clip: bool = ..., variance: _Optional[_Iterable[float]] = ...) -> None: ...

class PythonParameter(_message.Message):
    __slots__ = ["module", "layer"]
    MODULE_FIELD_NUMBER: _ClassVar[int]
    LAYER_FIELD_NUMBER: _ClassVar[int]
    module: str
    layer: str
    def __init__(self, module: _Optional[str] = ..., layer: _Optional[str] = ...) -> None: ...

class ReLUParameter(_message.Message):
    __slots__ = ["negative_slope", "engine"]
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[ReLUParameter.Engine]
        CAFFE: _ClassVar[ReLUParameter.Engine]
        CUDNN: _ClassVar[ReLUParameter.Engine]
    DEFAULT: ReLUParameter.Engine
    CAFFE: ReLUParameter.Engine
    CUDNN: ReLUParameter.Engine
    NEGATIVE_SLOPE_FIELD_NUMBER: _ClassVar[int]
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    negative_slope: float
    engine: ReLUParameter.Engine
    def __init__(self, negative_slope: _Optional[float] = ..., engine: _Optional[_Union[ReLUParameter.Engine, str]] = ...) -> None: ...

class ROIPoolingParameter(_message.Message):
    __slots__ = ["pooled_h", "pooled_w", "spatial_scale"]
    POOLED_H_FIELD_NUMBER: _ClassVar[int]
    POOLED_W_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    pooled_h: int
    pooled_w: int
    spatial_scale: float
    def __init__(self, pooled_h: _Optional[int] = ..., pooled_w: _Optional[int] = ..., spatial_scale: _Optional[float] = ...) -> None: ...

class ReshapeParameter(_message.Message):
    __slots__ = ["shape", "axis", "num_axes"]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    NUM_AXES_FIELD_NUMBER: _ClassVar[int]
    shape: BlobShape
    axis: int
    num_axes: int
    def __init__(self, shape: _Optional[_Union[BlobShape, _Mapping]] = ..., axis: _Optional[int] = ..., num_axes: _Optional[int] = ...) -> None: ...

class SigmoidParameter(_message.Message):
    __slots__ = ["engine"]
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[SigmoidParameter.Engine]
        CAFFE: _ClassVar[SigmoidParameter.Engine]
        CUDNN: _ClassVar[SigmoidParameter.Engine]
    DEFAULT: SigmoidParameter.Engine
    CAFFE: SigmoidParameter.Engine
    CUDNN: SigmoidParameter.Engine
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    engine: SigmoidParameter.Engine
    def __init__(self, engine: _Optional[_Union[SigmoidParameter.Engine, str]] = ...) -> None: ...

class SliceParameter(_message.Message):
    __slots__ = ["axis", "slice_point", "slice_dim"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    SLICE_POINT_FIELD_NUMBER: _ClassVar[int]
    SLICE_DIM_FIELD_NUMBER: _ClassVar[int]
    axis: int
    slice_point: _containers.RepeatedScalarFieldContainer[int]
    slice_dim: int
    def __init__(self, axis: _Optional[int] = ..., slice_point: _Optional[_Iterable[int]] = ..., slice_dim: _Optional[int] = ...) -> None: ...

class SoftmaxParameter(_message.Message):
    __slots__ = ["engine", "axis"]
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[SoftmaxParameter.Engine]
        CAFFE: _ClassVar[SoftmaxParameter.Engine]
        CUDNN: _ClassVar[SoftmaxParameter.Engine]
    DEFAULT: SoftmaxParameter.Engine
    CAFFE: SoftmaxParameter.Engine
    CUDNN: SoftmaxParameter.Engine
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    engine: SoftmaxParameter.Engine
    axis: int
    def __init__(self, engine: _Optional[_Union[SoftmaxParameter.Engine, str]] = ..., axis: _Optional[int] = ...) -> None: ...

class TanHParameter(_message.Message):
    __slots__ = ["engine"]
    class Engine(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        DEFAULT: _ClassVar[TanHParameter.Engine]
        CAFFE: _ClassVar[TanHParameter.Engine]
        CUDNN: _ClassVar[TanHParameter.Engine]
    DEFAULT: TanHParameter.Engine
    CAFFE: TanHParameter.Engine
    CUDNN: TanHParameter.Engine
    ENGINE_FIELD_NUMBER: _ClassVar[int]
    engine: TanHParameter.Engine
    def __init__(self, engine: _Optional[_Union[TanHParameter.Engine, str]] = ...) -> None: ...

class ThresholdParameter(_message.Message):
    __slots__ = ["threshold"]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    threshold: float
    def __init__(self, threshold: _Optional[float] = ...) -> None: ...

class WindowDataParameter(_message.Message):
    __slots__ = ["source", "scale", "mean_file", "batch_size", "crop_size", "mirror", "fg_threshold", "bg_threshold", "fg_fraction", "context_pad", "crop_mode", "cache_images", "root_folder"]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    MEAN_FILE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    CROP_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIRROR_FIELD_NUMBER: _ClassVar[int]
    FG_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    BG_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    FG_FRACTION_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_PAD_FIELD_NUMBER: _ClassVar[int]
    CROP_MODE_FIELD_NUMBER: _ClassVar[int]
    CACHE_IMAGES_FIELD_NUMBER: _ClassVar[int]
    ROOT_FOLDER_FIELD_NUMBER: _ClassVar[int]
    source: str
    scale: float
    mean_file: str
    batch_size: int
    crop_size: int
    mirror: bool
    fg_threshold: float
    bg_threshold: float
    fg_fraction: float
    context_pad: int
    crop_mode: str
    cache_images: bool
    root_folder: str
    def __init__(self, source: _Optional[str] = ..., scale: _Optional[float] = ..., mean_file: _Optional[str] = ..., batch_size: _Optional[int] = ..., crop_size: _Optional[int] = ..., mirror: bool = ..., fg_threshold: _Optional[float] = ..., bg_threshold: _Optional[float] = ..., fg_fraction: _Optional[float] = ..., context_pad: _Optional[int] = ..., crop_mode: _Optional[str] = ..., cache_images: bool = ..., root_folder: _Optional[str] = ...) -> None: ...

class BNParameter(_message.Message):
    __slots__ = ["scale_filler", "shift_filler", "var_eps", "moving_average", "decay"]
    SCALE_FILLER_FIELD_NUMBER: _ClassVar[int]
    SHIFT_FILLER_FIELD_NUMBER: _ClassVar[int]
    VAR_EPS_FIELD_NUMBER: _ClassVar[int]
    MOVING_AVERAGE_FIELD_NUMBER: _ClassVar[int]
    DECAY_FIELD_NUMBER: _ClassVar[int]
    scale_filler: FillerParameter
    shift_filler: FillerParameter
    var_eps: float
    moving_average: bool
    decay: float
    def __init__(self, scale_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., shift_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., var_eps: _Optional[float] = ..., moving_average: bool = ..., decay: _Optional[float] = ...) -> None: ...

class CTCParameter(_message.Message):
    __slots__ = ["threshold", "decode_type"]
    class Decoder(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        best_path: _ClassVar[CTCParameter.Decoder]
        best_path_thres: _ClassVar[CTCParameter.Decoder]
        prefix_search: _ClassVar[CTCParameter.Decoder]
    best_path: CTCParameter.Decoder
    best_path_thres: CTCParameter.Decoder
    prefix_search: CTCParameter.Decoder
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DECODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    threshold: float
    decode_type: CTCParameter.Decoder
    def __init__(self, threshold: _Optional[float] = ..., decode_type: _Optional[_Union[CTCParameter.Decoder, str]] = ...) -> None: ...

class PReLUParameter(_message.Message):
    __slots__ = ["filler", "channel_shared", "quantize_param"]
    FILLER_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_SHARED_FIELD_NUMBER: _ClassVar[int]
    QUANTIZE_PARAM_FIELD_NUMBER: _ClassVar[int]
    filler: FillerParameter
    channel_shared: bool
    quantize_param: QuantizeParameter
    def __init__(self, filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., channel_shared: bool = ..., quantize_param: _Optional[_Union[QuantizeParameter, _Mapping]] = ...) -> None: ...

class CropParameter(_message.Message):
    __slots__ = ["type", "crop_w", "crop_h", "print_info"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        CENTER: _ClassVar[CropParameter.Type]
        RANDOM: _ClassVar[CropParameter.Type]
    CENTER: CropParameter.Type
    RANDOM: CropParameter.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CROP_W_FIELD_NUMBER: _ClassVar[int]
    CROP_H_FIELD_NUMBER: _ClassVar[int]
    PRINT_INFO_FIELD_NUMBER: _ClassVar[int]
    type: CropParameter.Type
    crop_w: int
    crop_h: int
    print_info: bool
    def __init__(self, type: _Optional[_Union[CropParameter.Type, str]] = ..., crop_w: _Optional[int] = ..., crop_h: _Optional[int] = ..., print_info: bool = ...) -> None: ...

class AffineTransParameter(_message.Message):
    __slots__ = ["output_h", "output_w", "border_value", "affine_mat", "scale", "translate_x", "translate_y"]
    OUTPUT_H_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_W_FIELD_NUMBER: _ClassVar[int]
    BORDER_VALUE_FIELD_NUMBER: _ClassVar[int]
    AFFINE_MAT_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    TRANSLATE_X_FIELD_NUMBER: _ClassVar[int]
    TRANSLATE_Y_FIELD_NUMBER: _ClassVar[int]
    output_h: int
    output_w: int
    border_value: float
    affine_mat: _containers.RepeatedScalarFieldContainer[float]
    scale: float
    translate_x: float
    translate_y: float
    def __init__(self, output_h: _Optional[int] = ..., output_w: _Optional[int] = ..., border_value: _Optional[float] = ..., affine_mat: _Optional[_Iterable[float]] = ..., scale: _Optional[float] = ..., translate_x: _Optional[float] = ..., translate_y: _Optional[float] = ...) -> None: ...

class AffineTransPointParameter(_message.Message):
    __slots__ = ["inverse", "translate_x", "translate_y", "affine_mat", "scale"]
    INVERSE_FIELD_NUMBER: _ClassVar[int]
    TRANSLATE_X_FIELD_NUMBER: _ClassVar[int]
    TRANSLATE_Y_FIELD_NUMBER: _ClassVar[int]
    AFFINE_MAT_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    inverse: bool
    translate_x: float
    translate_y: float
    affine_mat: _containers.RepeatedScalarFieldContainer[float]
    scale: float
    def __init__(self, inverse: bool = ..., translate_x: _Optional[float] = ..., translate_y: _Optional[float] = ..., affine_mat: _Optional[_Iterable[float]] = ..., scale: _Optional[float] = ...) -> None: ...

class CalcAffineMatParameter(_message.Message):
    __slots__ = ["landmark_x", "landmark_y"]
    LANDMARK_X_FIELD_NUMBER: _ClassVar[int]
    LANDMARK_Y_FIELD_NUMBER: _ClassVar[int]
    landmark_x: _containers.RepeatedScalarFieldContainer[float]
    landmark_y: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, landmark_x: _Optional[_Iterable[float]] = ..., landmark_y: _Optional[_Iterable[float]] = ...) -> None: ...

class ROIParameter(_message.Message):
    __slots__ = ["crop_h", "crop_w", "center_x", "center_y", "type"]
    CROP_H_FIELD_NUMBER: _ClassVar[int]
    CROP_W_FIELD_NUMBER: _ClassVar[int]
    CENTER_X_FIELD_NUMBER: _ClassVar[int]
    CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    crop_h: int
    crop_w: int
    center_x: _containers.RepeatedScalarFieldContainer[float]
    center_y: _containers.RepeatedScalarFieldContainer[float]
    type: InterpType
    def __init__(self, crop_h: _Optional[int] = ..., crop_w: _Optional[int] = ..., center_x: _Optional[_Iterable[float]] = ..., center_y: _Optional[_Iterable[float]] = ..., type: _Optional[_Union[InterpType, str]] = ...) -> None: ...

class InterpParameter(_message.Message):
    __slots__ = ["height", "width", "zoom_factor", "shrink_factor", "pad_beg", "pad_end", "align_corners"]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    ZOOM_FACTOR_FIELD_NUMBER: _ClassVar[int]
    SHRINK_FACTOR_FIELD_NUMBER: _ClassVar[int]
    PAD_BEG_FIELD_NUMBER: _ClassVar[int]
    PAD_END_FIELD_NUMBER: _ClassVar[int]
    ALIGN_CORNERS_FIELD_NUMBER: _ClassVar[int]
    height: int
    width: int
    zoom_factor: int
    shrink_factor: int
    pad_beg: int
    pad_end: int
    align_corners: bool
    def __init__(self, height: _Optional[int] = ..., width: _Optional[int] = ..., zoom_factor: _Optional[int] = ..., shrink_factor: _Optional[int] = ..., pad_beg: _Optional[int] = ..., pad_end: _Optional[int] = ..., align_corners: bool = ...) -> None: ...

class RecurrentParameter(_message.Message):
    __slots__ = ["num_output", "weight_filler", "bias_filler", "debug_info"]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    DEBUG_INFO_FIELD_NUMBER: _ClassVar[int]
    num_output: int
    weight_filler: FillerParameter
    bias_filler: FillerParameter
    debug_info: bool
    def __init__(self, num_output: _Optional[int] = ..., weight_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., debug_info: bool = ...) -> None: ...

class FlattenParameter(_message.Message):
    __slots__ = ["axis", "end_axis"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    END_AXIS_FIELD_NUMBER: _ClassVar[int]
    axis: int
    end_axis: int
    def __init__(self, axis: _Optional[int] = ..., end_axis: _Optional[int] = ...) -> None: ...

class BatchNormParameter(_message.Message):
    __slots__ = ["use_global_stats", "moving_average_fraction", "eps"]
    USE_GLOBAL_STATS_FIELD_NUMBER: _ClassVar[int]
    MOVING_AVERAGE_FRACTION_FIELD_NUMBER: _ClassVar[int]
    EPS_FIELD_NUMBER: _ClassVar[int]
    use_global_stats: bool
    moving_average_fraction: float
    eps: float
    def __init__(self, use_global_stats: bool = ..., moving_average_fraction: _Optional[float] = ..., eps: _Optional[float] = ...) -> None: ...

class V1LayerParameter(_message.Message):
    __slots__ = ["bottom", "top", "name", "include", "exclude", "type", "blobs", "param", "blob_share_mode", "blobs_lr", "weight_decay", "loss_weight", "accuracy_param", "argmax_param", "concat_param", "contrastive_loss_param", "convolution_param", "data_param", "dropout_param", "dummy_data_param", "eltwise_param", "exp_param", "hdf5_data_param", "hdf5_output_param", "hinge_loss_param", "image_data_param", "infogain_loss_param", "inner_product_param", "lrn_param", "memory_data_param", "mvn_param", "pooling_param", "power_param", "relu_param", "sigmoid_param", "softmax_param", "slice_param", "tanh_param", "threshold_param", "window_data_param", "transform_param", "loss_param", "layer"]
    class LayerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        NONE: _ClassVar[V1LayerParameter.LayerType]
        ABSVAL: _ClassVar[V1LayerParameter.LayerType]
        ACCURACY: _ClassVar[V1LayerParameter.LayerType]
        ARGMAX: _ClassVar[V1LayerParameter.LayerType]
        BNLL: _ClassVar[V1LayerParameter.LayerType]
        CONCAT: _ClassVar[V1LayerParameter.LayerType]
        CONTRASTIVE_LOSS: _ClassVar[V1LayerParameter.LayerType]
        CONVOLUTION: _ClassVar[V1LayerParameter.LayerType]
        DATA: _ClassVar[V1LayerParameter.LayerType]
        DECONVOLUTION: _ClassVar[V1LayerParameter.LayerType]
        DROPOUT: _ClassVar[V1LayerParameter.LayerType]
        DUMMY_DATA: _ClassVar[V1LayerParameter.LayerType]
        EUCLIDEAN_LOSS: _ClassVar[V1LayerParameter.LayerType]
        ELTWISE: _ClassVar[V1LayerParameter.LayerType]
        EXP: _ClassVar[V1LayerParameter.LayerType]
        FLATTEN: _ClassVar[V1LayerParameter.LayerType]
        HDF5_DATA: _ClassVar[V1LayerParameter.LayerType]
        HDF5_OUTPUT: _ClassVar[V1LayerParameter.LayerType]
        HINGE_LOSS: _ClassVar[V1LayerParameter.LayerType]
        IM2COL: _ClassVar[V1LayerParameter.LayerType]
        IMAGE_DATA: _ClassVar[V1LayerParameter.LayerType]
        INFOGAIN_LOSS: _ClassVar[V1LayerParameter.LayerType]
        INNER_PRODUCT: _ClassVar[V1LayerParameter.LayerType]
        LRN: _ClassVar[V1LayerParameter.LayerType]
        MEMORY_DATA: _ClassVar[V1LayerParameter.LayerType]
        MULTINOMIAL_LOGISTIC_LOSS: _ClassVar[V1LayerParameter.LayerType]
        MVN: _ClassVar[V1LayerParameter.LayerType]
        POOLING: _ClassVar[V1LayerParameter.LayerType]
        POWER: _ClassVar[V1LayerParameter.LayerType]
        RELU: _ClassVar[V1LayerParameter.LayerType]
        SIGMOID: _ClassVar[V1LayerParameter.LayerType]
        SIGMOID_CROSS_ENTROPY_LOSS: _ClassVar[V1LayerParameter.LayerType]
        SILENCE: _ClassVar[V1LayerParameter.LayerType]
        SOFTMAX: _ClassVar[V1LayerParameter.LayerType]
        SOFTMAX_LOSS: _ClassVar[V1LayerParameter.LayerType]
        SPLIT: _ClassVar[V1LayerParameter.LayerType]
        SLICE: _ClassVar[V1LayerParameter.LayerType]
        TANH: _ClassVar[V1LayerParameter.LayerType]
        WINDOW_DATA: _ClassVar[V1LayerParameter.LayerType]
        THRESHOLD: _ClassVar[V1LayerParameter.LayerType]
    NONE: V1LayerParameter.LayerType
    ABSVAL: V1LayerParameter.LayerType
    ACCURACY: V1LayerParameter.LayerType
    ARGMAX: V1LayerParameter.LayerType
    BNLL: V1LayerParameter.LayerType
    CONCAT: V1LayerParameter.LayerType
    CONTRASTIVE_LOSS: V1LayerParameter.LayerType
    CONVOLUTION: V1LayerParameter.LayerType
    DATA: V1LayerParameter.LayerType
    DECONVOLUTION: V1LayerParameter.LayerType
    DROPOUT: V1LayerParameter.LayerType
    DUMMY_DATA: V1LayerParameter.LayerType
    EUCLIDEAN_LOSS: V1LayerParameter.LayerType
    ELTWISE: V1LayerParameter.LayerType
    EXP: V1LayerParameter.LayerType
    FLATTEN: V1LayerParameter.LayerType
    HDF5_DATA: V1LayerParameter.LayerType
    HDF5_OUTPUT: V1LayerParameter.LayerType
    HINGE_LOSS: V1LayerParameter.LayerType
    IM2COL: V1LayerParameter.LayerType
    IMAGE_DATA: V1LayerParameter.LayerType
    INFOGAIN_LOSS: V1LayerParameter.LayerType
    INNER_PRODUCT: V1LayerParameter.LayerType
    LRN: V1LayerParameter.LayerType
    MEMORY_DATA: V1LayerParameter.LayerType
    MULTINOMIAL_LOGISTIC_LOSS: V1LayerParameter.LayerType
    MVN: V1LayerParameter.LayerType
    POOLING: V1LayerParameter.LayerType
    POWER: V1LayerParameter.LayerType
    RELU: V1LayerParameter.LayerType
    SIGMOID: V1LayerParameter.LayerType
    SIGMOID_CROSS_ENTROPY_LOSS: V1LayerParameter.LayerType
    SILENCE: V1LayerParameter.LayerType
    SOFTMAX: V1LayerParameter.LayerType
    SOFTMAX_LOSS: V1LayerParameter.LayerType
    SPLIT: V1LayerParameter.LayerType
    SLICE: V1LayerParameter.LayerType
    TANH: V1LayerParameter.LayerType
    WINDOW_DATA: V1LayerParameter.LayerType
    THRESHOLD: V1LayerParameter.LayerType
    class DimCheckMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        STRICT: _ClassVar[V1LayerParameter.DimCheckMode]
        PERMISSIVE: _ClassVar[V1LayerParameter.DimCheckMode]
    STRICT: V1LayerParameter.DimCheckMode
    PERMISSIVE: V1LayerParameter.DimCheckMode
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    BLOBS_FIELD_NUMBER: _ClassVar[int]
    PARAM_FIELD_NUMBER: _ClassVar[int]
    BLOB_SHARE_MODE_FIELD_NUMBER: _ClassVar[int]
    BLOBS_LR_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    LOSS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_PARAM_FIELD_NUMBER: _ClassVar[int]
    ARGMAX_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONCAT_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONTRASTIVE_LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    CONVOLUTION_PARAM_FIELD_NUMBER: _ClassVar[int]
    DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    DROPOUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    DUMMY_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    ELTWISE_PARAM_FIELD_NUMBER: _ClassVar[int]
    EXP_PARAM_FIELD_NUMBER: _ClassVar[int]
    HDF5_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    HDF5_OUTPUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    HINGE_LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    IMAGE_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    INFOGAIN_LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    INNER_PRODUCT_PARAM_FIELD_NUMBER: _ClassVar[int]
    LRN_PARAM_FIELD_NUMBER: _ClassVar[int]
    MEMORY_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    MVN_PARAM_FIELD_NUMBER: _ClassVar[int]
    POOLING_PARAM_FIELD_NUMBER: _ClassVar[int]
    POWER_PARAM_FIELD_NUMBER: _ClassVar[int]
    RELU_PARAM_FIELD_NUMBER: _ClassVar[int]
    SIGMOID_PARAM_FIELD_NUMBER: _ClassVar[int]
    SOFTMAX_PARAM_FIELD_NUMBER: _ClassVar[int]
    SLICE_PARAM_FIELD_NUMBER: _ClassVar[int]
    TANH_PARAM_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_PARAM_FIELD_NUMBER: _ClassVar[int]
    WINDOW_DATA_PARAM_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_PARAM_FIELD_NUMBER: _ClassVar[int]
    LOSS_PARAM_FIELD_NUMBER: _ClassVar[int]
    LAYER_FIELD_NUMBER: _ClassVar[int]
    bottom: _containers.RepeatedScalarFieldContainer[str]
    top: _containers.RepeatedScalarFieldContainer[str]
    name: str
    include: _containers.RepeatedCompositeFieldContainer[NetStateRule]
    exclude: _containers.RepeatedCompositeFieldContainer[NetStateRule]
    type: V1LayerParameter.LayerType
    blobs: _containers.RepeatedCompositeFieldContainer[BlobProto]
    param: _containers.RepeatedScalarFieldContainer[str]
    blob_share_mode: _containers.RepeatedScalarFieldContainer[V1LayerParameter.DimCheckMode]
    blobs_lr: _containers.RepeatedScalarFieldContainer[float]
    weight_decay: _containers.RepeatedScalarFieldContainer[float]
    loss_weight: _containers.RepeatedScalarFieldContainer[float]
    accuracy_param: AccuracyParameter
    argmax_param: ArgMaxParameter
    concat_param: ConcatParameter
    contrastive_loss_param: ContrastiveLossParameter
    convolution_param: ConvolutionParameter
    data_param: DataParameter
    dropout_param: DropoutParameter
    dummy_data_param: DummyDataParameter
    eltwise_param: EltwiseParameter
    exp_param: ExpParameter
    hdf5_data_param: HDF5DataParameter
    hdf5_output_param: HDF5OutputParameter
    hinge_loss_param: HingeLossParameter
    image_data_param: ImageDataParameter
    infogain_loss_param: InfogainLossParameter
    inner_product_param: InnerProductParameter
    lrn_param: LRNParameter
    memory_data_param: MemoryDataParameter
    mvn_param: MVNParameter
    pooling_param: PoolingParameter
    power_param: PowerParameter
    relu_param: ReLUParameter
    sigmoid_param: SigmoidParameter
    softmax_param: SoftmaxParameter
    slice_param: SliceParameter
    tanh_param: TanHParameter
    threshold_param: ThresholdParameter
    window_data_param: WindowDataParameter
    transform_param: TransformationParameter
    loss_param: LossParameter
    layer: V0LayerParameter
    def __init__(self, bottom: _Optional[_Iterable[str]] = ..., top: _Optional[_Iterable[str]] = ..., name: _Optional[str] = ..., include: _Optional[_Iterable[_Union[NetStateRule, _Mapping]]] = ..., exclude: _Optional[_Iterable[_Union[NetStateRule, _Mapping]]] = ..., type: _Optional[_Union[V1LayerParameter.LayerType, str]] = ..., blobs: _Optional[_Iterable[_Union[BlobProto, _Mapping]]] = ..., param: _Optional[_Iterable[str]] = ..., blob_share_mode: _Optional[_Iterable[_Union[V1LayerParameter.DimCheckMode, str]]] = ..., blobs_lr: _Optional[_Iterable[float]] = ..., weight_decay: _Optional[_Iterable[float]] = ..., loss_weight: _Optional[_Iterable[float]] = ..., accuracy_param: _Optional[_Union[AccuracyParameter, _Mapping]] = ..., argmax_param: _Optional[_Union[ArgMaxParameter, _Mapping]] = ..., concat_param: _Optional[_Union[ConcatParameter, _Mapping]] = ..., contrastive_loss_param: _Optional[_Union[ContrastiveLossParameter, _Mapping]] = ..., convolution_param: _Optional[_Union[ConvolutionParameter, _Mapping]] = ..., data_param: _Optional[_Union[DataParameter, _Mapping]] = ..., dropout_param: _Optional[_Union[DropoutParameter, _Mapping]] = ..., dummy_data_param: _Optional[_Union[DummyDataParameter, _Mapping]] = ..., eltwise_param: _Optional[_Union[EltwiseParameter, _Mapping]] = ..., exp_param: _Optional[_Union[ExpParameter, _Mapping]] = ..., hdf5_data_param: _Optional[_Union[HDF5DataParameter, _Mapping]] = ..., hdf5_output_param: _Optional[_Union[HDF5OutputParameter, _Mapping]] = ..., hinge_loss_param: _Optional[_Union[HingeLossParameter, _Mapping]] = ..., image_data_param: _Optional[_Union[ImageDataParameter, _Mapping]] = ..., infogain_loss_param: _Optional[_Union[InfogainLossParameter, _Mapping]] = ..., inner_product_param: _Optional[_Union[InnerProductParameter, _Mapping]] = ..., lrn_param: _Optional[_Union[LRNParameter, _Mapping]] = ..., memory_data_param: _Optional[_Union[MemoryDataParameter, _Mapping]] = ..., mvn_param: _Optional[_Union[MVNParameter, _Mapping]] = ..., pooling_param: _Optional[_Union[PoolingParameter, _Mapping]] = ..., power_param: _Optional[_Union[PowerParameter, _Mapping]] = ..., relu_param: _Optional[_Union[ReLUParameter, _Mapping]] = ..., sigmoid_param: _Optional[_Union[SigmoidParameter, _Mapping]] = ..., softmax_param: _Optional[_Union[SoftmaxParameter, _Mapping]] = ..., slice_param: _Optional[_Union[SliceParameter, _Mapping]] = ..., tanh_param: _Optional[_Union[TanHParameter, _Mapping]] = ..., threshold_param: _Optional[_Union[ThresholdParameter, _Mapping]] = ..., window_data_param: _Optional[_Union[WindowDataParameter, _Mapping]] = ..., transform_param: _Optional[_Union[TransformationParameter, _Mapping]] = ..., loss_param: _Optional[_Union[LossParameter, _Mapping]] = ..., layer: _Optional[_Union[V0LayerParameter, _Mapping]] = ...) -> None: ...

class V0LayerParameter(_message.Message):
    __slots__ = ["name", "type", "num_output", "biasterm", "weight_filler", "bias_filler", "pad", "kernelsize", "group", "stride", "pool", "dropout_ratio", "local_size", "alpha", "beta", "k", "source", "scale", "meanfile", "batchsize", "cropsize", "mirror", "blobs", "blobs_lr", "weight_decay", "rand_skip", "det_fg_threshold", "det_bg_threshold", "det_fg_fraction", "det_context_pad", "det_crop_mode", "new_num", "new_channels", "new_height", "new_width", "shuffle_images", "concat_dim", "ntile_width", "ntile_height", "hdf5_output_param"]
    class PoolMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MAX: _ClassVar[V0LayerParameter.PoolMethod]
        AVE: _ClassVar[V0LayerParameter.PoolMethod]
        STOCHASTIC: _ClassVar[V0LayerParameter.PoolMethod]
    MAX: V0LayerParameter.PoolMethod
    AVE: V0LayerParameter.PoolMethod
    STOCHASTIC: V0LayerParameter.PoolMethod
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NUM_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    BIASTERM_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    PAD_FIELD_NUMBER: _ClassVar[int]
    KERNELSIZE_FIELD_NUMBER: _ClassVar[int]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    STRIDE_FIELD_NUMBER: _ClassVar[int]
    POOL_FIELD_NUMBER: _ClassVar[int]
    DROPOUT_RATIO_FIELD_NUMBER: _ClassVar[int]
    LOCAL_SIZE_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    BETA_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    MEANFILE_FIELD_NUMBER: _ClassVar[int]
    BATCHSIZE_FIELD_NUMBER: _ClassVar[int]
    CROPSIZE_FIELD_NUMBER: _ClassVar[int]
    MIRROR_FIELD_NUMBER: _ClassVar[int]
    BLOBS_FIELD_NUMBER: _ClassVar[int]
    BLOBS_LR_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    RAND_SKIP_FIELD_NUMBER: _ClassVar[int]
    DET_FG_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DET_BG_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DET_FG_FRACTION_FIELD_NUMBER: _ClassVar[int]
    DET_CONTEXT_PAD_FIELD_NUMBER: _ClassVar[int]
    DET_CROP_MODE_FIELD_NUMBER: _ClassVar[int]
    NEW_NUM_FIELD_NUMBER: _ClassVar[int]
    NEW_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    NEW_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    NEW_WIDTH_FIELD_NUMBER: _ClassVar[int]
    SHUFFLE_IMAGES_FIELD_NUMBER: _ClassVar[int]
    CONCAT_DIM_FIELD_NUMBER: _ClassVar[int]
    NTILE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NTILE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    HDF5_OUTPUT_PARAM_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    num_output: int
    biasterm: bool
    weight_filler: FillerParameter
    bias_filler: FillerParameter
    pad: int
    kernelsize: int
    group: int
    stride: int
    pool: V0LayerParameter.PoolMethod
    dropout_ratio: float
    local_size: int
    alpha: float
    beta: float
    k: float
    source: str
    scale: float
    meanfile: str
    batchsize: int
    cropsize: int
    mirror: bool
    blobs: _containers.RepeatedCompositeFieldContainer[BlobProto]
    blobs_lr: _containers.RepeatedScalarFieldContainer[float]
    weight_decay: _containers.RepeatedScalarFieldContainer[float]
    rand_skip: int
    det_fg_threshold: float
    det_bg_threshold: float
    det_fg_fraction: float
    det_context_pad: int
    det_crop_mode: str
    new_num: int
    new_channels: int
    new_height: int
    new_width: int
    shuffle_images: bool
    concat_dim: int
    ntile_width: int
    ntile_height: int
    hdf5_output_param: HDF5OutputParameter
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., num_output: _Optional[int] = ..., biasterm: bool = ..., weight_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., pad: _Optional[int] = ..., kernelsize: _Optional[int] = ..., group: _Optional[int] = ..., stride: _Optional[int] = ..., pool: _Optional[_Union[V0LayerParameter.PoolMethod, str]] = ..., dropout_ratio: _Optional[float] = ..., local_size: _Optional[int] = ..., alpha: _Optional[float] = ..., beta: _Optional[float] = ..., k: _Optional[float] = ..., source: _Optional[str] = ..., scale: _Optional[float] = ..., meanfile: _Optional[str] = ..., batchsize: _Optional[int] = ..., cropsize: _Optional[int] = ..., mirror: bool = ..., blobs: _Optional[_Iterable[_Union[BlobProto, _Mapping]]] = ..., blobs_lr: _Optional[_Iterable[float]] = ..., weight_decay: _Optional[_Iterable[float]] = ..., rand_skip: _Optional[int] = ..., det_fg_threshold: _Optional[float] = ..., det_bg_threshold: _Optional[float] = ..., det_fg_fraction: _Optional[float] = ..., det_context_pad: _Optional[int] = ..., det_crop_mode: _Optional[str] = ..., new_num: _Optional[int] = ..., new_channels: _Optional[int] = ..., new_height: _Optional[int] = ..., new_width: _Optional[int] = ..., shuffle_images: bool = ..., concat_dim: _Optional[int] = ..., ntile_width: _Optional[int] = ..., ntile_height: _Optional[int] = ..., hdf5_output_param: _Optional[_Union[HDF5OutputParameter, _Mapping]] = ...) -> None: ...

class ChannelShuffleParameter(_message.Message):
    __slots__ = ["group"]
    GROUP_FIELD_NUMBER: _ClassVar[int]
    group: int
    def __init__(self, group: _Optional[int] = ...) -> None: ...

class MovingAvgParameter(_message.Message):
    __slots__ = ["decay"]
    DECAY_FIELD_NUMBER: _ClassVar[int]
    decay: float
    def __init__(self, decay: _Optional[float] = ...) -> None: ...

class SlicingParameter(_message.Message):
    __slots__ = ["coefficient_len", "depth_d", "scale_x", "scale_y", "offset_x", "offset_y", "offset_z"]
    COEFFICIENT_LEN_FIELD_NUMBER: _ClassVar[int]
    DEPTH_D_FIELD_NUMBER: _ClassVar[int]
    SCALE_X_FIELD_NUMBER: _ClassVar[int]
    SCALE_Y_FIELD_NUMBER: _ClassVar[int]
    OFFSET_X_FIELD_NUMBER: _ClassVar[int]
    OFFSET_Y_FIELD_NUMBER: _ClassVar[int]
    OFFSET_Z_FIELD_NUMBER: _ClassVar[int]
    coefficient_len: int
    depth_d: int
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int
    offset_z: int
    def __init__(self, coefficient_len: _Optional[int] = ..., depth_d: _Optional[int] = ..., scale_x: _Optional[float] = ..., scale_y: _Optional[float] = ..., offset_x: _Optional[int] = ..., offset_y: _Optional[int] = ..., offset_z: _Optional[int] = ...) -> None: ...

class BilateralSlicingParameter(_message.Message):
    __slots__ = ["coefficient_len"]
    COEFFICIENT_LEN_FIELD_NUMBER: _ClassVar[int]
    coefficient_len: int
    def __init__(self, coefficient_len: _Optional[int] = ...) -> None: ...

class BilateralSliceApplyParameter(_message.Message):
    __slots__ = ["coefficient_len", "has_offset"]
    COEFFICIENT_LEN_FIELD_NUMBER: _ClassVar[int]
    HAS_OFFSET_FIELD_NUMBER: _ClassVar[int]
    coefficient_len: int
    has_offset: bool
    def __init__(self, coefficient_len: _Optional[int] = ..., has_offset: bool = ...) -> None: ...

class AdaptiveBilateralSliceApplyParameter(_message.Message):
    __slots__ = ["coefficient_len", "has_offset", "depth_d"]
    COEFFICIENT_LEN_FIELD_NUMBER: _ClassVar[int]
    HAS_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DEPTH_D_FIELD_NUMBER: _ClassVar[int]
    coefficient_len: int
    has_offset: bool
    depth_d: int
    def __init__(self, coefficient_len: _Optional[int] = ..., has_offset: bool = ..., depth_d: _Optional[int] = ...) -> None: ...

class PointsCurveParameter(_message.Message):
    __slots__ = ["numpoints", "slope_filler", "bias_filler"]
    NUMPOINTS_FIELD_NUMBER: _ClassVar[int]
    SLOPE_FILLER_FIELD_NUMBER: _ClassVar[int]
    BIAS_FILLER_FIELD_NUMBER: _ClassVar[int]
    numpoints: int
    slope_filler: FillerParameter
    bias_filler: FillerParameter
    def __init__(self, numpoints: _Optional[int] = ..., slope_filler: _Optional[_Union[FillerParameter, _Mapping]] = ..., bias_filler: _Optional[_Union[FillerParameter, _Mapping]] = ...) -> None: ...

class SubpixelDownParameter(_message.Message):
    __slots__ = ["downsample"]
    DOWNSAMPLE_FIELD_NUMBER: _ClassVar[int]
    downsample: int
    def __init__(self, downsample: _Optional[int] = ...) -> None: ...

class SubpixelUpParameter(_message.Message):
    __slots__ = ["upsample", "output_datatype", "mode"]
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        FLOAT32: _ClassVar[SubpixelUpParameter.DataType]
        UINT8: _ClassVar[SubpixelUpParameter.DataType]
    FLOAT32: SubpixelUpParameter.DataType
    UINT8: SubpixelUpParameter.DataType
    class ModeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        caffe: _ClassVar[SubpixelUpParameter.ModeType]
        pytorch: _ClassVar[SubpixelUpParameter.ModeType]
    caffe: SubpixelUpParameter.ModeType
    pytorch: SubpixelUpParameter.ModeType
    UPSAMPLE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DATATYPE_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    upsample: int
    output_datatype: SubpixelUpParameter.DataType
    mode: SubpixelUpParameter.ModeType
    def __init__(self, upsample: _Optional[int] = ..., output_datatype: _Optional[_Union[SubpixelUpParameter.DataType, str]] = ..., mode: _Optional[_Union[SubpixelUpParameter.ModeType, str]] = ...) -> None: ...

class ClipParameter(_message.Message):
    __slots__ = ["min", "max"]
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    min: float
    max: float
    def __init__(self, min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...

class EltwiseAffineTransformParameter(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class TileParameter(_message.Message):
    __slots__ = ["axis", "tiles"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    TILES_FIELD_NUMBER: _ClassVar[int]
    axis: int
    tiles: int
    def __init__(self, axis: _Optional[int] = ..., tiles: _Optional[int] = ...) -> None: ...

class ReduceParameter(_message.Message):
    __slots__ = ["axis", "mode"]
    class ReduceOp(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        MEAN: _ClassVar[ReduceParameter.ReduceOp]
    MEAN: ReduceParameter.ReduceOp
    AXIS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    axis: int
    mode: int
    def __init__(self, axis: _Optional[int] = ..., mode: _Optional[int] = ...) -> None: ...

class PadParameter(_message.Message):
    __slots__ = ["pad", "mode", "pad_w", "pad_h"]
    class PadMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        REFLECTION: _ClassVar[PadParameter.PadMode]
    REFLECTION: PadParameter.PadMode
    PAD_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    PAD_W_FIELD_NUMBER: _ClassVar[int]
    PAD_H_FIELD_NUMBER: _ClassVar[int]
    pad: int
    mode: PadParameter.PadMode
    pad_w: int
    pad_h: int
    def __init__(self, pad: _Optional[int] = ..., mode: _Optional[_Union[PadParameter.PadMode, str]] = ..., pad_w: _Optional[int] = ..., pad_h: _Optional[int] = ...) -> None: ...

class ReLU6Parameter(_message.Message):
    __slots__ = ["negative_slope"]
    NEGATIVE_SLOPE_FIELD_NUMBER: _ClassVar[int]
    negative_slope: float
    def __init__(self, negative_slope: _Optional[float] = ...) -> None: ...

class Correlation2DParameter(_message.Message):
    __slots__ = ["groups"]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    groups: int
    def __init__(self, groups: _Optional[int] = ...) -> None: ...

class MatMulParameter(_message.Message):
    __slots__ = ["mode"]
    class MatMulMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        NN: _ClassVar[MatMulParameter.MatMulMode]
        NT: _ClassVar[MatMulParameter.MatMulMode]
        TN: _ClassVar[MatMulParameter.MatMulMode]
        TT: _ClassVar[MatMulParameter.MatMulMode]
    NN: MatMulParameter.MatMulMode
    NT: MatMulParameter.MatMulMode
    TN: MatMulParameter.MatMulMode
    TT: MatMulParameter.MatMulMode
    MODE_FIELD_NUMBER: _ClassVar[int]
    mode: MatMulParameter.MatMulMode
    def __init__(self, mode: _Optional[_Union[MatMulParameter.MatMulMode, str]] = ...) -> None: ...

class ParameterParameter(_message.Message):
    __slots__ = ["batch", "m", "n", "channel", "height", "width"]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    M_FIELD_NUMBER: _ClassVar[int]
    N_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    batch: int
    m: int
    n: int
    channel: int
    height: int
    width: int
    def __init__(self, batch: _Optional[int] = ..., m: _Optional[int] = ..., n: _Optional[int] = ..., channel: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ...) -> None: ...

class PixelShuffleParameter(_message.Message):
    __slots__ = ["upscale_factor"]
    UPSCALE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    upscale_factor: int
    def __init__(self, upscale_factor: _Optional[int] = ...) -> None: ...

class ROITransformParameter(_message.Message):
    __slots__ = ["roi_pooled_h", "roi_pooled_w"]
    ROI_POOLED_H_FIELD_NUMBER: _ClassVar[int]
    ROI_POOLED_W_FIELD_NUMBER: _ClassVar[int]
    roi_pooled_h: int
    roi_pooled_w: int
    def __init__(self, roi_pooled_h: _Optional[int] = ..., roi_pooled_w: _Optional[int] = ...) -> None: ...

class MainTransformParameter(_message.Message):
    __slots__ = ["roi_pooled_h", "roi_pooled_w"]
    ROI_POOLED_H_FIELD_NUMBER: _ClassVar[int]
    ROI_POOLED_W_FIELD_NUMBER: _ClassVar[int]
    roi_pooled_h: int
    roi_pooled_w: int
    def __init__(self, roi_pooled_h: _Optional[int] = ..., roi_pooled_w: _Optional[int] = ...) -> None: ...

class InstanceNormParameter(_message.Message):
    __slots__ = ["num_features", "eps", "affine"]
    NUM_FEATURES_FIELD_NUMBER: _ClassVar[int]
    EPS_FIELD_NUMBER: _ClassVar[int]
    AFFINE_FIELD_NUMBER: _ClassVar[int]
    num_features: int
    eps: float
    affine: bool
    def __init__(self, num_features: _Optional[int] = ..., eps: _Optional[float] = ..., affine: bool = ...) -> None: ...

class GridSampleParameter(_message.Message):
    __slots__ = ["padding_mode"]
    class PaddingMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        ZEROS: _ClassVar[GridSampleParameter.PaddingMode]
        BORDER: _ClassVar[GridSampleParameter.PaddingMode]
    ZEROS: GridSampleParameter.PaddingMode
    BORDER: GridSampleParameter.PaddingMode
    PADDING_MODE_FIELD_NUMBER: _ClassVar[int]
    padding_mode: GridSampleParameter.PaddingMode
    def __init__(self, padding_mode: _Optional[_Union[GridSampleParameter.PaddingMode, str]] = ...) -> None: ...

class GridSample3DParameter(_message.Message):
    __slots__ = ["padding_mode", "align_corners", "output_channel"]
    class PaddingMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        ZEROS: _ClassVar[GridSample3DParameter.PaddingMode]
        BORDER: _ClassVar[GridSample3DParameter.PaddingMode]
    ZEROS: GridSample3DParameter.PaddingMode
    BORDER: GridSample3DParameter.PaddingMode
    PADDING_MODE_FIELD_NUMBER: _ClassVar[int]
    ALIGN_CORNERS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_CHANNEL_FIELD_NUMBER: _ClassVar[int]
    padding_mode: GridSample3DParameter.PaddingMode
    align_corners: bool
    output_channel: int
    def __init__(self, padding_mode: _Optional[_Union[GridSample3DParameter.PaddingMode, str]] = ..., align_corners: bool = ..., output_channel: _Optional[int] = ...) -> None: ...

class ReduceL2Parameter(_message.Message):
    __slots__ = ["axes", "keepdims"]
    AXES_FIELD_NUMBER: _ClassVar[int]
    KEEPDIMS_FIELD_NUMBER: _ClassVar[int]
    axes: int
    keepdims: int
    def __init__(self, axes: _Optional[int] = ..., keepdims: _Optional[int] = ...) -> None: ...

class VarianceParameter(_message.Message):
    __slots__ = ["dim", "keepdim"]
    DIM_FIELD_NUMBER: _ClassVar[int]
    KEEPDIM_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedScalarFieldContainer[int]
    keepdim: bool
    def __init__(self, dim: _Optional[_Iterable[int]] = ..., keepdim: bool = ...) -> None: ...

class MeanParameter(_message.Message):
    __slots__ = ["dim", "keepdim"]
    DIM_FIELD_NUMBER: _ClassVar[int]
    KEEPDIM_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedScalarFieldContainer[int]
    keepdim: bool
    def __init__(self, dim: _Optional[_Iterable[int]] = ..., keepdim: bool = ...) -> None: ...

class Interp3dParameter(_message.Message):
    __slots__ = ["depth", "height", "width", "zoom_factor", "shrink_factor", "pad_beg", "pad_end", "align_corners"]
    DEPTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    ZOOM_FACTOR_FIELD_NUMBER: _ClassVar[int]
    SHRINK_FACTOR_FIELD_NUMBER: _ClassVar[int]
    PAD_BEG_FIELD_NUMBER: _ClassVar[int]
    PAD_END_FIELD_NUMBER: _ClassVar[int]
    ALIGN_CORNERS_FIELD_NUMBER: _ClassVar[int]
    depth: int
    height: int
    width: int
    zoom_factor: int
    shrink_factor: int
    pad_beg: int
    pad_end: int
    align_corners: bool
    def __init__(self, depth: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., zoom_factor: _Optional[int] = ..., shrink_factor: _Optional[int] = ..., pad_beg: _Optional[int] = ..., pad_end: _Optional[int] = ..., align_corners: bool = ...) -> None: ...

class GRUParameter(_message.Message):
    __slots__ = ["hidden_size", "bidirectional"]
    HIDDEN_SIZE_FIELD_NUMBER: _ClassVar[int]
    BIDIRECTIONAL_FIELD_NUMBER: _ClassVar[int]
    hidden_size: int
    bidirectional: bool
    def __init__(self, hidden_size: _Optional[int] = ..., bidirectional: bool = ...) -> None: ...

class CorrelationMigParameter(_message.Message):
    __slots__ = ["shift_step"]
    SHIFT_STEP_FIELD_NUMBER: _ClassVar[int]
    shift_step: int
    def __init__(self, shift_step: _Optional[int] = ...) -> None: ...

class ArgSortParameter(_message.Message):
    __slots__ = ["descending", "axis"]
    DESCENDING_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    descending: bool
    axis: int
    def __init__(self, descending: bool = ..., axis: _Optional[int] = ...) -> None: ...

class CumProdParameter(_message.Message):
    __slots__ = ["axis"]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    axis: int
    def __init__(self, axis: _Optional[int] = ...) -> None: ...
