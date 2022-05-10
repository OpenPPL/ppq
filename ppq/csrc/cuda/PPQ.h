# include <vector>
# include <map>
# include <string>

# ifndef PPQ_CPP_EXTENSION
    /*
    RoundingPolicy is a core setting for PPQ quantization calculation.
        It defines rounding behaviour inside quantization calculation.

    Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

    PPQ Supports 7 different rounding policies now.
    Take a look at https://en.wikipedia.org/wiki/Rounding

    ATTENTION: RoundingPolicy greatly affects PPQ executor behaviour in some cases,
        to get a correct result from PPQ executor,
        make sure your RoundingPolicy is the same as your hardware.
    */
    # define PPQ_ROUND_HALF_EVEN     0
    # define PPQ_ROUND_HALF_UP       1
    # define PPQ_ROUND_HALF_DOWN     2
    # define PPQ_ROUND_HALF_TOWARDS_ZERO    3
    # define PPQ_ROUND_HALF_FAR_FORM_ZERO   4
    # define PPQ_ROUND_TO_NEAR_INT   5
    # define PPQ_ROUND_UP            6
    # define PPQ_ROUND_DOWN          7

    /*
    QuantizationStates is a core data structure for PPQ quantization.
    QuantizationStates tells whether a quantization configuration is activated.

    ATTENTION: Changes of QuantizationState will greatly affect execution result.

    For a TensorQuantizationConfig instance, there are 9 available quantization states now.
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

        SHAPE: whenever a tensor quantization configuration holds SHAPE state,
            it will be never quantized and will not be included into any optimization algorithm.
        it means underlying tensor is shape-related tensor, and it can not be quantized.

        ACTIVATE: means corresponding tensor is ready to be quantized with its configuration.

        PASSIVE: means corresponding tensor is ready to be quantized with its configuration.
            (however its configuration is not stand alone, it still depends on someone else.)

        BAKED: means corresponding tensor has been pre-quantized, its value can directly
            go forward without quantization.
    */
    # define PPQ_STATE_INITIAL       1   // 量化参数刚刚被初始化，当前 config 不生效，数据不能被使用
    # define PPQ_STATE_BAKED         2   // 只针对参数量化，表示参数已经被静态量化，当前 config 不生效，数据可以直接使用
    # define PPQ_STATE_OVERLAPPED    3   // 只针对activation量化，表示数据流的量化由其他 config 管理，当前 config 不生效
    # define PPQ_STATE_DEACTIVATED   4   // 表示当前 config 不生效
    # define PPQ_STATE_ACTIVATED     5   // 表示当前 config 生效
    # define PPQ_STATE_DEQUANTIZED   6   // 表示当前 config 被暂时停用
    # define PPQ_STATE_SHAPE         7   // 表示这一路输入与 Shape 相关，不量化
    # define PPQ_STATE_PASSIVE       8   // 表示这一路输入被动量化，如 bias, clip value 等
    # define PPQ_STATE_PASSIVE_INIT  9   // 表示这一路输入被动量化，并且刚刚初始化不能被使用

    /*
    QuantizationProperty is a core abstraction for PPQ quantization calculation.
    QuantizationProperty and QuantizationPolicy together build a bitmap to describe quantization policy.

    A QuantizationPolicy instance contains multiple QuantizationProperty,
        QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.

    During simulating, executor will quantize tensor corresponding to its QuantizationPolicy.
        (QuantizationPolicy is included by TensorQuantizationConfig)

    There are 7 different quantization property(s) supported by PPQ now.

        PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
            (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

        PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

        LINEAR: Indicates a linear quantization, follow formula: quant(x) = clip(round(x / scale))

        EXPONENTIAL: Indicates an exponential quantization, not yet used.

        SYMMETRICAL: Indicates a symmetrical quantization, offset is deactivated in this mode.

        ASYMMETRICAL: Indicates an asymmetrical quantization, offset is activated in this mode.

        POWER_OF_2: Indicates a power-of-2 quantization, scale must be pow(2, k) in this mode.

    ATTENTION: Not all combinations of all 7 QuantizationProperty are valid, see QuantizationPolicy.__check_valid
    ATTENTION: QuantizationPolicy is read-only, user can only assign its value when created, the only interface of
        QuantizationPolicy is function QuantizationPolicy.has_property.
    */
    # define PPQ_QPROPERTY_PER_TENSOR   0x00000001
    # define PPQ_QPROPERTY_PER_CHANNEL  0x00000002
    # define PPQ_QPROPERTY_LINEAR       0x00000004
    # define PPQ_QPROPERTY_EXPONENTIAL  0x00000008
    # define PPQ_QPROPERTY_SYMMETRICAL  0x00000010
    # define PPQ_QPROPERTY_ASYMMETRICAL 0x00000020
    # define PPQ_QPROPERTY_POWER_OF_2   0x00000040

    # define PPQ_CPP_EXTENSION
# endif


template<typename ScaleType>
class TensorQuantConfig {
    /*
    TensorQuantizationConfig, as known as tensor quantization configuration, is the most
        important data structure in PPQ system.

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
        (see the defination of PPQ.IR.quant.QuantableVariable for more information)

    You are supposed to change tensor quantization configuration during optimization passes, this abstraction
        is widely tested among various platforms, it shall satisfy most of your quantization demands.
    */
public:
    int state;
    int policy;
    int num_of_bits;
    int quant_min;
    int quant_max;
    int rounding;
    int channel_axis;
    ScaleType *scale; // scale 内存由 PPQ 管理，任何情况下都不可以释放这片内存
    int *offset; // offset 内存由 PPQ 管理，任何情况下都不可以释放这片内存

    /* TensorQuantConfig 作为 PPQ 送往后端框架的中间对象，只应该由 PPQ 完成创建。 */
    /* 我们直接删除了 TensorQuantConfig 的默认构造函数，拷贝构造函数，等号运算 */
    TensorQuantConfig() = delete;
    TensorQuantConfig(const TensorQuantConfig &obj) = delete;
    TensorQuantConfig& operator=(const TensorQuantConfig& obj) = delete;
    TensorQuantConfig(
        const int state, const int policy, const int num_of_bits,
        const int quant_min, const int quant_max,
        const int rounding, const int channel_axis,
        const Dtype *scale, const int *offset);
    /*
        Create a PPQ Tensor Quantization Configuration Instance

        Args:
            policy (QuantizationPolicy):
                Quantization policy instance which defines the quantization behaviour from marco view.

            rounding (RoundingPolicy): Rounding policy used in quantization.

            num_of_bits (int): Quantization bits. (2 < num_of_bits < 32)

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

            inplace (bool, optional): Indicates whether quantization is taken inplace(rewrite tensor value).

            state (QuantizationStates, optional):
                Defaults to QuantizationStates.INITIAL, see QuantizationStates for more detail.
    */

    /* TensorQuantConfig 内存由 PPQ 管理，不要释放其中指针指向的内存 */
    ~TensorQuantConfig() = delete;
};

template<typename ScaleType>
class OperatorQuantConfig {
public:
    std::vector<TensorQuantConfig<ScaleType>> input_configs;
    std::vector<TensorQuantConfig<ScaleType>> output_configs;

    /* OperatorQuantConfig 作为 PPQ 送往后端框架的中间对象，只应该由 PPQ 完成创建。 */
    /* 我们直接删除了 OperatorQuantConfig 的默认构造函数，拷贝构造函数，等号运算 */
    OperatorQuantConfig() = delete;
    OperatorQuantConfig(const OperatorQuantConfig &obj) = delete;
    OperatorQuantConfig& operator=(const OperatorQuantConfig& obj) = delete;
    OperatorQuantConfig(
        std::vector<TensorQuantConfig<ScaleType>> input_configs,
        std::vector<TensorQuantConfig<ScaleType>> output_configs);

    /* TensorQuantConfig 内存由 PPQ 管理，不要释放其管理的内存 */
    ~OperatorQuantConfig() = delete;
};

/*
    Inherit IPPQKernelImpl to create custimized PPQ C++ extenstion function.
*/
template<typename Dtype, typename ScaleType>
class IPPQKernelImpl{
public:
    virtual const int forward(
        const std::map<std::string, object> &operator_attributes,
        const std::vector<Dtype*> &inputs,
        const std::vector<Dtype*> &outputs,
        const OperatorQuantConfig<typename ScaleType> &config
    );
    virtual const int backward(
        const std::map<std::string, object> &operator_attributes,
        const std::vector<Dtype*> &inputs,
        const std::vector<Dtype*> &outputs,
        const OperatorQuantConfig<typename ScaleType> &config
    );
};
