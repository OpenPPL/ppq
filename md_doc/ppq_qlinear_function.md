# Introduction to PPQ QLinear Function:

Hello Developers, in this tutorial we'll show how to quantize a torch Tensor with ppq TensorQuantizationConfig.
First of all, we'd like to give you an example of how to use pytorch native functions to quantize a tensor:

    import torch
    v_fp32 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    scale, clip_min, clip_max = 2, -128, 127
    v_int8 = torch.clip(torch.round(v_fp32 / scale), min=clip_min, max=clip_max)
    print(f'Torch Quantized v_int8: {v_int8}')
    # tensor([0., 1., 2., 2., 2., 3., 4., 4., 4., 5.])

    # let's dequant v_int8
    dqv_fp32 = v_int8 * scale
    print(f'Torch Dequantized dqv_fp32: {dqv_fp32}')
    # tensor([ 0.,  2.,  4.,  4.,  4.,  6.,  8.,  8.,  8., 10.])

Here we take an "SYMMETRICAL" quantization on our tensor with scale = 2, the clip min and max was set as -128, 127 respectively. The quant function used here is:

    y = torch.clip(torch.round(x / scale), min=clip_min, max=clip_max)

Similarly, the dequant function used here is:

    y = x * scale

# Quantize a Convolution:

To quantize a convolution layer with pytorch, a resonable way is to quantize convolution input, weight before its execution, and quantize its output after we got its result. Following code block is a example to show how a convolution layer get quantized:

    import torch
    import torch.nn.functional as F

    def quant(tensor: torch.Tensor, scale = 0.1, clip_min = -128, clip_max = 127):
        y = torch.clip(torch.round(tensor / scale), min=clip_min, max=clip_max)
        y = y.char() # convert y to int8 dtype
        return y

    def dequant(tensor: torch.Tensor, scale = 0.1):
        y = tensor * scale
        y = y.float() # convert y to fp32 dtype
        return y

    # fp32 version
    v_fp32 = torch.rand(size=[1, 3, 96, 96])
    w_fp32 = torch.rand(size=[3, 3, 3, 3])
    b_fp32 = torch.rand(size=[3])
    o_fp32 = F.conv2d(v_fp32, w_fp32, b_fp32)
    print(o_fp32)
    
    # int8 version
    v_fp32 = torch.rand(size=[1, 3, 96, 96])
    w_fp32 = torch.rand(size=[3, 3, 3, 3])
    b_fp32 = torch.rand(size=[3])

    dqv_fp32 = dequant(quant(v_fp32))
    dqw_fp32 = dequant(quant(w_fp32))
    o_fp32 = F.conv2d(dqv_fp32, dqw_fp32, b_fp32)
    dqo_fp32 = dequant(quant(o_fp32))
    print(dqo_fp32)

Notice that we send dqv_fp32, dqw_fp32 to F.conv2d instead of v_int8, w_int8, although pytorch do have an implementation of int8 convolution in lastest version.
In fact using dequantized value will have the same result as using quantized int8 value, however applying int8 version will blocks gradient backpropagation during execution.
Gradient is necessary in PTQ, it enables us to finetuning your network for a better quantization performance, so we perfer to use dequantized value in forward execution.

# PPQ QLinear Function:

Now is time to play with PPQ Qlinear functions, let's start with import them from PPQ libraries:

    from ppq import TensorQuantizationConfig
    from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt, PPQLinearQuantFunction

PPQLinearQuantFunction and PPQLinearQuant_toInt are quant(dequant) functions used in PPQ executor: PPQLinearQuant_toInt will quantize a fp32 tensor to int8, PPQLinearQuantFunction will quantize and dequantize a fp32 tensor. TensorQuantizationConfig is the data structure to describe quantization parameter(scale, offset, and etc.). In other words, TensorQuantizationConfig tells how to quantize your tensor.

    import torch
    from ppq import TensorQuantizationConfig
    from ppq.core import *
    from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt, PPQLinearQuantFunction

    v_fp32 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tqc = TensorQuantizationConfig(
        policy = (QuantizationPolicy(
            QuantizationProperty.SYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR 
        )),
        rounding = RoundingPolicy.ROUND_HALF_EVEN,
        num_of_bits = 8,
        quant_min = -128, quant_max = 128,
        scale  = torch.tensor([2.0]),
        offset = torch.tensor([0.0]),
        observer_algorithm = None,
        state = QuantizationStates.ACTIVATED)
    v_int8   = PPQLinearQuant_toInt(tensor=v_fp32, config=tqc)
    dqv_fp32 = PPQLinearQuantFunction(tensor=v_fp32, config=tqc)
    print(f'PPQ Quantized v_int8: {v_int8}') # tensor([0, 1, 2, 2, 2, 3, 4, 4, 4, 5], dtype=torch.int32)
    print(f'PPQ Dequantized dqv_fp32: {dqv_fp32}') # tensor([ 0.,  2.,  4.,  4.,  4.,  6.,  8.,  8.,  8., 10.])

A TensorQuantizationConfig object is initialized here, the attributes of TensorQuantizationConfig are QuantizationPolicy, RoundingPolicy, num_of_bits, quant_min, quant_max, scale, offset and QuantizationStates.

## QuantizationPolicy: 
A QuantizationPolicy is a combination of some QuantizationPropertys, QuantizationPolicy is used in PPQ (alone with other configuration) to describe how a tensor is quantized.
There are 7 different quantization property(s) supported by PPQ now:

    PER_TENSOR: Also known as per-layer quantization, which mean all parameters of this layer share the same scale and offset.
        (For Convulution layer and Gemm layer which has bias, bias layer will be negative quantized, they do not have a valid scale)

    PER_CHANNEL: parameters are quantized alone channel axis, each channel has a stand-alone scale and offset.

    LINEAR: Indicates a linear quantization, follow formula: quant(x) = clip(round(x / scale))

    EXPONENTIAL: Indicates an exponential quantization, not yet used.

    SYMMETRICAL: Indicates a symmetrical quantization, offset is deactivated in this mode.

    ASYMMETRICAL: Indicates an asymmetrical quantization, offset is activated in this mode.

    POWER_OF_2: Indicates a power-of-2 quantization, scale must be pow(2, k) in this mode.

## RoundingPolicy
RoundingPolicy is a core setting for PPQ quantization calculation. It
defines rounding behaviour inside quantization calculation.

Formula: quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

PPQ Supports 7 different rounding policies now.
Take a look at https://en.wikipedia.org/wiki/Rounding

    ROUND_HALF_EVEN            = 0
    ROUND_HALF_UP              = 1
    ROUND_HALF_DOWN            = 2
    ROUND_HALF_TOWARDS_ZERO    = 3
    ROUND_HALF_FAR_FORM_ZERO   = 4
    ROUND_TO_NEAR_INT          = 5
    ROUND_UP                   = 6

## QParams
Scale, offset, quant_min, quant_max are all called as qparams, they are parameters used in quant function:

    quant(x) = clip(round(x / scale, RoundingPolicy), -128, 127)

offset will be ignored with SYMMETRICAL policy, scale and offset are supposed to be one-element tensor with PER_TENSOR policy. While for PER_CHANNEL policy, scale and offset are supposed to have one element for each channel.
Offset is always a float tensor in PPQ, PPQ executor will round it to int before execution.

## QuantizationStates
QuantizationStates is a core data structure for PPQ quantization. QuantizationStates tells whether a quantization configuration is activated.

For a TensorQuantizationConfig instance, there are 11 available quantization states now.

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
        (however its configuration is not stand alone, it still depends on someone else.)

    BAKED: means corresponding tensor has been pre-quantized, its value can directly
        go forward without quantization.
