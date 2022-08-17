## PPQ Quantize Fusion Pass(通用量化图融合过程)

Operation fusion (or kernel/layer fusion) is key optimization in many state-of-the-art execution frameworks.

Graph fusion can combine operations into a single op to obtain higher accuracy and performance,
    Pattern like: Conv + Relu can be reduced to ConvRelu. This fusion will reduce memory accesses,
    and the quantization point after conv can also be removed.

Technically we can fuse those layers before quantization, while fused layers are not supported by onnx standard.
    So to say ConvRelu is not a valid onnx operation, no execution framework can parse it.

Therefore, PPQ will simulate the graph fusion by adjusting quantization config: if PPQ finds their is a
    pattern like Conv + Relu, the output quantization of Conv will be forbidden, pretending that the Conv + Relu 
    fusion has happened.

This Pass is designed for 2 types fusion:

* activation fusion

    For activation fusion, PPQ will identify the pattern: Computing op + Activation Op from your network. The output
        quantization of computing op will be disabled with their state being set to QuantizationState.OVERLAPPED.

    Activation fusion here supports only simple activation patterns,
        for complex activation functions like mish, swish, 
        will be represented as mish = tanh + mul + softplus, swish = sigmoid + mul in onnx,
        cause onnx does not have a op defination for them.
        Identifying those complex patterns requires pattern matching, which is implemented in ppq.IR.search.py

    Complex quantization fusions must be invoked manually, PPQ implemented softplus & swish fusion functions in
        ppq.quantization.optim.refine.MishFusionPass
        ppq.quantization.optim.refine.SwishFusionPass

* passive operation fusion

    For passive operation fusion, PPQ will keep the input and the output variable share a same scale for passive operations.
        An operation is identified as passive op only if its attribute "is_active_quant_op" = False, this
        attribute is initilized by quantizer.

    If there is a passive operation having multiple input and output, the fusion procedure will make its
    FIRST input variable and ALL output variables share the same scale(the same scale as its first input).
    The quantization states of all output variables will be set to QuantizationState.OVERLAPPED.

### Parameters:

* activation_type(Set[str]):

        A collection contains all activation types.

        The pattern will be recognized as [Computing Op -> Activation Op],

        By graph fusion, the output quantization of the Computing Op and 
            the input quantization of the activation op will be disabled.

* fuse_activation(bool)

        Whether to fuse activation op with computing op.

* fuse_passive_op(bool)

        Whether to fuse passive op so that the input variable and output variable will share a same scale.

### Usage
This pass is included in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.fusion = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
    collate_fn=collate_fn)
