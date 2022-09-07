## PPQ Quantize Simplify Pass(通用量化精简过程)

PPQ use Tensor Quantization Configuration(A data structure defined in ppq.core) to
control quantization. Each quantable op will have a list of TQC as its quantization config,
which contains necessary quantization parameter(scale, offset), in order to quantize its input(s) and output(s).

While TQC is a powerful tool for describing quantization, it introduces some undiserible features:

For a subgraph like:

    Relu1 - Relu2

PPQ will create at least 4 TQC here, namely the input TQC of Relu1 and Relu2, and the output TQC of Relu1 and Relu2.
Problem here is the output TQC of Relu1 and the input TQC of Relu2 is actually duplicated, the output variable
should not be quantized twice.

This Simplify Pass will detect all the duplicated TQCs in your network, disable them and create a link with their
dominating TQCs. Disabled TQC will have and inactive state(QuantizationState.OVERRLAPED), so PPQ executor will 
simply ignore them when executing.

A duplicated TQC is an input TQC(A) whose binding variable has been quantized by another output TQC(B),
and the input TQC(A) should have the same bit-width as the output TQC(B)

### Parameters:

* No Parameter

### Usage
This pass is included in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.fusion = True
    setting.fusion_setting.remove_useless_quantization = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
    collate_fn=collate_fn)
