## Runtime Calibration Pass(量化参数校准过程)

For integer quantization, you need to calibrate or estimate the scale of all floating-point tensors in the model.

Formula:

        Quant(Y, scale_Y) = Clip(Round(Y / scale_Y))

        Dequant(Y, scale_Y) = Y * scale_Y

Only activations that have quantization state = INITIAL are going to be calibrated via this optimization pass. 
While if the parameter "override" is set to True, activations with quantization state = ACTIVATED will also be re-calibrated.

Runtime Calibration Pass will write estimated scales and offsets to tensor quantization configs, and set their state to ACTIVATED.

Unlike constant tensors such as weights and biases, variable tensors such as model input, activations (outputs of intermediate layers) and model output cannot be calibrated unless we run a few inference cycles. 

As a result, PPQ Runtime Calibration Pass requires a representative dataset to calibrate them. 

This dataset is supposed to be a small subset (around ~100-500 samples) of the training or validation data.

### Parameters:

* method(str):

        String that representing the algorithm used to estimate scales and offsets for activations.

        Can be mse, kl, percentile, minmax, this parameter is case insensitive.

        You can register your own calibration method through functions in ppq.api

* override(bool)

        if this parameter is set to True, activations with quantization state = ACTIVATED will also be re-calibrated, 
        runtime calibration pass will overwrite their scales and offsets.

        This parameter is introduced since ppq 0.6.4

### Observer Support Matrix:
| observer     | Symmetrical | Asymmetrical | Per-channel | Per-tensor | Cuda Acceleration   |
| ---          | ---         | ---          | ---        | ---        | ---                 |
| minmax       | &#10004;         | &#10004;          | &#10004;        | &#10004;        |                  |
| mse          | &#10004;         | &#10004;          |         | &#10004;        | &#10004;                 |
| precentile   | &#10004;         | &#10004;          | &#10004;        | &#10004;        | &#10004;               |
| kl           | &#10004;         |          |        | &#10004;        | &#10004;                 |
| isotone      | &#10004;         | &#10004;          |          | &#10004;        |                 |

If possible, using Cuda kernel can speed up observer by 10~100x.

### Usage:

Runtime Calibration Pass should be invoked before Passive Parameter Quantize Pass

This pass is included in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.quantize_activation = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
    collate_fn=collate_fn)

You can manually create this optimization by:

    from ppq import RuntimeCalibrationPass

    optim = RuntimeCalibrationPass()

### Register Calibration Method:

Using api function register_calibration_observer to resister new observer algorithm to PPQ system.
Once Algorithm is registered, Runtime Calibration Pass will automatically calling them by name.

This feature requires PPQ > 0.6.5
