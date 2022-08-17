## Bias Correction Optimization Pass(Bias 校准过程)

  Bias correction is the process of shifting quantized model outputs to account for their statistical errors.

  Network quantization will bring some error(noise) to the result. To improve the accuracy of a quantized model, we can correct the network by adding an extra term on bias in order to make the output has zero expectation. 
  
  Bias correction is used to eliminate bias error, generally it will take a few mintues to correct all bias terms.

  For those layers have no bias, Bias Correction Optimization will skip them directly.

      let: Y = WX + b

      Quant(Y) = Qunat(W) Quant(X) + b

      bias_error = reduce_mean(Y - Quant(Y))

      This pass will correct bias with: b = b + bias_error

### Parameters:

  * interested_layers(List[str]):

        A list of operation names, only the layers listed in this parameter will be processed.

        If interested_layers is None, all layers will be processed.

  * steps(int)

        Forward steps for collecting bias error, a large value of this parameter means more data will be collected so the bias error will be estimated better, while it takes more time.

        Usually 8 ~ 32 step is enough in most cases.

  * block_size(int)

        Bias Correction Optimization will split your graph into blocks, bias error will be collected and corrected block by block.

        A large block size will greatly reduce running time of this optimization, while it might give an unstable result when blocksize is too large.

        By default this value is set to 4, to have the best result of optimization, you are recommended to set blocksize = 1.

  * loss_fn(Callable)

        A function that used to measure the loss after optimization.

        Bias Correction Optimization is a training-based pass, we will check the loss at the end of block optimization.

        If the optimization created worsen result, the optimization result will be drop.

### Usage:

  Bias Correction Optimization Pass should be invoked after Runtime Calibration Pass.

  This pass is inclueded in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.bias_correct = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
      model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
      platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
      collate_fn=collate_fn)

  You can manully create this optimization by:

    from ppq import BiasCorrectionPass

    optim = BiasCorrectionPass()

### Version:

Require PPQ 0.5.2 +

Interface changed since PPQ 0.6.5
