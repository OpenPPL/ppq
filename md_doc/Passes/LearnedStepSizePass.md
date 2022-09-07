
## Learned Step Size Pass(网络微调过程-LSQ)

Learned Step Size optimization, a training-based optimization pass that tunes weights and scales for high precision quantization.

[This method is proposed by Steven K. Esser] (https://arxiv.org/pdf/1902.08153.pdf)

This is an alternative version of LSQ, this pass will split your graph into multiple trainable blocks, each blocks will be trained separately.
Warning: PPQ Learned Step Size minimize only the output loss of each block, which means after training the internal results probably goes far away from original. 

PPQ Learned Step Size optimization requires 256 ~ 2048 samples for finetuning your network, while the data label is not necessary. All training data are cache in GPU memory or CPU memory for acceleration.

The training loss will be computed as:

    let: Y = WX + b

    Quant(Y, scale_Y) = Qunat(W, scale_W) Quant(X, scale_X) + b

    loss = loss_func(Y, Quant(Y, scale_Y)) # loss between fp output and int8 output, that is why we do not need labeled data.

The formula of calculating the derivatives of y and scale_Y:

    if y > scale_Y * -128 and y < scale_Y * 127:
    dQuant(y, scale_Y)/dy       = dQuant(y, scale_Y)
    dQuant(y, scale_Y)/dscale_Y = Quant(y, scale_Y) - y

    if y < scale_Y * -128:
    dQuant(y, scale_Y)/dy       = 0
    dQuant(y, scale_Y)/dscale_Y = -128

    if y > scale_Y * 127:
    dQuant(y, scale_Y)/dy       = 0
    dQuant(y, scale_Y)/dscale_Y = 127

### Parameters:

* interested_layers(List[str]):

        A list of operation names, only the layers listed in this parameter will be trained.

        If interested_layers is None, all layers(conv and gemm) will be trained.

* steps(int)

        Training steps for finetuning your network, default is 500.

* block_size(int)

        PPQ Learned Step Size optimization split your graph into blocks at first, 
        each block will be finetuned separately.

        A large block size will greatly reduce running time of this optimization,
        while it might give an unstable result when blocksize is too large.

        By default this value is set to 4.

* is_scale_trainable(bool)

        If is_scale_trainable = False, optimization will not apply to scales, only network parameters will be tuned.

        Scale is trainable when all the following conditions are fulfilled:
            #   1. scale is valid
            #   2. corresponding tensor quantization config state is active
            #   3. do not have POWER_OF_2 policy
            #   4. is_scale_trainable = True

* gamma(float)

        A regularization term for minimize the distance of Y and Quant(Y)

        If gamma is not 0, loss = loss_func(Y, Quant(Y, scale_Y)) + MSE(Y, Quant(Y)) * gamma

        Default is 0

* lr(float)

        Learning rate, Default is 5e-5

* collecting_device(str)

        String that representing the device on which cache data is or will be allocated.

        Can be cpu, cuda, disk

* loss_fn(Callable)

        A function that used to measure the loss after optimization.

        LSQ is a training-based pass, 
        we will check the loss at the end of block optimization.

        If the result goes worsen, optimized weights and scales will be drop.

### Usage:

LSQ Optimization Pass should be invoked after Runtime Calibration Pass.

This pass is inclueded in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.lsq_optimization = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
    collate_fn=collate_fn)

You can manually create this optimization by:

    from ppq import LearnedStepSizePass

    optim = LearnedStepSizePass()


### Block-partitioning Algorithm

PPQ use Block-partitioning algorithm for spliting your graph into blocks, this method is based on graph theory.

Parameter block_size will controls the maximum size of created blocks.

If block_size = 1, then each block will contains exactly 1 layer within it, blockwise optimization will degenerate to layerwise optimization then.

If block_size is set to a large value, training progress will be unstable since batchnorm layers have been merged at first.

### Version:

Require PPQ 0.6.2 +

Interface changed since PPQ 0.6.5
