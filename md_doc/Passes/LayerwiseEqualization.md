## Layer-wise Equalization Pass(层间权重均衡过程)

With only one quantization scale, per-tensor quantization has its trouble for representing the value among channels cause weight distributions can differ strongly between output channels. For example, in the case where one channel has weights in the range [−128, 128] and another channel has weights in the range (−0.5, 0.5), 
the weights in the latter channel will all be quantized to 0 when quantizing to 8-bits.

Hopefully, the performance can be improved by adjusting the weights for each output channel such that their ranges are more similar.

Formula:

        Take 2 convolution layers as an example

        Where Y = W_2 * (W_1 * X + b_1) + b_2

        Adjusting W_1, W_2 by a scale factor s:

        Y = W_2 / s * (W_1 * s * X + b_1 * s) + b_2

        Where s has the same dimension as the output channel of W_1

This method is called as Layer-wise Equalization, which is proposed by Markus Nagel.

https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf

### Parameters:

* iterations(int):

        Integer value of Algorithm iterations.

        More iterations will give more plainness in your weight distribution,
        iteration like 100 can flatten all the parameter in your network to a same level.

        You are not recommended to iterate until value converges, 
        in some cases stop iteration earlier will give you a better performance.

* weight_threshold(float)

        A threshold that stops processing value that is too small.

        By default, the scale factor of equalization method is computed as sqrt(max(abs(W_1)) / max(abs(W_2))),
        the maximum value of W_2 can be very small(like 1e-14), while the maximum value W_1 can be 0.5.

        In this case, the computed scale factor is 1e7, the optimization will loss its numerical stability and even give an unreasonable result.

        To prevent the scale factor becoming too large, ppq clips all the value smaller than this threshold before iterations.

        This parameter will significantly affects the optimization result.

        Recommended values are 0, 0.5, 2.

* including_bias(bool)

        Whether to include bias in computing scale factor.

        If including_bias is True, the scale factor will be computed as sqrt(max(abs(W_1 : b_1)) / max(abs(W_2 : b_2)))

        Where W_1 : b_1 mean an augmented matrix with W_1 and b_1

* including_bias(float)

        Only take effects when including_bias = True

        the scale factor will be computed as sqrt(max(abs(W_1 : b_1 * bias_multiplier)) / max(abs(W_2 : b_2 * bias_multiplier)))

        This is an correction term for bias.

* including_activation(bool)

        Same as the parameter including_bias, whether to include activation in computing scale factor.

* activation_multiplier(float)

        Same as the including_bias, this is an correction term for activation.

* optimize_level(int)

        level - 1: equalization will only cross ('Relu', 'MaxPool', 'GlobalMaxPool', 'PRelu', 'AveragePool', 'GlobalAveragePool')

        level - 2: equalization will cross ('Relu', 'MaxPool', 'GlobalMaxPool', 'Add', 'Sub', 'PRelu', 'AveragePool', 'GlobalAveragePool')

        Here is an example for illustrating the difference, if we got a graph like: 

            Conv1 - Relu - Conv2

        Both level - 1 and level - 2 optimization can find there is a equalization pair: (Conv1 - Conv2).

        however for a complex graph like: 

            Conv1 - Add - Conv2

        level - 1 optimization will simply skip Conv1 and Conv2.

        level - 2 optimization will trace another input from Add, and then PPQ will take all the input operations of Add
        as the upstream layers in equalization.

        PPQ use graph search engine for pasring graph structure, check ppq.IR.search.py for more information.

* interested_layers(List[str])

        Only layer that listed in interested_layers will be processed by this pass.

        If interested_layers is None or empty list, all the layers will be processed.

### Warning:
You can not compare a equalized graph with an unequalized graph layer by layer,
since equalization pass guarantees only the output of your network will be kept as same,
the intermediate result can be changed rapidly.

Since then, PPQ invokes this pass before network quantization.

### Usage
Layer-wise equalization are designed for per-layer quantization.

| Symmetrical | Asymmetrical | Per-chanel    | Per-tensor |
| ---         | ---          | ---           | ---        |
|             |              | Not recommend |            |

Layer-wise Equalization Pass should be invoked in pre-quant optimization pipeline which before the network quantization.

This pass is included in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.equalization = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
    collate_fn=collate_fn)

You can manually create this optimization by:

    from ppq import LayerwiseEqualizationPass

    optim = LayerwiseEqualizationPass()
