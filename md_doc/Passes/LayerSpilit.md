## Horizontal Layer Split Pass(算子分裂过程)

Split convolution layers or GEMM layers for better performance.

Formula:

    Y = W * X + b

    where W can be divided into W_1 + W_2

    Y = (W_1 * X + b) + (W_2 * X)

By splitting W like this, we are able to represent W more accurately. 
In the case where one channel has weights in the range [-32, 32] and another channel has weights in the range [-0.5, 0.5].
the large channel will be divided so the range will come to [-16, 16], which leads us to use scale = 0.125 for representing
the weight tensor rather than 0.25.

The Estimation of Quantization Error is shown as a quadratic function of scale:

        E(Quantization Error) = scale ^ 2 / 12

This Formula is proved by Bernard Widrow, according to the formula, a scale = 0.125 will decrease the quantization error by 75%.

All the value larger than value_threshold will be divided into 2 part via this function, thus the layer itself will be
splitted, an new Add operation are going to be created.

### Parameters:
    self.interested_layers = interested_layers
    self.value_threshold   = value_threshold
    self.method            = str(method).lower()
    self.verbose           = verbose

* interested_layers(List[str])

    Only layer that listed in interested_layers will be processed by this pass.

    If interested_layers is None or empty list, NO layer will be processed.

* value_threshold(float)

    This pass split value only when value is larger than value_threshold

    If there is no value large enough to be processed, corresponding layer will be skipped.

* method(str)

    Splitting method, 'balance' or 'random'

    With balance method, W_1 and W_2 will be evenly divided.

    With random method, W_1 and W_2 will be randomly divided.

### Warning:

Creating new operation in your network probably slows down the execution.

Thus horizontal splitting is somehow a trade-off between speed and accuracy.

### Usage

You can create this optimization manually:

    from ppq import HorizontalLayerSplitPass

    optim = HorizontalLayerSplitPass()
