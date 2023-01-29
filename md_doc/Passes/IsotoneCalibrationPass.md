## Isotone Calibration Pass(保序量化校准过程)

在神经网络中，一些算子的输出并不需要保证总体的精确性，而只关注于最大最小值所在的位置，
例如图像分类网络中，网络的输出通常是一个1000维的向量，用于表达图像属于特定类别的概率。
为了保证分类的正确性，我们并不需要这个1000维的向量在量化后是整体准确的，只需要其中的最大值出现在正确的位置上。
因此我们希望最大值与次大值之间相差至少半个 scale，并且次大值能够不被截断。

因此传统的 min-max, percentile, kl 方法在这一情景中并不能得到最高的分类精度，
保序量化是为了解决这一问题而设计的，在这一校准过程中，程序将网络输出变量的校准方式改写为 Isotone(保序校准)。
默认设置下，该过程只对 softmax 算子的输出进行保序校准。对于其他情况，用户需要手动指定需要进行保序校准的变量名。

保序量化需要设定一个分类轴，同样地以分类网络为例，其输出形为 [Batch, 1000]。
分类操作将在数据的最后一维展开，因此需要设置保序轴为 -1。

Algorithm:

For softmax or sigmoid activations, usually we just need
argmax(softmax(x)) == argmax(softmax(quant(x)))

Inspired by this Property, Isotone Observer is designed to provide an order-preserving calibration method,
    which cares only about argmax(x) [or argmin(x)]

To keep argmax(x) == argmax(quant(x)), we only need to
    distinguish the largest element and the second largert element with quantization

    let L1 represents the largest element of x,
    while L2 represents the second largest.

    For Symmetrical Quantization, We want:
        
        1. round(L1 / scale) - round(L2 / scale) > 0
        
        2. round(L2 / scale) < quant_max
        
    Hence that, we will have:
        
        1. scale < 2 * (L1 - L2)
        
        2. scale > L2 / (self._quant_cfg.quant_max - .5)
        
    For Asymmetircal Quantization, We want:

        1. round(L1 / scale) + offset - round(L2 / scale) - offset > 0

        2. round(L2 / scale) + offset < quant_max

    Hence that, we will have:
        
        1. scale < 2 * (L1 - L2)
        
        2. scale > L2 / (self._quant_cfg.quant_max - offset - .5)

The best setting of scale, offset can be solved by PPQ Isotone observer.

Time Complexity: O(nlogn)