## PPQ Quant Alignment Pass(通用量化对齐过程)

When deploy on real hardware and inference framework, 
    we will find that there are various restrictions or rules that we have to follow.

* AVERAGE_POOL_2D: Input and outputs must all have same scale/zero_point

* CONCATENATION: Input and outputs must all have same scale/zero_point

* SLICE: Input and outputs must all have same scale/zero_point

More detailed restrictions please refer to: https://www.tensorflow.org/lite/performance/quantization_spec
    
Those restrictions, can be concluded as some quantization should share 
    the same quantization parameter with others. PPQ Quant Alignment Pass is designed
    for dealing with problems like this.

PPQ uses Tensor Quantization Config (A data structure defined in ppq.core) to control the
    quantization logic, so to say if we want to align quantization parameters, we align
    their TQC in fact.

The way to align TQC is simple, code like:
    tqc1.set_master(master=tqc2)
Will make tqc1 and tqc2 share the same quantization parameters as tqc1 has, and change the
state of tqc2 to be QuantizationState.SLAVE

If we access the scale of tqc2, PPQ will return its master TQC's scale instead, so does offset.

That is tqc1 and tqc2 are bonuded with statement "tqc1.set_master(master=tqc2)".

### Parameters:

* elementwise_merge_method(Set[str]):

        Alignment method for elementwise ops.

        All elementwise ops are listed in ppq.core.common.py

* concat_merge_method(bool)

        Alignment method for concat-like ops.

        All concat-like ops are listed in ppq.core.common.py

* averagepool_method(bool)

        Alignment method for pooling-like ops.

        All pooling-like ops are listed in ppq.core.common.py

* force_overlap(bool)

        TQC alignment might cause serious cascade effect.

        For subgraph like this:

        Conv1 ---     
                + --- Add1
        Conv2 ---
                + --- Conv3

        If we demand Add1 to have same input scale, this alignment will affect Conv3 also,
            cause Conv2's output is feed to both Add1 and Conv3.

        If force_overlap = False, PPQ alignment procedure will remain the output scale of
            Conv2 as unchanged, while only align the input scale of Add1.

        If force_overlap = True, the input of Add1, Conv3 and the output of Conv2 will all
            be aligned to a same scale.

### Usage
This pass is included in PPQ Quantization Setting, you can calling this optimization by:

    setting = QuantizationSettingFactory.default_setting()

    setting.fusion = True
    setting.fusion_setting.force_alignment_overlap = True

    # calling ppq.api.quantize_onnx_model function with this setting.
    ir = quantize_torch_model(
    model=model, calib_dataloader=load_calibration_dataset(), setting=setting,
    platform=TargetPlatform.PPL_CUDA_INT8, calib_steps=8, input_shape=INPUT_SHAPE, 
    collate_fn=collate_fn)
