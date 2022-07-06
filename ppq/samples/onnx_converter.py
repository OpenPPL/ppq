import onnx
from onnx import version_converter, helper

# Preprocessing: load the model to be converted.
model_path = 'models/mbfill.onnx'
original_model = onnx.load(model_path)

# A full list of supported adapters can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/version_converter.py#L21
# Apply the version conversion on the original model
converted_model = version_converter.convert_version(original_model, 11)

onnx.save(converted_model, 'models/mbfill_11.onnx')