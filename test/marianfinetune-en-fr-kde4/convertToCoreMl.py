import coremltools as ct
import onnx

mlmodel = ct.converters.onnx.convert(
    model="marianmt.onnx",
    minimum_deployment_target=ct.target.iOS15  # or macOS13 if M-series
)
mlmodel.save("marianmt_float32.mlmodel")
