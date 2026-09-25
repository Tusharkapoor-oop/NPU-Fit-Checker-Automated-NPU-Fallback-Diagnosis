# Optimized (INT8) Profile Notes

- Device: Snapdragon X2 Elite CRD
- Model: MobileNetV2 (INT8 Quantized QDQ)
- Quantize Job URL: https://workbench.aihub.qualcomm.com/jobs/jgjrwld7p/
- Compile Job URL: https://workbench.aihub.qualcomm.com/jobs/jgzl47oz5/
- Profile Job URL: https://workbench.aihub.qualcomm.com/jobs/j57e7d0qp/
- Compile Options: --target_runtime onnx
- Profile Options: --onnx_execution_providers=qnn
- Raw Profile: experiments/optimized_int8/profile.json
