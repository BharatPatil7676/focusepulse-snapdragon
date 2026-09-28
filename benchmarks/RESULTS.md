Profiled with Qualcomm AI Hub on a cloud-hosted Snapdragon X Elite CRD (Windows 11, ONNX

runtime). Inference times are the tool's estimates.





|**Model**|**Est. inference**| **Ops on NPU**|**CPU/GPU ops**|
|-|-|-|-|
|MediaPipe Face Detector|0.7ms|145/145|0|
|MediaPipe Face Landmark|0.3ms|105/105|0|



Accuracy vs local CPU reference: PSNR 67–138 dB (above 30 dB is considered good).



**Job links**

**Detector profile**: https://workbench.aihub.qualcomm.com/jobs/jgjr63jep/

**Landmark profile**: https://workbench.aihub.qualcomm.com/jobs/jpe706jv5/

**Accuracy check**: https://workbench.aihub.qualcomm.com/jobs/jgddy6jzg/



**Note**: The app in this repo currently runs MediaPipe on CPU; these results validate the NPU

deployment path.

