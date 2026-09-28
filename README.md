#### **Look Again: Adaptive Re-Sensing for Edge Vision**



A lightweight edge-vision pipeline that decides \*when to look again\* before trusting a detection. It is demonstrated on left-behind-item detection.



###### **Idea**



Most detection pipelines either accept whatever confidence the model returns, or escalate straight to a bigger model when unsure. Look Again does something cheaper first: when a detection is uncertain, it crops and zooms into that region and re-runs the same detector to get better evidence. An escalation policy then picks the result, so re-sensing can never make a detection worse.



###### **Pipeline**



1\. Motion trigger: cheap frame differencing on webcam frames, no ML. The detector only runs when motion is seen.

2\. Detection: YOLOv7 (Qualcomm AI Hub Models) on the frame.

3\. Look again: detections with confidence between 0.50 and 0.75 are cropped (with context padding and a minimum crop size), upscaled, and re-run.

4\. Escalation policy: improved confidence is used; same or worse keeps the original; no detection on the crop falls back to the original.

5\. Event logic: a tracked item that stays put for 15 detections is flagged LEFT BEHIND. People are excluded, and only item classes (backpack, umbrella, handbag, suitcase, bottle, laptop, phone, book, and similar) can be flagged.



Every re-sense is logged to look\_again\_log.csv, and every stage is timed in timing\_log.csv.



###### **Results**



**Look-again (initial 47-event test run, webcam):**



\- 63.8% of uncertain detections improved after re-sensing; 36.2% kept the original.

\- Average confidence used rose from 0.674 to 0.716.

\- By design, the policy never lowers a confidence.



**Snapdragon NPU (Qualcomm AI Hub, hosted Snapdragon X Elite CRD, ONNX Runtime):**



\- 10.8 ms estimated inference time, versus about 79 ms per detection pass on the development laptop CPU.

\- 7 MB peak memory, and all 193 ops run on the NPU.

\- PSNR of 66 to 68 dB between on-device and local outputs, so accuracy is preserved.



###### **Honest notes and limitations**



\- The live demo was developed and run on a non-Snapdragon laptop (CPU). The NPU numbers come from Qualcomm AI Hub's hosted Snapdragon X Elite device, not local hardware.

\- Look-again is a digital crop and zoom, not optical zoom. About a third of re-checks on small or distant objects find nothing and fall back to the original detection.

\- A re-sense costs about one extra detector pass (\~79 ms on CPU). The saving is that this is only paid on uncertain detections.

\- The detector uses COCO's 80 classes, so objects outside COCO (for example, glasses) are not detected.

\- The motion trigger means a completely still scene is not re-evaluated.



**Setup (Windows)**



&#x20;   python -m venv qai\_env

&#x20;   qai\_env\\Scripts\\activate

&#x20;   pip install "qai-hub-models\[yolov7]" opencv-python

&#x20;   set PYTHONUTF8=1

&#x20;   chcp 65001

&#x20;   python motion\_monitor.py



Press q in the video window to quit.



**Snapdragon deployment**



&#x20;   qai-hub-models export yolov7 --device "Snapdragon X Elite CRD" --runtime onnx



This needs a free Qualcomm AI Hub account and API token.



###### **Files**



\- motion\_monitor.py: the full pipeline

\- look\_again\_log.csv: before and after confidence for every re-sense

\- timing\_log.csv: per-stage latency measurements

