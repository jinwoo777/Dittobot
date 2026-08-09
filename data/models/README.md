# MediaPipe model assets

`hand_landmarker.task` is the Google MediaPipe Hand Landmarker float16 model bundle,
version 1:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`

The model is stored locally so hand tracking never downloads assets at runtime.

## Local tool detector

`yolov8n_tools_0122.pt` is the local tool detector supplied in `example.zip`.
It is loaded only by the optional live-scene detector and is never downloaded
at runtime. Its class map is: drill, hammer, pliers, screwdriver, wrench.

SHA-256: `a52a3aaadc8318fb65515ea859fb31b4c7383f1249e88d62d0f9c761b2266b34`
