# Alerting Driver Drowsiness Detection  
 
A Python-based project to detect drowsiness in drivers using computer vision techniques and alert them with an audio alarm. The system ensures road safety by monitoring the driver’s facial features in real time.
  
## Features
- Detects drowsiness by monitoring eye aspect ratio (EAR). 
- Detects yawning by calculating the lip distance.
- Alerts the driver with an audio alarm when drowsiness or yawning is detected.

## Requirements
- Python 3.7 or later
- OpenCV
- dlib
- imutils
- numpy
- scipy
- pyttsx3

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/daniel-mwendwa/alerting-driver-drowsiness-detection.git
cd alerting-driver-drowsiness-detection

also download shape predictor from
https://github.com/MortezaNedaei/Facial-Landmarks-Detection/blob/master/shape_predictor_68_face_landmarks.dat
add this .dat file and the XML file in your project file.
