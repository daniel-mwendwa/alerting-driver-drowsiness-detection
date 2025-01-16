alerting-driver-drowsiness-detection

Overview

alerting-driver-drowsiness-detection is a Python-based project designed to detect driver drowsiness and yawning using a webcam. The system leverages facial landmark detection to monitor eye aspect ratio (EAR) and lip distance, triggering alerts when thresholds indicate drowsiness or yawning. The project aims to enhance road safety by alerting drivers to take preventive actions.

Features

Real-Time Eye Aspect Ratio (EAR) Monitoring: Detects prolonged eye closure indicating drowsiness.

Yawn Detection: Monitors lip distance to detect yawns.

Audio Alerts: Provides real-time audio warnings for drowsiness or yawning.

Facial Landmark Detection: Utilizes dlib’s 68-point facial landmark predictor for precise tracking.

Customizable Thresholds: Configurable EAR and yawn thresholds.

Prerequisites

Python 3.7 or higher

Webcam (for live video feed)

Installation

Clone the repository:

git clone https://github.com/your-username/alerting-driver-drowsiness-detection.git

Navigate to the project directory:

cd alerting-driver-drowsiness-detection

Install the required dependencies:

pip install -r requirements.txt

Dependencies

argparse: For command-line interface.

os: To execute system-level commands.

time: For handling time-related functionalities.

threading: For managing concurrent operations.

pyttsx3: For text-to-speech conversion.

cv2 (OpenCV): For real-time video processing and face detection.

dlib: For facial landmark detection.

imutils: For image and video processing utilities.

numpy: For numerical operations.

scipy: For distance calculations.

Usage

Ensure you have the required dependencies installed and a webcam connected.

Run the script:

python main.py --webcam 0

Replace 0 with the index of your webcam if using multiple cameras.

Press Q to quit the application.

How It Works

Eye Aspect Ratio (EAR):

Calculates the EAR using Euclidean distance between eye landmarks.

Triggers a drowsiness alert if the EAR is below the set threshold for a consecutive number of frames.

Yawn Detection:

Measures the distance between upper and lower lip landmarks.

Triggers a yawn alert if the distance exceeds the set threshold.

Audio Alerts:

Text-to-speech messages alert the driver about drowsiness or yawning.

Uses threading to ensure smooth execution without delays.

Thresholds

EAR Threshold: Default is 0.20. Can be adjusted in the code.

Yawn Distance Threshold: Default is 25. Can be adjusted in the code.

Files

main.py: The main script for running the application.

shape_predictor_68_face_landmarks.dat: Pre-trained dlib model for facial landmarks.

haarcascade_frontalface_default.xml: Haarcascade file for faster face detection.

Customization

Modify EAR and yawn thresholds in the script to suit specific requirements:

EYE_AR_THRESH = 0.20
YAWN_THRESH = 25

Use a different webcam by changing the --webcam argument.

Limitations

Lighting conditions and camera quality can affect detection accuracy.

Haarcascade face detection is faster but less accurate than dlib’s detector.

Future Improvements

Integrate support for external audio devices for louder alerts.

Enhance detection with deep learning models for higher accuracy.

Add mobile application integration for broader usability.
also download shape predictor from
https://github.com/MortezaNedaei/Facial-Landmarks-Detection/blob/master/shape_predictor_68_face_landmarks.dat
add this .dat file and the XML file in your project file.
