import argparse #for command line user interface
import os #provides easy use of operating system dependent functionalities
import time #provides various time related functionalities
from threading import Thread #to manage concurrent operation within a process
import pyttsx3 #python text to speech conversion

import cv2 #open computer vision, provides haarcascade file for face detection
import dlib #facial landmark detector with pre-trained model
import imutils #provides function for image resizing, rotation and translation
import numpy as np #translating image into grayscale
from imutils import face_utils #provides functions for rects
from imutils.video import VideoStream #used to resize the videostream resolution
from scipy.spatial import distance as dist #to calculate eucledian distance

engine = pyttsx3.init() #init function to get an engine instance for the speech synthesis
#ay method on the engine that passing input text to be spoken
engine.say("we wish you safe journey, please make sure you wear your seatbelt.")
engine.say("please follow system guidelines for safety on the road. thank you in advance")
#run and wait method, it processes the voice commands
engine.runAndWait()


def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying
    #alarm function which using espeak which is a tool to convert text to speech
    while alarm_status:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

# calculate the EAR using euclidean distance between the eye points
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5]) #p1-p5
    B = dist.euclidean(eye[2], eye[4]) #p2-p4

    C = dist.euclidean(eye[0], eye[3]) #p0-p3

    ear = (A + B) / (2.0 * C) #EAR formula
    return ear


def final_ear(shape):
    #getting start and end array index of each eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    #getting position of the right and left eye in the shape using dlibs points
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    #getting EAR for the two eyes using the function above
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0 #average for the two EAR
    return (ear, leftEye, rightEye) #returning ear, lefteye and righteye positions

#lip distance function
def lip_distance(shape):
    top_lip = shape[50:53] #upper lip points in dlibs landmark predictor
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59] #lower lip points in dlibs landmark predictor
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0) #calculating the mean of the upper points
    low_mean = np.mean(low_lip, axis=0) #calculating the mean of the lower points
    
    distance = abs(top_mean[1] - low_mean[1]) #calculate the difference in mean
    return distance

#take arguments from the command line. here am using default webcam
ap = argparse.ArgumentParser()
ap.add_argument("--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())
#declaring threshhold constants and initializing counter
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 25
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
print("->press Q to quit")
#starting videostream with the argument passed in this case it is 0 for webcam
vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
#use sleep function to initialize the camera
time.sleep(1.0)

while True: #infinite loop

    frame = vs.read() #reading frames from the camera
    frame = imutils.resize(frame, width=450) #resizing the frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converting frames to grayscale
    faces = detector.detectMultiScale(gray ,1.3, 5)
    for(x1,y1,w1,h2) in faces:
        cv2.rectangle(frame,(x1,y1),(x1+w1, y1+h2),(0,255,0),2)
   #  rects = detector(gray, 0)
    # detecting faces from the grayscale images
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # for rect in rects:
    for (x, y, w, h) in rects: #process each face in the rects
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))


        shape = predictor(gray, rect) #pass the frames to shape predictor to predict the 68 face landmarks
        shape = face_utils.shape_to_np(shape) #converting shape into numpy array

        eye = final_ear(shape) #passing shape into final ear function created above
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape) #passing the shape to lip distance function
        #drawing convexhull for the lefteye, righteye and then lip
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0,255,0), 1)

        if ear < EYE_AR_THRESH: #if ear is less than set threshold increment the counter
            COUNTER +=1

            if COUNTER >= EYE_AR_CONSEC_FRAMES: #the person is drowsing and not blinking
                if alarm_status == False and saying == False:
                    alarm_status = True
                    #use thread to keep the project processing even when alarm status =true
                    t = Thread(target=alarm, args=('please wake up',))
                    t.deamon = True
                    t.start()

                #outputting the drowsiness alert text
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #if ear !< threshold counter=0 and alarm status is initialized to False
        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH): #if distance is greater than set threshhold, puttext yawn alert
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #play the alert alarm
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('please take some fresh air',))
                t.deamon = True
                t.start()
        # if distance !> set threshhold, alarm status is set to False
        else:
            alarm_status2 = False
        # printing the EAR and YAWN distances
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #showing the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if q is pressed, the execution breaks
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
