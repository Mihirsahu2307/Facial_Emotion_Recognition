from flask import Flask, render_template, Response, request
from prediction_model import *
from Scheduler import *
import os
import cv2
import numpy as np
import time

# Test results (FPS):

# Normal webcam: 50-100

# Haar face detection: 8-12
# Face + emotion detection with haar: 4-6
# Face + emotion detection (haar) with threaded timer optimization: 8-10 (kinda incorrect prediction, did not fix)

# Face detetction with DNN: 18-20
# Face + emotion detection with DNN: 7
# Face _ emotion detection (DNN) with threaded timer optimization: 15-19 (timer = 0.3s, calculated avg = 16 fps)
# Fixed the incorrect prediction issue, still the naive algorithm with haar cascade had better real-time predictions (as in the notebook)

global face_roi, status, fd_model, counter, prev_frame_time, new_frame_time, emotion_detect

face_roi = np.zeros((3, 3, 3))
status = 'neutral'
counter = 0
prev_frame_time = 0
new_frame_time = 0
emotion_detect = 0 # boolean

modelFile = "saved_model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "saved_model/deploy.prototxt.txt"
fd_model = cv2.dnn.readNetFromCaffe(configFile, modelFile)

images = 'images'
if os.path.isdir(images):
    for file in os.listdir(images):
        os.remove(os.path.join(images, file))
else:
    os.mkdir(images)

app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)

def predict_emotion(save_images = 0):
    global status, face_roi, counter, emotion_detect
    
    if not emotion_detect:
        return
    
    img_size = 224
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    if save_images:
        cv2.imwrite(os.path.join(images, 'face_' + str(counter) + '.png'), face_roi)
        counter += 1

    try:
        final_image = cv2.resize(face_roi, (img_size, img_size))
        final_image = np.expand_dims(final_image, axis = 0)
            
        Predictions = model6.predict(final_image)
        class_num = np.argmax(Predictions)   # **Provides the index of the max argument
        
        status = classes[class_num]
    except:
        pass
    
    print("Emotion: ", status)
    

def detect_face(frame):
    global fd_model, face_roi
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    fd_model.setInput(blob)
    detections = fd_model.forward()
    confidence = detections[0, 0, 0, 2] # atmost 1 face detected

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (x, y, x1, y1) = box.astype("int")
    try:
        # dim = (h, w)
        face_roi = frame[y:y1, x:x1]
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        raise
    
    return frame


rt = Scheduler(0.3, predict_emotion, 0) # Call predict every x seconds


# generator to yield frames from webcam
def gen_frames():  # generate frame by frame from camera
    global prev_frame_time, new_frame_time, emotion_detect, status, face_roi
    
    while True:
        success, frame = camera.read() 
        if success:
            # Calculating the fps
            new_frame_time = time.time()
            # try:
            #     fps = 1/(new_frame_time-prev_frame_time)
            #     prev_frame_time = new_frame_time
            #     fps = int(fps)
            #     print("FPS: ", fps)
            # except ZeroDivisionError as e:
            #     pass
                
            frame = cv2.flip(frame,1)
            frame = detect_face(frame)
            
            if(emotion_detect):      
                # pass          
                # predict_emotion()
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                x1, y1, w1, h1 = 0, 0, 175, 75
                
                cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
                cv2.putText(frame, status, (x1 + int(w1 / 10), y1 + int(h1 / 2)), font, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # HTTP format for images
            except Exception as e:
                pass
        else:
            pass

        
@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/requests',methods=['POST','GET'])
def user_input():
    if request.method == 'POST':
        if request.form.get('detect_emotion') == 'Detect Emotion On/Off':
            global emotion_detect
            emotion_detect =not emotion_detect
            
            # if emotion_detect:
            #     print("Now detecting emotions")
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     