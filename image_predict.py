import os
import cv2
import numpy as np
from mobilenet_model import *

# Single image prediction (debugging tool)
# Used to predict on images stored in 'images' directory 

def predict_emotion(face_roi, path):
    img_size = 224
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    try:
        final_image = cv2.resize(face_roi, (img_size, img_size))
        final_image = np.expand_dims(final_image, axis = 0)
            
        Predictions = model6.predict(final_image)
        class_num = np.argmax(Predictions)   # **Provides the index of the max argument
        
        status = classes[class_num]
    except:
        pass
    
    cv2.imwrite(path + status + '.png', face_roi)
    
    
images = 'images'
for file in os.listdir(images):
    face_roi = cv2.imread(os.path.join(images, file))
    path = os.path.join(images, file)
    path = os.path.splitext(path)[0]
    predict_emotion(face_roi, path)