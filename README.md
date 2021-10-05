# Facial_Emotion_Recognition
Several models were built and meticulously tested on the [FER-2013 dataset](https://www.kaggle.com/msambare/fer2013). Of all those models, 5 models have been presented in the repository to compare and analyze their performances.
The images are resized to 128 X 128 or 224 X 224 for some models.

Code is provided in the Facial_Emotion_Recognition.ipynb notebook file. 

### Installations:

* Install all dependencies:

```shell
pip install -r requirements.txt
```  

* **Git LFS** - To clone and use this repository, you'll need [Git Large File Storage (LFS)](https://git-lfs.github.com/).

* Note that if you want to follow the notebook, you will have to make appropriate changes to the directory paths which should be trivial.


### Model Architecture and Weights:
Model weights are provided in the folders of respective models. Their architectures can be found in the notebook.

### Results:
* models 1 and 2 were trained locally on my PC with GPU (NVIDIA GeForce MX130). Due to hardware limitations, they couldn't be trained to their optimal state.
* Model3, built from scratch, is a decent model that has relatively fewer parameters (654,335). It was also trained locally and various hyperparameters were tuned so that its accuracy reaches its optimum.
An accuracy of nearly **50%** was achieved on the testing set. The following emotions can be easily detected with this model: happy, neutral, angry, and fear.
* The models, model_mobilenet and model_resnet_FineTuned_Large, could achieve **over 60%** accuracy on the testing set. These models were trained on a Kaggle notebook using GPU and their optimal weights were saved and downloaded.
The notebook can be found [here](https://www.kaggle.com/masterofsnippets/face-emotion-recognition)
* Due to hardware limitations, the Resnet50 based model couldn't be tested on a live webcam. It should have a performance comparable to that of the MobileNetV2 based model.
* The MobileNetV2 based model can easily detect most emotions on a live webcam. It also has the highest testing accuracy (**60.67%**) among all the models I tested.

*Further analysis is provided in the notebook.*

### Further Improvements:
* Although the resnet and mobilenet based models were trained on Kaggle GPU, due to RAM limitations, I wasn't able to augment the data. With more RAM, or perhaps, with a 
a more memory-efficient code, one can train the models on an augmented training set which could substantially improve the accuracy.
* I have also tuned the learning rate while testing the models but perhaps, more careful tuning of other hyperparameters may result in improved accuracy.
* For the resnet based model, one can try freezing fewer or more layers and see if it improves the accuracy.
* Changing the image size might help, but since the original images are too small, this would not make a huge difference.

*Following some or all of the above suggestions, I believe the accuracy of model_mobilenet or model_resnet_FineTuned_Large could easily go up to 65-70%*

### Credits:
* The code for the live webcam was inspired from [this article](https://medium.com/analytics-vidhya/realtime-face-emotion-recognition-using-transfer-learning-in-tensorflow-3add4f4f3ff3)
* For face detection, haarcascade_frontalface_default.xml was used.
