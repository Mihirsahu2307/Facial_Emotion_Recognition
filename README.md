# Facial_Emotion_Recognition

### Description:
Several models were built and they were trained on the [FER-2013 dataset](https://www.kaggle.com/msambare/fer2013) and their performances were compared. Further, these models were used to build a web application for real-time facial emotion recognition with optimized frame rate.

### Methodology:

**Preprocessing**: Images are resized to 128 X 128 or 224 X 224 for some models. Training, validation and testing sets are created.

**Training**: Models are trained on GPU and hyperparameters are tuned accordingly.

**Face detection**: Haarcascade frontal face classifier is used to detect faces in the image.

**Emotion Recognition**: Predictions are made on the detected faces.

## Installation

* Install all dependencies using requirements.txt like so:

```shell
pip install -r requirements.txt
```  

* **Git LFS** - To clone and use this repository, you'll need [Git Large File Storage (LFS)](https://git-lfs.github.com/).

* Note that if you want to follow the notebook, you will have to make appropriate changes to the directory paths which should be trivial.

### How to use:

* Use the following command to run the app.py file

```shell
python app.py
```  

* Open the http://127.0.0.1:5000/ on any browser

## Model Architecture and Weights
Model weights are provided in the folders of respective models. Their architectures can be found in the notebook.

## Results
* models 1 and 2 were trained locally on my PC with GPU (NVIDIA GeForce MX130). Due to hardware limitations, they couldn't be trained to their optimal state.
* Model3, built from scratch, is a decent model that has relatively fewer parameters (654,335). It was also trained locally and various hyperparameters were tuned so that its accuracy reaches its optimum.
An accuracy of nearly **50%** was achieved on the testing set. The following emotions can be easily detected with this model: happy, neutral, angry, and fear.
* The models, model_mobilenet and model_resnet_FineTuned_Large, could achieve **over 60%** accuracy on the testing set. These models were trained on a Kaggle notebook using GPU and their optimal weights were saved and downloaded.
The notebook can be found [here](https://www.kaggle.com/masterofsnippets/face-emotion-recognition)
* Due to hardware limitations, the Resnet50 based model couldn't be tested in real-time. It should have a performance comparable to that of the MobileNetV2 based model.
* The MobileNetV2 based model can easily detect most emotions in real-time. It also has the highest testing accuracy (**60.67%**) among all the models I tested.

*Further analysis is provided in the notebook.*

## Examples

<p float="left">
  <img src="https://github.com/Mihirsahu2307/Facial_Emotion_Recognition/blob/master/Examples/Happy_Predicted.jpg" width="500" />
  <img src="https://github.com/Mihirsahu2307/Facial_Emotion_Recognition/blob/master/Examples/Surprise_Predicted.jpg" width="500" /> 
  <img src="https://github.com/Mihirsahu2307/Facial_Emotion_Recognition/blob/master/Examples/Neutral_Predicted.jpg" width="500" />
</p>

## Further Improvements
* Due to RAM limitations of Kaggle, I wasn't able to augment the training data. With more RAM, or perhaps, with a more memory-efficient code, one can train the models on an augmented training set which could substantially improve the accuracy.
* I have also tuned the learning rate while testing the models but perhaps, more meticulous tuning of other hyperparameters may result in improved accuracy.
* For the resnet based model, one can try freezing fewer or more layers and see if it improves the accuracy.

*Following some or all of the above suggestions, I believe the accuracies of model_mobilenet and model_resnet_FineTuned_Large could easily go up to 65-70%*

## To Be Added

Easily accessible demo for real-time emotion recognition.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
* [This](https://github.com/oarriaga/face_classification) repository provided helpful resources for this project.
