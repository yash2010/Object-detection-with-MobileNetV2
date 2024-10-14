# Object Detection and ROS integration

## Introduction ℹ️
This repository contains two main components: a deep learning-based object detection pipeline and a ROS node integration for real-time object detection. The object detection model predicts both bounding boxes and object classes, while the ROS node captures images from a camera feed and uses the model to detect and classify objects in real-time.

The files 📂: 

**[Object_detection-1](/Object_detection-1)** contains the datasets of the test environment. These images were used to train the MobileNetV2 model.

**[trained_model(.keras files)](/trained_model (.keras files))** containes the pretrained models to use in the ROS integration program.


## Features ⚙️
### 1. Object Detection Pipeline: 📂Filename: [Object_detection](Object_detection)
+ Uses MobileNetV2 as the base model for feature extraction.
+ Trains the model to detect objects and predict bounding boxes using a custom dataset.
+ Supports multiple object classes like batteries, doors, tables, etc.
+ Visualization of predictions with bounding boxes and object labels.
### 2. ROS Integration: 📂Filename: [ROS_int](ROS_int)
+ Real-time object detection from a ROS topic (/camera/rgb/image_raw).
+ The detected object is highlighted with a bounding box and label.
+ Publishes output images to a ROS topic (/object_detection/output_image).
+ Allows you to specify a target object to detect through user input.

## Object Detection Pipeline 💻
### Dependencies
To run the object detection pipeline, ensure you have the following dependencies installed:
+ Python 3.x
+ TensorFlow 2.x
+ OpenCV
+ Numpy
+ Matplotlib
+ Scikit-learn
+ Scikit-image

### Steps
**1. Preprocessing:**
+ The COCO-style annotations are parsed to extract images, bounding boxes, and labels.
+ The images are resized to 224x224 for input into the model, and the labels are converted to one-hot encoded format.

**2. Model Architecture:**
+ MobileNetV2 is used as a backbone for feature extraction.
+ A global average pooling layer is applied to the extracted features, followed by fully connected layers.
+ The model has two outputs:
    + Bounding box coordinates (4 values).
    + Object class (categorical prediction).

**3. Training:**
+ The model is trained using a combination of mean squared error (for bounding boxes) and categorical crossentropy (for object class) losses.
+ The dataset is split into training, validation, and test sets.
+ The model checkpoints are saved at the end of each epoch.

**4. Evaluation and Visualization:**
+ The model's performance is evaluated on the test set, and predicted bounding boxes and classes are visualized.
![output](https://github.com/user-attachments/assets/e75a5a13-2832-4c9d-b619-16b5dc58d312)
  
## ROS Integration 🤖
### Dependencies
Make sure to have the following ROS-related dependencies installed:
+ ROS (tested on ROS Noetic)
+ OpenCV
+ cv_bridge
+ sensor_msgs

### How it Works
**1. Image Subscriber:**  Subscribes to a camera topic (/camera/rgb/image_raw) to receive real-time images.

**2. Model Prediction:** The received image is preprocessed and passed through the object detection model to get the bounding box and class predictions.

**3. Result Publishing:** The result is visualized with a bounding box and class label and published to a new ROS topic (/object_detection/output_image).

### Configuring the Object Detection Model 🔨
+ The model (bbox_67.keras) and label encoder (classes.npy) should be placed in the same directory as the ROS node script.
+ Modify the ROS parameters if necessary, such as the camera topic or target object.

### Conclusion ✅
This repository provides a complete pipeline for object detection, from training the model to integrating it with ROS for real-time deployment. You can further customize the model, ROS nodes, and dataset to suit your specific requirements.

## Note🧾
This is an ongoing project, and contributions or improvements are highly encouraged. Feel free to modify the code, experiment with different architectures, or improve the ROS integration to suit your specific needs. Contributions via pull requests are welcome!






