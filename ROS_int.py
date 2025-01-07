import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

bridge = CvBridge()
model = None
label_encoder = None
target_object = None
object_detected = False  # Initialize object detection flag

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def image_callback(data):
    global bridge, model, label_encoder, object_detected
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    preprocessed_frame = preprocess_image(cv_image)
    bbox_pred, class_pred = model.predict(preprocessed_frame)
    predicted_class = np.argmax(class_pred, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    bbox = bbox_pred[0]

    if predicted_label.lower() == target_object.lower():
        height, width, _ = cv_image.shape
        x_min, y_min, bbox_width, bbox_height = bbox
        x_min = int(x_min * width)
        y_min = int(y_min * height)
        bbox_width = int(bbox_width * width)
        bbox_height = int(bbox_height * height)

        cv2.rectangle(cv_image, (x_min, y_min), (x_min + bbox_width, y_min + bbox_height), (255, 0, 0), 2)
        cv2.putText(cv_image, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        if not object_detected:
            rospy.loginfo("Object detected!")
            object_detected = True
            cv2.putText(cv_image, "Object Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    try:
        image_pub.publish(bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")

def main():
    global model, label_encoder, target_object
    rospy.init_node('object_detection_node', anonymous=True)

    model = load_model('bbox_67.keras')
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    target_object = input("Enter the object you want to detect: ").strip().lower()

    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
    global image_pub
    image_pub = rospy.Publisher("/object_detection/output_image", Image, queue_size=1)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down object detection node.")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()