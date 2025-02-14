# %%
import numpy as np
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Precision, Recall, AUC, Accuracy
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
#from roboflow import Roboflow
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

# %%
annotation_path = 'Object_detection-1/train/_annotations.coco.json'
image_path = 'Object_detection-1/train/'

images = []
labels = []
bboxes = []

# %%
with open(annotation_path) as f:
    data = json.load(f)

# %%
for annotations in data['annotations']:
    image_id = annotations['image_id']
    image_info = next(item for item in data['images'] if item['id'] == image_id)
    image_name = image_info['file_name']
    img_path = os.path.join(image_path, image_name)
    
    if os.path.exists(img_path):
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.astype('float32') / 255

        bbox = annotations['bbox']
        label = annotations['category_id']

        images.append(image)
        labels.append(label)
        bboxes.append(bbox)

# %%
print(labels)

# %%
label_map = {0: 'Lab-images', 1: 'Battery', 2: 'Bottle', 3: 'Chair', 4: 'Door', 5: 'Door Knob', 6: 'Posters', 7: 'Socket', 8: 'Storage Box', 9: 'Table', 10: 'carboard box', 11: 'cupboard', 12: 'dustbin'}
labels = [label_map[label] for label in labels]

# %%
print(labels)

# %%
images = np.array(images)
bboxes = np.array(bboxes)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# %%
np.save('labels.npy', label_encoder.classes_)

# %%
print(labels)

# %%
labels = to_categorical(labels, num_classes=12)

# %%
print(labels)

# %%
X_train, X_temp, y_train, y_temp, bbox_train, bbox_temp = train_test_split(images, labels, bboxes, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test, bbox_val, bbox_test = train_test_split(X_temp, y_temp, bbox_temp, test_size=0.5, random_state=42)

# %%
input_shape = (224, 224, 3)
num_classes = len(label_encoder.classes_)
reg = l2(0.0005)

# %%
print(num_classes)

# %%
def create_model(input_shape, num_classes, reg):

    base_model = MobileNetV2(input_shape = input_shape, include_top = False, weights='imagenet')
    for layers in base_model.layers[:-10]:
        layers.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    bbox_out = Dense(4, activation = 'linear', name = 'bbox_output')(x)
    class_out = Dense(num_classes, activation = 'softmax', name = 'class_output')(x)

    model = Model(inputs = base_model.input, outputs = [bbox_out, class_out])

    model.compile(optimizer = Adam(learning_rate = 0.0001),
                  loss = {'bbox_output': 'mean_squared_error', 'class_output': 'categorical_crossentropy'},
                  metrics = {'bbox_output': 'mse', 'class_output': 'accuracy'})
    model.summary()

    return model
    

# %%
model = create_model(input_shape, num_classes, reg)

# %%
history = model.fit(X_train, 
                    {'bbox_output': bbox_train, 'class_output': y_train},
                    validation_data=(X_val, {'bbox_output': bbox_val, 'class_output': y_val}),
                    batch_size=20, 
                    epochs=150, 
                    verbose=1, 
                    callbacks=[ModelCheckpoint('bbox_.keras', save_best_only=True)])


# %%
predicitons = model.predict(X_test)
predicitons = np.argmax(predicitons[1], axis=1)
predicitons = label_encoder.inverse_transform(predicitons)

# %%
predictions = predicitons 
images = X_test
num_images = len(predictions)
num_cols = 5
num_rows = (num_images // num_cols) + 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(100, 100))  
axes = axes.flatten()
for i, (prediction, image) in enumerate(zip(predictions, images)):
    ax = axes[i]
    ax.imshow(image, aspect='equal')
    ax.set_title(prediction)
    ax.axis('off')

plt.show()

# %%
train_loss = history.history['loss']
epochs = range(1, len(train_loss) + 1)
val_loss = history.history['val_loss']
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%



