import tensorflow as tf
import numpy as np
import cv2

def load_model():
    return tf.keras.models.load_model("./model/dogs.h5")

class SingletonClass(object):
  model = None
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SingletonClass, cls).__new__(cls)
    return cls.instance
  
def process_image(image):
    resized = cv2.resize(image, (331,331), interpolation=cv2.INTER_AREA)
    image_result = np.expand_dims(resized, axis=0)
    image_result = image_result / 255.
    return image_result

def find_label(label):
    ims = np.load('./tmp/cat_images.npy')
    labs = np.load('./tmp/cat_labels.npy')
    for i, lab in enumerate(labs):
        if label == lab:
            return ims[i]
    