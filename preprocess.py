import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data():
    all_images = np.load("./tmp/all_images.npy")
    all_labels = np.load("./tmp/all_labels.npy")

    # convert labels text to integers
    le = LabelEncoder()
    integer_labels = le.fit_transform(all_labels)

    num_categories = len(np.unique(integer_labels))

    # convert the integer labels to categorical -> prepare for train
    all_labels_for_model = to_categorical(integer_labels, num_classes=num_categories)

    # normalize the images
    all_images_for_model = all_images / 255.

    #create train and test data
    x_train, x_test, y_train, y_test = train_test_split(all_images_for_model, all_labels_for_model, test_size=0.3, random_state = 0)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
