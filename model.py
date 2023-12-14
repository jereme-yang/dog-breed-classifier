from preprocess import load_data
import numpy as np
from constants import *
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.nasnet import NASNetLarge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping   

(x_train, y_train), (x_test, y_test) = load_data()


my_model = NASNetLarge(input_shape=IMAGE_FULL_SIZE, weights='imagenet', include_top=False)

# don't train existing layers
for layer in my_model.layers:
    layer.trainable = False

flatten_layer = Flatten()(my_model.output)

# prediction layer should be 120 classes
prediction_layer = Dense(120, activation='softmax')(flatten_layer)

model = Model(inputs=my_model.input, outputs=prediction_layer)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 8
steps_per_epoch = np.ceil(len(x_train) / batch_size)
validation_steps = np.ceil(len(x_test) / batch_size)


best_model_file = "./tmp/dogs.h5"

# early stopping
callbacks = [
    ModelCheckpoint(best_model_file, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor='val_accuracy', patience=7, verbose=1)
]

model.fit(x_train, y_train_cat,
        validation_data = (x_test, y_test),
        epochs = 30,
        batch_size = batch_size,
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
        callbacks=[callbacks]
)


