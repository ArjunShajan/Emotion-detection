import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,
    add, Dropout
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os

print("Loading dataset...")
data = pd.read_csv("fer2013.csv")

# Extract pixels and emotions
pixels = data['pixels'].tolist()
emotions = data['emotion'].values

faces = np.array([np.fromstring(p, dtype=int, sep=' ') for p in pixels])
faces = faces.reshape(-1, 48, 48, 1).astype("float32") / 255.0
emotions = to_categorical(emotions, num_classes=7)

# Split into train/validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    faces, emotions, test_size=0.2, stratify=emotions, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")

def mini_xception(input_shape, num_classes):
    input_img = Input(shape=input_shape)

    # First block
    x = Conv2D(8, (3,3), strides=(1,1), padding="same")(input_img)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second block
    x = Conv2D(8, (3,3), strides=(1,1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Residual blocks
    def residual_block(y, filters, strides=(1,1)):
        shortcut = Conv2D(filters, (1,1), strides=strides, padding="same")(y)
        shortcut = BatchNormalization()(shortcut)

        y = SeparableConv2D(filters, (3,3), padding="same", strides=strides)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        y = SeparableConv2D(filters, (3,3), padding="same")(y)
        y = BatchNormalization()(y)

        y = add([y, shortcut])
        y = Activation("relu")(y)
        return y

    x = residual_block(x, 16, strides=(2,2))
    x = residual_block(x, 32, strides=(2,2))
    x = residual_block(x, 64, strides=(2,2))
    x = residual_block(x, 128, strides=(2,2))

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = Model(input_img, output)
    return model

print("Building model...")
model = mini_xception((48, 48, 1), 7)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


checkpoint_path = "fer2013_mini_XCEPTION_best.hdf5"

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
]


print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)


final_path = "fer2013_mini_XCEPTION_final.hdf5"
model.save(final_path)
print(f"Training complete. Model saved at {final_path}")
