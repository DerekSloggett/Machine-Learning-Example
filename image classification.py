import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_resnet(input_shape, num_classes, pool2=None):
    # Input
    inputs = Input(shape=input_shape)

    # Convolutional layers
    conv1 = Conv2D(64, 7, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(act1)

    conv2_x = tf.keras.layers.concatenate([pool1, pool1])
    conv2_x = Conv2D(64, 3, padding='same')(conv2_x)
    conv2_x = BatchNormalization()(conv2_x)
    conv2_x = Activation('relu')(conv2_x)
    conv2_x = Conv2D(64, 3, padding='same')(conv2_x)
    conv2_x = BatchNormalization()(conv2_x)

    conv2_x_shortcut = Conv2D(64, 1, padding='same')(pool1)
    conv2_x_shortcut = BatchNormalization()(conv2_x_shortcut)

    conv2_x = tf.keras.layers.add([conv2_x, conv2_x_shortcut])
    conv2_x = Activation('relu')(conv2_x)

    conv3_x = Conv2D(128, 3, strides=(2, 2), padding='same')(conv2_x)
    conv3_x = BatchNormalization()(conv3_x)
    conv3_x = Activation('relu')(conv3_x)
    conv3_x = Conv2D(128, 3, padding='same')(conv3_x)
    conv3_x = BatchNormalization()(conv3_x)

    conv3_x_shortcut = Conv2D(128, 1, strides=(2, 2), padding='same')(conv2_x)
    conv3_x_shortcut = BatchNormalization()(conv3_x_shortcut)

    conv3_x = tf.keras.layers.add([conv3_x, conv3_x_shortcut])
    conv3_x = Activation('relu')(conv3_x)

    conv4_x = Conv2D(256, 3, strides=(2, 2), padding='same')(conv3_x)
    conv4_x = BatchNormalization()(conv4_x)
    conv4_x = Activation('relu')(conv4_x)
    conv4_x = Conv2D(256, 3, padding='same')(conv4_x)
    conv4_x = BatchNormalization()(conv4_x)

    conv4_x_shortcut = Conv2D(256, 1, strides=(2, 2), padding='same')(conv3_x)
    conv4_x_shortcut = BatchNormalization()(conv4_x_shortcut)

    conv4_x = tf.keras.layers.add([conv4_x, conv4_x_shortcut])
    conv4_x = Activation('relu')(conv4_x)

    conv5_x = Conv2D(512, 3, strides=(2, 2), padding='same')(conv4_x)
    conv5_x = BatchNormalization()(conv5_x)
    conv5_x = Activation('relu')(conv5_x
    conv5_x = Conv2D(512, 3, padding='same')(conv5_x)
    conv5_x = BatchNormalization()(conv5_x)

    conv5_x_shortcut = Conv2D(512, 1, strides=(2, 2), padding='same')(conv4_x)
    conv5_x_shortcut = BatchNormalization()(conv5_x_shortcut)

    conv5_x = tf.keras.layers.add([conv5_x, conv5_x_shortcut])
    conv5_x = Activation('relu')(conv5_x)

    # Global average pooling
    pool2 = GlobalAveragePooling2D()(conv5_x)

    # Output
    outputs = Dense(num_classes, activation='softmax')(pool2)

    # Model
    model = Model(inputs=inputs, outputs=outputs)

    return model

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
'/path/to/train/directory',
image_size=(224, 224),
batch_size=32,
validation_split=0.2,
subset='training',
seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
'/path/to/train/directory',
image_size=(224, 224),
batch_size=32,
validation_split=0.2,
subset='validation',
seed=123
)

model = build_resnet(input_shape=(224, 224, 3), num_classes=10)

model.compile(
optimizer=Adam(lr=1e-4),
loss='categorical_crossentropy',
metrics=['accuracy']
)

model.fit(
train_ds,
validation_data=val_ds,
epochs=50
)