# %%
import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import Sequential, Model
from keras.utils import image_dataset_from_directory, plot_model
from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# %% 1. Data loading
DATASET_PATH = os.path.join(os.getcwd(), 'dataset', 'Concrete Crack Images for Classification')
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
SEED = 12345

train_ds = image_dataset_from_directory(DATASET_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, 
                                        seed=SEED, validation_split=0.2, subset='training')

val_ds = image_dataset_from_directory(DATASET_PATH, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, 
                                      seed=SEED, validation_split=0.2, subset='validation')

# %% 2. Data inspection
# Visualize a few image from the BatchDataset to make sure images are loaded properly
CLASS_NAME = train_ds.class_names

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype(np.uint8))
        plt.title(CLASS_NAME[labels[i]])
        plt.axis(False)
plt.show()

# %% 3. Data preparation
# Peform validation test split
batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(batches // 5)
val_ds = val_ds.skip(batches // 5)

# Convert train, validation and test dataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

train_pf = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_pf = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_pf = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define image augmentation layer
image_augmentation =  Sequential([
    RandomFlip(seed=SEED),
    RandomRotation(factor=0.2, seed=SEED),
    RandomZoom(0.2, seed=SEED)
])

# Visualize image augmentation layer
for images, labels in train_ds.take(1):
    image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = image_augmentation(tf.expand_dims(image, axis=0))
        plt.imshow(augmented_image[0].numpy().astype('uint8'))
        plt.axis(False)
plt.show()

# %% Model development
INPUT_SHAPE = list(IMAGE_SIZE) + [3,]

# Define normalization layer
normalization_layer = preprocess_input

# Define base model
base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)
base_model.trainable = False

# Build a model with functional method
inputs = Input(shape=INPUT_SHAPE)
x = image_augmentation(inputs)
x = normalization_layer(x)
x = base_model(x, training=False)
# Classifier for base model
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(len(CLASS_NAME), activation='softmax')(x)

# Instantiate the model
model = Model(inputs=inputs, outputs=outputs)

# Model summary
model.summary()
plot_model(model, to_file=os.path.join(os.getcwd(), 'resources', 'model_architecture.png'), show_layer_names=True, show_shapes=True)

# Model compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='acc')

# Define callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tb = TensorBoard(log_dir=LOG_DIR)
es = EarlyStopping(patience=5, monitor='val_acc', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.2, monitor='val_acc', patience=5, min_lr=0.0001)
mc = ModelCheckpoint(filepath=os.path.join(os.getcwd(), 'temp', 'checkpoint'), monitor='val_acc', save_best_only=True)

# Model training
EPOCHS = 10

history = model.fit(train_pf, validation_data=val_pf, epochs=EPOCHS, callbacks=[tb, es, reduce_lr, mc])

# %% Model evaluation
# Doing prediction with model
image_batch, label_batch = test_pf.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch), axis=1)

# Classification report
print('Classifcation report:\n', classification_report(label_batch, y_pred))

# %% Model saving
# Save model
model.save(filepath=os.path.join(os.getcwd(), 'saved_model', 'model.h5'))

# %%
