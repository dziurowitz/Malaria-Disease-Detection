# %%
# import libs
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D

# %%
# load data
data_dir = os.path.join(os.getcwd(), 'cell_images/images')

malaria_dir = os.path.join(data_dir,'Parasitized')
healthy_dir = os.path.join(data_dir,'Uninfected')

malaria_files = glob.glob(malaria_dir+'/*.png')
healthy_files = glob.glob(healthy_dir+'/*.png')
len(malaria_files), len(healthy_files)

data = pd.DataFrame({
    'filename': malaria_files + healthy_files,
    'class': ['malaria'] * len(malaria_files) + ['healthy'] * len(healthy_files)
}).sample(frac=1).reset_index(drop=True)

# %%
# CNN build
model = Sequential()

# convolutional layer
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# pooling layer with dropout
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

# convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))

# pooling layer with dropout
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

# flattening
model.add(Flatten())

# fully connected layer
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# %%
# Image processing
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.2)


train_generator = train_datagen.flow_from_dataframe(data,
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 subset='training')

val_generator = train_datagen.flow_from_dataframe(data,
                                            target_size = (64, 64),
                                            batch_size = 64,
                                            class_mode = 'binary',
                                            subset='validation')

train_steps = train_generator.samples // train_generator.batch_size
val_steps =  train_generator.samples  // val_generator.batch_size
#%%
# plot 5 images from image generator
x_batch, y_batch = next(train_generator)
for i in range (0,5):
    image = x_batch[i]
    plt.imshow(image)
    plt.show()
# %%
# Model fitting
history = model.fit_generator(train_generator,
                                steps_per_epoch = train_steps,
                                epochs = 10,
                                validation_data = val_generator,
                                validation_steps = val_steps)


# %%
