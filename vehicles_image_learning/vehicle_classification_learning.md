

```python
from google_images_download import google_images_download
```


```python
response = google_images_download.googleimagesdownload() 
```


```python
#arguments = {"keywords":"forsythia,cherry blossoms,magnolia,azalea,tulip",
arguments = {"keywords":"Planes, Bicycles, Cars",
             "limit":600,
             "print_urls":True,
             format: "jpg,png",
             "chromedriver" : "./chromedriver.exe",
             "exact_size":"200,200"}
paths = response.download(arguments) 
```


```python
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
 
```


```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,      # 40도까지 회전
        width_shift_range=0.2,  # 20%까지 좌우 이동
        height_shift_range=0.2, # 20%까지 상하 이동
        shear_range=0.2,        # 20%까지 기울임
        zoom_range=0.2,         # 20%까지 확대
        horizontal_flip=True   # 좌우 뒤집기
)
```


```python
train = train_datagen.flow_from_directory(
    'vehicles/train',
    target_size=(200, 200),
    #batch_size=32,
    class_mode='categorical')

valid = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
    'vehicles/validation',
    target_size=(200, 200),
    #batch_size=32,
    class_mode='categorical',
    shuffle=False)
```

    Found 884 images belonging to 3 classes.
    Found 298 images belonging to 3 classes.
    


```python
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(3, activation='softmax'))
```


```python
model1.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_8 (Conv2D)            (None, 198, 198, 32)      896       
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 99, 99, 32)        0         
    _________________________________________________________________
    flatten_7 (Flatten)          (None, 313632)            0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 3)                 940899    
    =================================================================
    Total params: 941,795
    Trainable params: 941,795
    Non-trainable params: 0
    _________________________________________________________________
    


```python
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
```


```python
history1 = model1.fit_generator(
    train, validation_data=valid, epochs=30,
    callbacks=[
        EarlyStopping(monitor = "val_loss", patience=2),
        TensorBoard(log_dir='tensorboard_logs/log_model1')
    ])
```

    Epoch 1/30
     1/28 [>.............................] - ETA: 46s - loss: 1.1692 - acc: 0.3438

    C:\Users\south\Anaconda3\lib\site-packages\PIL\Image.py:918: UserWarning: Palette images with Transparency   expressed in bytes should be converted to RGBA images
      'to RGBA images')
    

    28/28 [==============================] - 36s 1s/step - loss: 3.7392 - acc: 0.3803 - val_loss: 1.0593 - val_acc: 0.4564
    Epoch 2/30
    28/28 [==============================] - 35s 1s/step - loss: 1.1686 - acc: 0.4723 - val_loss: 1.0099 - val_acc: 0.5839
    Epoch 3/30
    28/28 [==============================] - 36s 1s/step - loss: 0.9614 - acc: 0.5998 - val_loss: 0.9786 - val_acc: 0.5772
    Epoch 4/30
    28/28 [==============================] - 36s 1s/step - loss: 0.9075 - acc: 0.6209 - val_loss: 0.9295 - val_acc: 0.5805
    Epoch 5/30
    28/28 [==============================] - 36s 1s/step - loss: 0.8156 - acc: 0.6610 - val_loss: 0.8961 - val_acc: 0.6242
    Epoch 6/30
    28/28 [==============================] - 36s 1s/step - loss: 0.7874 - acc: 0.6738 - val_loss: 0.9063 - val_acc: 0.5973
    Epoch 7/30
    28/28 [==============================] - 37s 1s/step - loss: 0.7597 - acc: 0.6724 - val_loss: 0.8911 - val_acc: 0.6074
    Epoch 8/30
    28/28 [==============================] - 37s 1s/step - loss: 0.7685 - acc: 0.6638 - val_loss: 0.9621 - val_acc: 0.5772
    Epoch 9/30
    28/28 [==============================] - 34s 1s/step - loss: 0.7566 - acc: 0.6674 - val_loss: 0.8563 - val_acc: 0.6376
    Epoch 10/30
    28/28 [==============================] - 34s 1s/step - loss: 0.6990 - acc: 0.7087 - val_loss: 0.9243 - val_acc: 0.6074
    Epoch 11/30
    28/28 [==============================] - 35s 1s/step - loss: 0.7083 - acc: 0.7040 - val_loss: 0.9144 - val_acc: 0.6208
    


```python
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(3, activation='softmax'))
```


```python
model2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_11 (Conv2D)           (None, 198, 198, 32)      896       
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 99, 99, 32)        0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 97, 97, 32)        9248      
    _________________________________________________________________
    max_pooling2d_12 (MaxPooling (None, 48, 48, 32)        0         
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 73728)             0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 3)                 221187    
    =================================================================
    Total params: 231,331
    Trainable params: 231,331
    Non-trainable params: 0
    _________________________________________________________________
    


```python
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
```


```python
history2 = model2.fit_generator(
    train, validation_data=valid, epochs=100,
    callbacks=[
        EarlyStopping(monitor = "val_loss", patience=2),
        TensorBoard(log_dir='tensorboard_logs/log_model2')
    ])
```

    Epoch 1/30
     4/28 [===>..........................] - ETA: 44s - loss: 1.4224 - acc: 0.3391

    C:\Users\south\Anaconda3\lib\site-packages\PIL\Image.py:918: UserWarning: Palette images with Transparency   expressed in bytes should be converted to RGBA images
      'to RGBA images')
    

    28/28 [==============================] - 57s 2s/step - loss: 1.1169 - acc: 0.4312 - val_loss: 1.3070 - val_acc: 0.3658
    Epoch 2/30
    28/28 [==============================] - 53s 2s/step - loss: 1.0442 - acc: 0.4781 - val_loss: 1.0760 - val_acc: 0.4262
    Epoch 3/30
    28/28 [==============================] - 55s 2s/step - loss: 0.9853 - acc: 0.5179 - val_loss: 0.9727 - val_acc: 0.5503
    Epoch 4/30
    28/28 [==============================] - 55s 2s/step - loss: 0.9722 - acc: 0.5405 - val_loss: 1.0100 - val_acc: 0.6107
    Epoch 5/30
    28/28 [==============================] - 54s 2s/step - loss: 0.9162 - acc: 0.5844 - val_loss: 0.9376 - val_acc: 0.6040
    Epoch 6/30
    28/28 [==============================] - 59s 2s/step - loss: 0.7842 - acc: 0.6691 - val_loss: 1.0358 - val_acc: 0.6107
    Epoch 7/30
    28/28 [==============================] - 55s 2s/step - loss: 0.7229 - acc: 0.6908 - val_loss: 0.7254 - val_acc: 0.7148
    Epoch 8/30
    28/28 [==============================] - 55s 2s/step - loss: 0.7381 - acc: 0.6953 - val_loss: 0.8614 - val_acc: 0.6342
    Epoch 9/30
    28/28 [==============================] - 54s 2s/step - loss: 0.6778 - acc: 0.7174 - val_loss: 0.7262 - val_acc: 0.6980
    


```python
model3 = Sequential()
model3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model3.add(BatchNormalization())
model3.add(Conv2D(32, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(BatchNormalization())
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Flatten())
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(3, activation='softmax'))
```


```python
model3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_19 (Conv2D)           (None, 198, 198, 32)      896       
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 99, 99, 32)        0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 99, 99, 32)        0         
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 97, 97, 64)        18496     
    _________________________________________________________________
    max_pooling2d_17 (MaxPooling (None, 48, 48, 64)        0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 48, 48, 64)        0         
    _________________________________________________________________
    flatten_11 (Flatten)         (None, 147456)            0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 256)               37748992  
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 256)               0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 3)                 771       
    =================================================================
    Total params: 37,769,155
    Trainable params: 37,769,155
    Non-trainable params: 0
    _________________________________________________________________
    


```python
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)
```


```python
history3 = model3.fit_generator(
    train, validation_data=valid, epochs=30,
    callbacks=[
        EarlyStopping(monitor = "val_loss", patience=2),
        ModelCheckpoint('model3-{epoch:02d}.hdf5', save_best_only=True),
        TensorBoard(log_dir='tensorboard_logs/log_model3')
    ])
```

    Epoch 1/30
     1/28 [>.............................] - ETA: 1:34 - loss: 1.1490 - acc: 0.2500

    C:\Users\south\Anaconda3\lib\site-packages\PIL\Image.py:918: UserWarning: Palette images with Transparency   expressed in bytes should be converted to RGBA images
      'to RGBA images')
    

    28/28 [==============================] - 85s 3s/step - loss: 1.2178 - acc: 0.3502 - val_loss: 1.0796 - val_acc: 0.3893
    Epoch 2/30
    28/28 [==============================] - 86s 3s/step - loss: 1.0463 - acc: 0.4851 - val_loss: 1.0564 - val_acc: 0.4899
    Epoch 3/30
    28/28 [==============================] - 82s 3s/step - loss: 0.9830 - acc: 0.5198 - val_loss: 1.0233 - val_acc: 0.5839
    Epoch 4/30
    28/28 [==============================] - 86s 3s/step - loss: 0.8876 - acc: 0.6224 - val_loss: 0.9437 - val_acc: 0.5336
    Epoch 5/30
    28/28 [==============================] - 84s 3s/step - loss: 0.7544 - acc: 0.6600 - val_loss: 0.8263 - val_acc: 0.6577
    Epoch 6/30
    28/28 [==============================] - 86s 3s/step - loss: 0.7302 - acc: 0.7052 - val_loss: 0.7034 - val_acc: 0.6879
    Epoch 7/30
    28/28 [==============================] - 89s 3s/step - loss: 0.6840 - acc: 0.7004 - val_loss: 0.6810 - val_acc: 0.7047
    Epoch 8/30
    28/28 [==============================] - 89s 3s/step - loss: 0.6499 - acc: 0.7214 - val_loss: 0.6697 - val_acc: 0.7248
    Epoch 9/30
    28/28 [==============================] - 87s 3s/step - loss: 0.6163 - acc: 0.7403 - val_loss: 0.6561 - val_acc: 0.7383
    Epoch 10/30
    28/28 [==============================] - 107s 4s/step - loss: 0.6387 - acc: 0.7238 - val_loss: 0.6992 - val_acc: 0.6846
    Epoch 11/30
    28/28 [==============================] - 90s 3s/step - loss: 0.5976 - acc: 0.7457 - val_loss: 0.6568 - val_acc: 0.7114
    
