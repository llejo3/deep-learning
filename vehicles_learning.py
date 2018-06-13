
# coding: utf-8

# In[ ]:


from google_images_download import google_images_download


# In[ ]:


response = google_images_download.googleimagesdownload() 


# In[ ]:


#arguments = {"keywords":"forsythia,cherry blossoms,magnolia,azalea,tulip",
arguments = {"keywords":"Planes, Bicycles, Cars",
             "limit":700,
             "print_urls":True,
             format: "jpg,png",
             "chromedriver" : "./chromedriver.exe",
             "exact_size":"200,200"}
paths = response.download(arguments) 


# In[ ]:


import zipfile
         
fantasy_zip = zipfile.ZipFile('./vehicles.zip')
fantasy_zip.extractall('./vehicles')
 
fantasy_zip.close()


# In[ ]:


get_ipython().system('pip install --upgrade tensorflow keras numpy pandas sklearn pillow')


# In[1]:


from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
 


# In[2]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,      # 40도까지 회전
        width_shift_range=0.2,  # 20%까지 좌우 이동
        height_shift_range=0.2, # 20%까지 상하 이동
        shear_range=0.2,        # 20%까지 기울임
        zoom_range=0.2,         # 20%까지 확대
        horizontal_flip=True   # 좌우 뒤집기
)


# In[3]:


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


# In[ ]:


model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(3, activation='softmax'))


# In[ ]:


model1.summary()


# In[ ]:


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)


# In[ ]:


history1 = model1.fit_generator(
    train, validation_data=valid, epochs=200,
    callbacks=[
        #EarlyStopping(monitor = "val_loss", patience=2),
        TensorBoard(log_dir='tensorboard_logs/log_model1')
    ])


# In[ ]:


model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(3, activation='softmax'))


# In[ ]:


model2.summary()


# In[ ]:


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)


# In[ ]:


history2 = model2.fit_generator(
    train, validation_data=valid, epochs=200,
    callbacks=[
        #EarlyStopping(monitor = "val_loss", patience=2),
        TensorBoard(log_dir='tensorboard_logs/log_model2')
    ])


# In[ ]:


model3 = Sequential()
model3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)))
model3.add(BatchNormalization())
model3.add(Conv2D(32, (3, 3), activation='relu'))
#model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(BatchNormalization())
model3.add(Conv2D(64, (3, 3), activation='relu'))
#model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.25))

model3.add(Flatten())
model3.add(Dense(256, activation='relu'))
#model3.add(BatchNormalization())
model3.add(Dropout(0.5))
model3.add(Dense(3, activation='softmax'))


# In[ ]:


model3.summary()


# In[ ]:


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model3.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)


# In[ ]:


history3 = model3.fit_generator(
    train, validation_data=valid, epochs=200,
    callbacks=[
        #EarlyStopping(monitor = "val_loss", patience=2),
        #ModelCheckpoint('model3-{epoch:02d}.hdf5', save_best_only=True),
        TensorBoard(log_dir='tensorboard_logs/log_model3')
    ])


# In[4]:


model4 = Sequential()
model4.add(Conv2D(32, (4, 4), activation='relu', input_shape=(200, 200, 3)))
model4.add(BatchNormalization())
model4.add(Conv2D(32, (6, 6), activation='relu'))
#model4.add(BatchNormalization())
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Conv2D(64, (5, 5), activation='relu'))
model4.add(BatchNormalization())
model4.add(Conv2D(64, (3, 3), activation='relu'))
#model4.add(BatchNormalization())
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))

model4.add(Flatten())
model4.add(Dense(256, activation='relu'))
#model4.add(BatchNormalization())
model4.add(Dropout(0.5))
model4.add(Dense(3, activation='softmax'))


# In[5]:


model4.summary()


# In[6]:


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model4.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)


# In[ ]:


history4 = model4.fit_generator(
    train, validation_data=valid, epochs=200,
    callbacks=[
        #EarlyStopping(monitor = "val_loss", patience=2),
        #ModelCheckpoint('model3-{epoch:02d}.hdf5', save_best_only=True),
        TensorBoard(log_dir='tensorboard_logs/log_model4')  
    ])

