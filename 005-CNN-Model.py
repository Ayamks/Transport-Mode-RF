# Mount The Google Drive
from google.colab import drive
drive.mount('/content/gdrive')

# Import Libraries
import os
import glob
import numpy as np
import pandas as pd
import time
import datetime
import pickle
import random
import math
from collections import Counter
import statistics
from statistics import mode
pd.options.mode.chained_assignment = None
print("Libraries Are Imported")

# Define Path of Data
path = '/content/gdrive/MyDrive/Colab Notebooks'
os.chdir(path)

# Read The Labeled Data
DF = '004_Preprocessed_Data.pickle'
infile = open(DF,'rb')
df = pickle.load(infile)
infile.close()

# Data for User 104
df_filtered = df[df['User'] == 101]

# Now df_filtered contains only rows where User is 101
print(df_filtered)

# Replace Names with Characters
df.replace('bike', 'p', inplace=True)
df.replace('bus', 'b', inplace=True)
df.replace('car', 'c', inplace=True)
df.replace('train', 't', inplace=True)
df.replace('walk', 'w', inplace=True)

# Split All Data to Train and Test Set Randomly
User_List = df['User'].unique().tolist()
!pip install mpu

import mpu
random.seed(29)
list_one = User_List

list_one, list_two = mpu.consistent_shuffle(list_one,list_one)
train_user = list_one[:45]
test_user = list_one[45:]

Train_Data = df[df['User'].isin(train_user)]
Test_Data = df[df['User'].isin(test_user)]

Train_data = []
for i, g in Train_Data.groupby(['Trip','User']):
  Train_data.append(g)

Test_data = []
for l, m in Test_Data.groupby(['Trip','User']):
  Test_data.append(m)

# Define Extra Columns and Drop them
extra_columns = ['Latitude', 'Longitude', 'Altitude', 'Date', 'User', 'TS', 'Trip', 'Mode', 'Distance', 'DT', 'Bearing', 'Cum_Distance']

Train_Y_t = []
for df in Train_data:
    Train_Y_t.append(df['Mode'].values)
    df.drop(columns=extra_columns, inplace=True)

Test_Y_t = []
for df in Test_data:
    Test_Y_t.append(df['Mode'].values)
    df.drop(columns=extra_columns, inplace=True)

# Normalize Both Train and Test Datasets
Train_Normalize = []
for df in Train_data:
    Train_Normalize.append(df.values)

Test_Normalize = []
for df in Test_data:
    Test_Normalize.append(df.values)

# Define Maximum and Minimum Trip Sizes
max_trip_size = 200
min_trip_size = 60

# Break the Trips to Windows - Add Padding to Windows
def break_trip(trip, trip_Y, max_trip_size):
    length = max_trip_size
    jump = length
    split = [trip[i:i+length] for i in range(0,len(trip),jump)][:-1]+[trip[-length:]]
    split_Y = [trip_Y[i:i+length] for i in range(0,len(trip_Y),jump)][:-1]+[trip_Y[-length:]]
    return split, split_Y

def padd_trip(trip, trip_Y, max_trip_size):
    trip_padded = np.pad(trip, ((0, max_trip_size-trip.shape[0]), (0, 0)), 'constant')
    trip_padded_Y = np.pad(trip_Y, (0, max_trip_size-trip.shape[0]), 'constant', constant_values=(0,0))
    return trip_padded, trip_padded_Y

Train_X = []
Train_Y = []
for i, trip in enumerate(Train_Normalize):
    size_trip = trip.shape[0]
    if  size_trip <= min_trip_size:
        continue

    if size_trip > max_trip_size:
        trip_breaks, trip_breaks_Y = break_trip(trip, Train_Y_t[i], max_trip_size)
        Train_X.extend(trip_breaks)
        Train_Y.extend(trip_breaks_Y)

    if size_trip <= max_trip_size and size_trip > min_trip_size:
        trip_pad, trip_pad_Y = padd_trip(trip, Train_Y_t[i], max_trip_size)
        Train_X.append(trip_pad)
        Train_Y.append(Counter(Train_Y_t[i].flat).most_common(1)[0][0])

Test_X = []
Test_Y = []
for i, trip in enumerate(Test_Normalize):
    size_trip = trip.shape[0]
    if  size_trip <= min_trip_size:
        continue

    if size_trip > max_trip_size:
        trip_breaks, trip_breaks_Y = break_trip(trip, Test_Y_t[i], max_trip_size)
        Test_X.extend(trip_breaks)
        Test_Y.extend(trip_breaks_Y)

    if size_trip <= max_trip_size and size_trip > min_trip_size:
        trip_pad, trip_pad_Y = padd_trip(trip, Test_Y_t[i], max_trip_size)
        Test_X.append(trip_pad)
        Test_Y.append(Counter(Test_Y_t[i].flat).most_common(1)[0][0])

# Find the mode of each Single Window
Train_M = []
for i in Train_Y:
  if len(i) == 1:
    Train_M.append(i)
  else:
    lst = i.tolist()
    Train_M.append(max(set(lst), key=lst.count))

Test_M = []
for i in Test_Y:
  if len(i) == 1:
    Test_M.append(i)
  else:
    lst = i.tolist()
    Test_M.append(max(set(lst), key=lst.count))

train_Y = pd.DataFrame(Train_M,columns=['LL'])
test_Y = pd.DataFrame(Test_M,columns=['LL'])

# Replace Characters with Names in Train Dataset
train_Y.replace('p', 'bike', inplace=True)
train_Y.replace('b', 'bus', inplace=True)
train_Y.replace('c', 'car', inplace=True)
train_Y.replace('t', 'train', inplace=True)
train_Y.replace('w', 'walk', inplace=True)

# Replace Characters with Names in Test Dataset
test_Y.replace('p', 'bike', inplace=True)
test_Y.replace('b', 'bus', inplace=True)
test_Y.replace('c', 'car', inplace=True)
test_Y.replace('t', 'train', inplace=True)
test_Y.replace('w', 'walk', inplace=True)

# Save all Data in a Dictionary
data_dict = dict()
data_dict['Train_X'] = Train_X
data_dict['Train_Y'] = train_Y
data_dict['Test_X'] = Test_X
data_dict['Test_Y'] = test_Y

# Define Train_X, Test_X. train_Y, and test_Y
Train_X = np.asarray(data_dict['Train_X']).astype('float32')
train_Y = np.asarray(data_dict['Train_Y'])
Test_X = np.asarray(data_dict['Test_X']).astype('float32')
test_Y = np.asarray(data_dict['Test_Y'])

# Apply One Hot Encoding To Train_Y and Test_Y
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
Train_Y = enc.fit_transform(train_Y.reshape(-1, 1)).toarray()
print(Train_X.shape)
print(Train_Y.shape)
Test_Y = enc.fit_transform(test_Y.reshape(-1, 1)).toarray()
print(Test_X.shape)
print(Test_Y.shape)

# Define Number of Features and Number of Classes
num_features = Train_X.shape[-1]
print(num_features)
Modes = enc.categories_
NoClass = len(Modes[0])
print(NoClass)

# Import Deep Learning Libraries
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, BatchNormalization, Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import regularizers
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy.stats import zscore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = print(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)))

start_time = time.perf_counter()
np.random.seed(7)
random.seed(7)

trip_size = Train_X.shape[-2]
trip_size

# Define Kernel Size, Maxpooling Size, Stride
kernel = 16
pool = 4
stride = 1
Drop_Out = 0.5

# Structure of Model and Compile
model = Sequential()
model.add(Conv1D(64, kernel, strides=stride, padding='same', dilation_rate = 1, input_shape=(trip_size, num_features)))
model.add(Activation("relu"))
model.add(Conv1D(64, kernel, strides=stride, padding='same', dilation_rate = 1))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=pool))

model.add(Dropout(Drop_Out))

model.add(Conv1D(128, kernel, strides=stride, padding='same', dilation_rate = 1))
model.add(Activation("relu"))
model.add(Conv1D(128, kernel, strides=stride, padding='same', dilation_rate = 1))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=pool))

model.add(Dropout(Drop_Out))

model.add(Conv1D(256, kernel, strides=stride, padding='same', dilation_rate = 1))
model.add(Activation("relu"))
model.add(Conv1D(256, kernel, strides=stride, padding='same', dilation_rate = 1))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling1D(pool_size=pool))

model.add(Flatten())

model.add(Dropout(Drop_Out))

model.add(Dense(2048))
model.add(Activation("relu"))

model.add(Dropout(Drop_Out))

model.add(Dense(1024))
model.add(Activation("relu"))

model.add(Dense(NoClass, activation='softmax'))

EPOCHS = 100
def scheduler(epoch, lr):
    if epoch % EPOCHS == 0 and epoch != 0:
        print("[INFO] lr is  ... ", lr/10)
        return lr/10
    else:
        return lr

# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optimizer = RMSprop(lr=0.001)
# optimizer = SGD(lr=0.0001, momentum=0.9, decay=1e-4, nesterov=True)

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
hist = model.fit(Train_X, Train_Y, epochs=10, batch_size=64, validation_data=(Test_X, Test_Y), callbacks=[callback])

# Model Summary
model.summary()

# Prediction of Model
pred_Y = model.predict(Test_X)
pred_Y_N = np.argmax(pred_Y, axis=1)
Pred_Y = enc.fit_transform(pred_Y_N.reshape(-1, 1)).toarray()

# Apply One Hot Encoding Transform
print("Shape of Test_Y:", Test_Y.shape)
Pred_Y_N = enc.inverse_transform(Pred_Y)
Test_Y_N = enc.inverse_transform(Test_Y)

print(confusion_matrix(Test_Y_N, Pred_Y_N))
print(classification_report(Test_Y_N, Pred_Y_N))
print(accuracy_score(Test_Y_N, Pred_Y_N))

# Plt Confusion Matrix
LABELS = ['bike','bus','car','train','walk']
cm = confusion_matrix(Test_Y_N, Pred_Y_N)
bg_color = (0.88,0.85,0.95)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(15,10))
sn.heatmap(cm,xticklabels=LABELS, yticklabels=LABELS,annot=True, fmt="d", cmap='jet', annot_kws={'size':15})
plt.title('Confusion Matrix', fontsize = 20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Predicted', fontsize = 16)
plt.ylabel('True', fontsize = 16)
