#%%
#---------------------#
### IMPORT PACKAGES ###
#---------------------#
# Basic packages
import sys
!conda install --yes --prefix {sys.prefix} category_encoders

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data reading and converting
from data_processing import *

# scikit-learn, encoders
import sklearn
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from category_encoders import TargetEncoder

# Tensorflow + Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras

#%%
#------------------#
### PROCESS DATA ###
#------------------#
# Read in, transform data + get list of important features
data_files = ['hitdata7days_0.tar.gz', 'hitdata7days_1.tar.gz', 'hitdata7days_2.tar.gz', 'hitdata7days_3.tar.gz', 
              'hitdata7days_4.tar.gz', 'hitdata7days_5.tar.gz', 'hitdata7days_6.tar.gz']

df_0 = read_data(data_files[0])
df0_subset, imp_features, target_0 = feature_importance(df_0)
df0_x_train, df0_y_train = encode_data(df0_subset, imp_features, target_0)

#%%
df_1 = read_data(data_files[1])
df1_subset, dummy_vars, target_1 = feature_importance(df_1)
df1_x_train, df1_y_train = encode_data(df1_subset, imp_features, target_1)

df_2 = read_data(data_files[2])
df2_subset, dummy_vars, target_2 = feature_importance(df_2)
df2_x_train, df2_y_train = encode_data(df2_subset, imp_features, target_2)

df_3 = read_data(data_files[3])
df3_subset, dummy_vars, target_3 = feature_importance(df_3)
df3_x_train, df3_y_train = encode_data(df3_subset, imp_features, target_3)

df_4 = read_data(data_files[4])
df4_subset, dummy_vars, target_4 = feature_importance(df_4)
df4_x_train, df4_y_train = encode_data(df4_subset, imp_features, target_4)

df_5 = read_data(data_files[5])
df5_subset, dummy_vars, target_5 = feature_importance(df_5)
df5_x_train, df5_y_train = encode_data(df5_subset, imp_features, target_5)

df_6 = read_data(data_files[6])
df6_subset, dummy_vars, target_6 = feature_importance(df_6)
df6_x_test, df6_y_test = encode_data(df6_subset, imp_features, target_6)

#%%
x_train = np.concatenate((df0_x_train, df1_x_train, df2_x_train, df3_x_train, df4_x_train, df5_x_train), axis=0)
y_train = np.concatenate((df0_y_train, df1_y_train, df2_y_train, df3_y_train, df4_y_train, df5_y_train), axis=0)

x_train, y_train = shuffle(x_train, y_train, random_state=148)

#%%
x_test = df6_x_test
y_test = df6_y_test

x_test, y_test = shuffle(x_test, y_test, random_state=148)

#%%
#------------------------------#
### MODEL TRAINING + RESULTS ###
#------------------------------#
# Hyperparameters
num_epochs = 100 # 500 works well in practice
num_obs = 512 # batch size: larger values such as 512 generally work well in practice
adam_opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True) # Adam optimizer, 0.0025 learning rate generally works well
sgd_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) # Stochastic gradient method optimizer

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=20)) # Sigmoid for all layers works better than ReLU and Tanh
model.add(Dropout(0.1)) # Dropout regularization value, 0.1 generally works well
model.add(Dense(128, activation='sigmoid')) 
model.add(Dropout(0.25)) # Dropout regularization value, 0.1 generally works well
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

#%%
# Train neural network
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=num_obs) #, class_weight=class_weight

#%%
# Plot learning curves
sns.set()
auc_score = hist.history['auc']
val_auc = hist.history['val_auc']
epochs = range(1, len(auc_score) + 1)
plt.plot(epochs, auc_score, '-', label='Training AUC', alpha=1)
plt.plot(epochs, val_auc, '-', label='Validation AUC', alpha=0.75)
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend(loc='lower right')
plt.plot()
plt.show()

acc_score = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
plt.plot(epochs, acc_score, '-', label='Training Accuracy', alpha=1) 
plt.plot(epochs, val_acc, '-', label='Validation Accuracy', alpha=0.75)
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()

loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(epochs, loss, '-', label='Training Loss', alpha=1)
plt.plot(epochs, val_loss, '-', label='Validation Loss', alpha=0.75)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Confusion matrix
y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
classes = ['No Purchase', 'Purchase']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Counts)')
plt.show()

# Analysis of model inference
y_predicted = np.array(np.concatenate(model.predict(x_test)))
true_0_indices = np.where(y_test == 0)
true_1_indices = np.where(y_test == 1)
predict_0_indices = np.where(y_predicted < 0.5)[0]
predict_1_indices = np.where(y_predicted > 0.5)[0]

prob_0_true = y_predicted[np.intersect1d(true_0_indices, predict_0_indices)]
prob_0_false = y_predicted[np.intersect1d(true_0_indices, predict_1_indices)]
prob_1_true = y_predicted[np.intersect1d(true_1_indices, predict_1_indices)]
prob_1_false = y_predicted[np.intersect1d(true_1_indices, predict_0_indices)]

mean_prob = np.array([[np.mean(prob_0_true), np.mean(prob_0_false)], 
                     [np.mean(prob_1_false), np.mean(prob_1_true)]])

sd_prob = np.array([[np.std(prob_0_true), np.std(prob_0_false)], 
                    [np.std(prob_1_false), np.std(prob_1_true)]])

entries = (np.asarray([u"{0:.3f} \u00B1 {1:.3f} \n ({2:.0f})".format(value1, value2, value3)
                      for value1, value2, value3 in zip(mean_prob.flatten(),
                                                        sd_prob.flatten(),
                                                        mat.flatten())])
          ).reshape(2, 2)

sns.heatmap(mat, square=True, annot=entries, fmt='', cbar=True, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title(u'Test Set Probabilities (Mean \u00B1 S.D.) \n (Counts)')
plt.show()
#%%