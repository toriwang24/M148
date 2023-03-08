#%%
#---------------------#
### IMPORT PACKAGES ###
#---------------------#
import pandas as pd
import numpy as np
from data_processing import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#%%
#---------------------#
### DATA CONVERSION ###
#---------------------#
features = ['visitoridlow', 'checkoutthankyouflag', 'myaccountengagement', 'addonsymal', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
prop29 = ['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash']
data_files = ['hitdata7days_0.tar.gz', 'hitdata7days_1.tar.gz', 'hitdata7days_2.tar.gz', 'hitdata7days_3.tar.gz', 
              'hitdata7days_4.tar.gz', 'hitdata7days_5.tar.gz', 'hitdata7days_6.tar.gz']

df_subset_1_day0, df_subset_2_day0 = process_data(data_files[0])
df_subset_1_day1, df_subset_2_day1 = process_data(data_files[1])
df_subset_1_day2, df_subset_2_day2 = process_data(data_files[2])
df_subset_1_day3, df_subset_2_day3 = process_data(data_files[3])
df_subset_1_day4, df_subset_2_day4 = process_data(data_files[4])
df_subset_1_day5, df_subset_2_day5 = process_data(data_files[5])
df_subset_1_day6, df_subset_2_day6 = process_data(data_files[6])

#%%
#-------------------#
### DATA SAMPLING ###
#-------------------#
train_samp = 4000
test_samp = 6000

df_subset_day0 = pd.concat([df_subset_1_day0.sample(n=train_samp), df_subset_2_day0.sample(n=train_samp)])
df_subset_day0 = df_subset_day0.sample(frac = 1) #, random_state=148

df_subset_day1 = pd.concat([df_subset_1_day1.sample(n=train_samp), df_subset_2_day1.sample(n=train_samp)])
df_subset_day1 = df_subset_day1.sample(frac = 1) #, random_state=148

df_subset_day2 = pd.concat([df_subset_1_day2.sample(n=train_samp), df_subset_2_day2.sample(n=train_samp)])
df_subset_day2 = df_subset_day2.sample(frac = 1) #, random_state=148

df_subset_day3 = pd.concat([df_subset_1_day3.sample(n=train_samp), df_subset_2_day3.sample(n=train_samp)])
df_subset_day3 = df_subset_day3.sample(frac = 1) #, random_state=148

df_subset_day4 = pd.concat([df_subset_1_day4.sample(n=train_samp), df_subset_2_day4.sample(n=train_samp)])
df_subset_day4 = df_subset_day4.sample(frac = 1) #, random_state=148

df_subset_day5 = pd.concat([df_subset_1_day5.sample(n=train_samp), df_subset_2_day5.sample(n=train_samp)])
df_subset_day5 = df_subset_day5.sample(frac = 1) #, random_state=148

df_subset_day6 = pd.concat([df_subset_1_day6.sample(n=test_samp), df_subset_2_day6.sample(n=test_samp)])
df_subset_day6 = df_subset_day6.sample(frac = 1) #, random_state=148

#%%
#-------------------------------#
### MODEL BUILDING + TRAINING ###
#-------------------------------#
#----------------------#
### Train-test split ###
#----------------------#
df_train = pd.concat([df_subset_day0, df_subset_day1, df_subset_day2, df_subset_day3, df_subset_day4, df_subset_day5])
df_train = df_train.sample(frac = 1)
df_test = df_subset_day6.sample(frac = 1) # Test set is day 6 subsetted data

x_train = df_train.drop(['visitoridlow', 'checkoutthankyouflag'], axis=1)
x_train = np.asarray(x_train).astype('float32')
y_train = df_train['checkoutthankyouflag'] 
y_train = np.asarray(y_train).astype('float32')

x_test = df_test.drop(['visitoridlow', 'checkoutthankyouflag'], axis=1)
x_test = np.asarray(x_test).astype('float32')
y_test = df_test['checkoutthankyouflag']
y_test = np.asarray(y_test).astype('float32')

#%%
#---------------------#
### Hyperparameters ###
#---------------------#
num_epochs = 500 # 500 works well in practice
num_obs = 512 # batch size: larger values such as 512 generally work well in practice
adam_opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True) # Adam optimizer, 0.0025 learning rate generally works well
sgd_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) # Stochastic gradient method optimizer

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=len(features) - 2)) # Sigmoid for all layers works better than ReLU and Tanh
model.add(Dropout(0.01)) # Dropout regularization value, 0.1 generally works well
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.01)) # Dropout regularization value, 0.1 generally works well
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

#%%
#--------------------------#
### Train neural network ###
#--------------------------#
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=num_obs)

#%%
#-----------------------------------------#
### VISUALIZATION + ANALYSIS OF RESULTS ###
#-----------------------------------------#
#--------------------------#
### Plot learning curves ###
#--------------------------#
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
plt.legend(loc='lower right')
plt.show()

#----------------------#
### Confusion matrix ###
#----------------------#
y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
classes = ['0', '1']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix (Counts)')
plt.show()

#%%
#---------------------------------#
### Analysis of model inference ###
#---------------------------------#
y_predicted = np.array(np.concatenate(model.predict(x_test)))
print(mat)
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

entries = (np.asarray([u"{0:.3f} \u00B1 {1:.3f}".format(string, value)
                      for string, value in zip(mean_prob.flatten(),
                                               sd_prob.flatten())])
         ).reshape(2, 2)

sns.heatmap(mat, square=True, annot=entries, fmt='', cbar=True, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title(u'Test Set Probabilities (Mean \u00B1 S.D.)')
plt.show()
# %%