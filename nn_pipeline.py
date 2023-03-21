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
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from category_encoders import TargetEncoder
from category_encoders import BinaryEncoder
from category_encoders import WOEEncoder
from category_encoders import LeaveOneOutEncoder

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

df0 = read_data(data_files[0])
df0_subset, imp_features, target_0 = feature_importance(df0)

df1 = read_data(data_files[1])
df1_subset, dummy_vars, target_1 = feature_importance(df1)

df2 = read_data(data_files[2])
df2_subset, dummy_vars, target_2 = feature_importance(df2)

df3 = read_data(data_files[3])
df3_subset, dummy_vars, target_3 = feature_importance(df3)

df4 = read_data(data_files[4])
df4_subset, dummy_vars, target_4 = feature_importance(df4)

df5 = read_data(data_files[5])
df5_subset, dummy_vars, target_5 = feature_importance(df5)

df6 = read_data(data_files[6])
df6_subset, dummy_vars, target_6 = feature_importance(df6)

#%%
top_features = imp_features
top_features = top_features.drop(index=50)
top_features = top_features.drop(index=54)
top_features = top_features.drop(index=75)
top_features = top_features.drop(index=70)
top_features = top_features.iloc[0:30] # Can take anywhere from 20-40 features
print(top_features)
random_seed = 1 # Initialize random seed

df0_full = pd.concat([df0_subset['target'], df0_subset[top_features]], axis=1)
df1_full = pd.concat([df1_subset['target'], df1_subset[top_features]], axis=1)
df2_full = pd.concat([df2_subset['target'], df2_subset[top_features]], axis=1)
df3_full = pd.concat([df3_subset['target'], df3_subset[top_features]], axis=1)
df4_full = pd.concat([df4_subset['target'], df4_subset[top_features]], axis=1)
df5_full = pd.concat([df5_subset['target'], df5_subset[top_features]], axis=1)
df6_full = pd.concat([df6_subset['target'], df6_subset[top_features]], axis=1)

#%%
df_full = pd.concat([df0_full, df1_full, df2_full, df6_full, df4_full, df5_full], axis=0)

numeric = ['visitnumber', 'postbrowserheight', 'postbrowserwidth', 'lastpurchasenum', 
           'addonsymal', 'resolution', 'cdedspomodel', 'myaccountengagement', 'post_evar46', 
           'evar83', 'post_evar30', 'eventlistlength', 'productlistlength']

x_test_full = df3_full.iloc[:, 1:]
y_test_full = df3_full['target']

#%%
# Encode data and split into train + test sets
x_train_full = df_full.iloc[:, 1:]
y_train_full = df_full['target']

x_train_categorical = x_train_full.loc[:, ~x_train_full.columns.isin(numeric)]
x_test_categorical = x_test_full.loc[:, ~x_test_full.columns.isin(numeric)]

x_train_numerical = x_train_full.loc[:, x_train_full.columns.isin(numeric)]
x_test_numerical = x_test_full.loc[:, x_test_full.columns.isin(numeric)]

numerical_columns = list(x_train_numerical.columns)

# Ordinal encoding
ord_enc = OrdinalEncoder()
y_train = ord_enc.fit_transform(y_train_full.to_numpy().reshape(-1,1))
y_test = ord_enc.fit_transform(y_test_full.to_numpy().reshape(-1,1))

# Leave-one-out encoding
loo_enc = LeaveOneOutEncoder()
loo_enc.fit(x_train_categorical, y_train)
x_train_categorical_transformed = loo_enc.transform(x_train_categorical)
x_test_categorical_transformed = loo_enc.transform(x_test_categorical)

x_train_categorical_transformed = x_train_categorical_transformed.reset_index(drop=True)
x_test_categorical_transformed = x_test_categorical_transformed.reset_index(drop=True)

# Min-Max normalization
minmax = MinMaxScaler()
x_train_numerical_transformed = minmax.fit_transform(x_train_numerical)
x_test_numerical_transformed = minmax.transform(x_test_numerical)

# Create data frames and reset indices
x_train_numerical_transformed = pd.DataFrame(x_train_numerical_transformed, columns = [numerical_columns])
x_train_numerical_transformed = x_train_numerical_transformed.fillna(-1) #-999
x_test_numerical_transformed = pd.DataFrame(x_test_numerical_transformed, columns = [numerical_columns])
x_test_numerical_transformed = x_test_numerical_transformed.fillna(-1) #-999

y_train = (pd.DataFrame(y_train, columns=['target'])).reset_index(drop=True)
y_test = (pd.DataFrame(y_test, columns=['target'])).reset_index(drop=True)

x_full_train = pd.concat([y_train, x_train_categorical_transformed, x_train_numerical_transformed], axis=1)
x_full_test = pd.concat([y_test, x_test_categorical_transformed, x_test_numerical_transformed], axis=1)

#%%
# Construct training and testing set
x_train_1 = x_full_train[x_full_train['target'] == 1]
y_train_1 = x_train_1['target']
x_train_2 = x_full_train[x_full_train['target'] == 0]
y_train_2 = x_train_2['target']
x_train = (pd.concat([x_train_1, x_train_2])).iloc[:, 1:] + 1
y_train = pd.concat([y_train_1, y_train_2])

x_test_1 = x_full_test[x_full_test['target'] == 1]
y_test_1 = x_test_1['target']
x_test_2 = x_full_test[x_full_test['target'] == 0]
y_test_2 = x_test_2['target']
x_test = (pd.concat([x_test_1, x_test_2])).iloc[:, 1:] + 1
y_test = pd.concat([y_test_1, y_test_2])

x_train, y_train = shuffle(x_train, y_train)

x_train

#%%
#------------------------------#
### MODEL TRAINING + RESULTS ###
#------------------------------#
# Hyperparameters
num_epochs = 500 # number of epochs
num_obs = 4096 # batch size
adam_opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True) # Adam optimizer
sgd_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) # Stochastic gradient method optimizer

mlp = Sequential()
mlp.add(Dense(256, activation='sigmoid', input_dim=len(top_features))) # Sigmoid works better than ReLU and Tanh
mlp.add(Dropout(0.15))
mlp.add(Dense(256, activation='sigmoid'))
mlp.add(Dropout(0.15))
mlp.add(Dense(1, activation='sigmoid')) 
mlp.compile(loss='binary_crossentropy', optimizer=adam_opt, 
              metrics=['accuracy', keras.metrics.AUC(name='auc')])
mlp.summary()

#%%
# Class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)

class_weights = dict(zip(np.unique(y_train), class_weights))

#%%
# Train neural network
hist = mlp.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, 
                 batch_size=num_obs, class_weight=class_weights)

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

#%%
# Histogram
fig, ax = plt.subplots(figsize =(10, 7))
mlp_predictions = mlp.predict(x_test)
ax.hist(mlp_predictions * 100, bins = range(0,110,10))
plt.show()

# Confusion matrix
threshold_prob = 0.5 # depends on results
y_predicted = mlp_predictions > threshold_prob
mat = confusion_matrix(y_test, y_predicted)
classes = ['No Purchase', 'Purchase']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Counts)')
plt.show()

#%%
# Analysis of model inference
y_predicted = np.array(np.concatenate(mlp_predictions))
true_0_indices = np.where(y_test == 0)
true_1_indices = np.where(y_test == 1)
predict_0_indices = np.where(y_predicted < threshold_prob)[0]
predict_1_indices = np.where(y_predicted >= threshold_prob)[0]

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
#-------------------------#
### LOGISTIC REGRESSION ###
#-------------------------#
logreg = LogisticRegression(solver='lbfgs', max_iter=150)
    
logreg.fit(x_train, y_train)

fig, ax = plt.subplots(figsize =(10, 7))
logreg_predictions = logreg.predict(x_test)
ax.hist(logreg_predictions * 100, bins = range(0,110,10))
plt.show()

threshold_prob = 0.5 # depends on results
y_predicted = logreg_predictions > threshold_prob
mat = confusion_matrix(y_test, y_predicted)
classes = ['No Purchase', 'Purchase']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Counts)')
plt.show()