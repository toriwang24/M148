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
#------------------#
### PROCESS DATA ###
#------------------#
#%%
data_files = ['hitdata7days_0.tar.gz', 'hitdata7days_1.tar.gz', 'hitdata7days_2.tar.gz', 'hitdata7days_3.tar.gz', 
              'hitdata7days_4.tar.gz', 'hitdata7days_5.tar.gz', 'hitdata7days_6.tar.gz']

df_0 = read_data(data_files[0])
df_1 = read_data(data_files[1])
df_2 = read_data(data_files[2])
df_3 = read_data(data_files[3])
df_4 = read_data(data_files[4])
df_5 = read_data(data_files[5])
df_6 = read_data(data_files[6])

#%%
day0_subset, target_0, imp_features = feature_importance(df_0)
day1_subset, target_1, dummy_vars = feature_importance(df_1)
day2_subset, target_2, dummy_vars = feature_importance(df_2)
day3_subset, target_3, dummy_vars = feature_importance(df_3)
day4_subset, target_4, dummy_vars = feature_importance(df_4)
day5_subset, target_5, dummy_vars = feature_importance(df_5)
day6_subset, target_6, dummy_vars = feature_importance(df_6)

#%%
n = 30
day0_subset_n, n_features = data_n_features(day0_subset, imp_features, n, target_0)
day1_subset_n, dummy_vars = data_n_features(day1_subset, imp_features, n, target_1)
day2_subset_n, dummy_vars = data_n_features(day2_subset, imp_features, n, target_2)
day3_subset_n, dummy_vars = data_n_features(day3_subset, imp_features, n, target_3)
day4_subset_n, dummy_vars = data_n_features(day4_subset, imp_features, n, target_4)
day5_subset_n, dummy_vars = data_n_features(day5_subset, imp_features, n, target_5)
day6_subset_n, dummy_vars = data_n_features(day6_subset, imp_features, n, target_6)

print(n_features)

#%%
### Function to encode data ###
def encoded_data(x_train):
    x_train_1 = x_train[x_train['target'] == "1"]
    y_train_1 = x_train_1['target']
    x_train_2 = x_train[x_train['target'] == "0"].sample(n=len(x_train_1) + 500)
    y_train_2 = x_train_2['target']
    x_train = (pd.concat([x_train_1, x_train_2])).iloc[:, 1:]
    y_train = pd.concat([y_train_1, y_train_2])

    ord_enc = OrdinalEncoder()
    x_train = ord_enc.fit_transform(x_train)
    y_train = ord_enc.fit_transform(y_train.to_numpy().reshape(-1,1))
    full_train = np.append(x_train, y_train, axis=1)
    np.random.shuffle(full_train)
    x_train = full_train[:, :-1]
    y_train = full_train[:, -1]

    return x_train, y_train

x_train_0, y_train_0 = encoded_data(day0_subset_n)
x_train_1, y_train_1 = encoded_data(day1_subset_n)
x_train_2, y_train_2 = encoded_data(day2_subset_n)
x_train_3, y_train_3 = encoded_data(day3_subset_n)
x_train_4, y_train_4 = encoded_data(day4_subset_n)
x_train_5, y_train_5 = encoded_data(day5_subset_n)
x_test, y_test = encoded_data(day6_subset_n)

#%%
x_train = np.concatenate((x_train_0, x_train_1, x_train_2, x_train_3, x_train_4, x_train_5), axis=0)
y_train = np.concatenate((y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5), axis=0)

#%%
#------------------------------#
### MODEL TRAINING + RESULTS ###
#------------------------------#

### Hyperparameters ###
num_epochs = 1000 # 500 works well in practice
num_obs = 128 # batch size: larger values such as 256 generally work well in practice
adam_opt = keras.optimizers.Adam(learning_rate=0.00075, amsgrad=True) # Adam optimizer, 0.0025 learning rate generally works well
sgd_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) # Stochastic gradient method optimizer

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=len(n_features))) # Sigmoid for all layers works better than ReLU and Tanh
model.add(Dropout(0.1)) # Dropout regularization value, 0.1 generally works well
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2)) # Dropout regularization value, 0.1 generally works well
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

#%%
### Train neural network ###
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=num_obs)

#%%
### Plot learning curves ###
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

### Confusion matrix ###
y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
classes = ['0', '1']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix (Counts)')
plt.show()

### Analysis of model inference ###
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
#%%