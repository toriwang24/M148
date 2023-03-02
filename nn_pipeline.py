#---------------------#
### IMPORT PACKAGES ###
#---------------------#
#%%
import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import time

#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#-----------------------------------#
### DATA PROCESSING 1: CONVERSION ###
#-----------------------------------#
#%%
features = ['visitoridlow', 'checkoutthankyouflag', 'myaccountengagement', 'addonsymal', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
prop29 = ['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash']

#-------------------#
### Convert day 0 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_0.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_0.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_0 = pd.concat(dfs)
end = time.time()
print('Converted day 0 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-------------------#
### Convert day 1 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_1.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_1.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_1 = pd.concat(dfs)
end = time.time()
print('Converted day 1 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-------------------#
### Convert day 2 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_2.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_2.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_2 = pd.concat(dfs)
end = time.time()
print('Converted day 2 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-------------------#
### Convert day 3 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_3.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_3.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_3 = pd.concat(dfs)
end = time.time()
print('Converted day 3 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-------------------#
### Convert day 4 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_4.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_4.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_4 = pd.concat(dfs)
end = time.time()
print('Converted day 4 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-------------------#
### Convert day 5 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_5.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_5.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_5 = pd.concat(dfs)
end = time.time()
print('Converted day 5 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-------------------#
### Convert day 6 ###
#-------------------#
start = time.time()

pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_6.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_6.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = features)
        dfs.append(df_temp)

df_all_6 = pd.concat(dfs)
end = time.time()
print('Converted day 6 in ' + str(round(end - start, 2)) + ' seconds')
tar = None

#-----------------------------------#
### DATA PROCESSING 2: SUBSETTING ###
#-----------------------------------#
#%%
start = time.time()

#------------------#
### Subset day 0 ###
#------------------#
df_subset = df_all_0.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day0 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day0 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day0 = df_subset_1_day0[~df_subset_1_day0['visitoridlow'].isin(df_subset_2_day0['visitoridlow'].unique())]
df_subset_1_day0 = df_subset_1_day0.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day0 = df_subset_2_day0.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day0['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day0['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset = None

#------------------#
### Subset day 1 ###
#------------------#
df_subset = df_all_1.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day1 = df_subset_1_day1[~df_subset_1_day1['visitoridlow'].isin(df_subset_2_day1['visitoridlow'].unique())]
df_subset_1_day1 = df_subset_1_day1.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day1 = df_subset_2_day1.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day1['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day1['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset = None

#------------------#
### Subset day 2 ###
#------------------#
df_subset = df_all_2.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day2 = df_subset_1_day2[~df_subset_1_day2['visitoridlow'].isin(df_subset_2_day2['visitoridlow'].unique())]
df_subset_1_day2 = df_subset_1_day2.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day2 = df_subset_2_day2.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day2['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day2['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)

df_subset = None

#------------------#
### Subset day 3 ###
#------------------#
df_subset = df_all_3.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day3 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day3 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day3 = df_subset_1_day3[~df_subset_1_day3['visitoridlow'].isin(df_subset_2_day3['visitoridlow'].unique())]
df_subset_1_day3 = df_subset_1_day3.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day3 = df_subset_2_day3.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day3['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day3['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)

df_subset = None

#------------------#
### Subset day 4 ###
#------------------#
df_subset = df_all_4.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day4 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day4 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day4 = df_subset_1_day4[~df_subset_1_day4['visitoridlow'].isin(df_subset_2_day4['visitoridlow'].unique())]
df_subset_1_day4 = df_subset_1_day4.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day4 = df_subset_2_day4.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day4['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day4['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)

df_subset = None

#------------------#
### Subset day 5 ###
#------------------#
df_subset = df_all_5.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day5 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day5 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day5 = df_subset_1_day5[~df_subset_1_day5['visitoridlow'].isin(df_subset_2_day5['visitoridlow'].unique())]
df_subset_1_day5 = df_subset_1_day5.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day5 = df_subset_2_day5.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day5['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day5['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)

df_subset = None

#----------------------------#
### Subset day 6: test set ###
#----------------------------#
df_subset = df_all_6.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1_day6 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2_day6 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1_day6 = df_subset_1_day6[~df_subset_1_day6['visitoridlow'].isin(df_subset_2_day6['visitoridlow'].unique())]
df_subset_1_day6 = df_subset_1_day6.drop_duplicates(subset=['visitoridlow'])
df_subset_2_day6 = df_subset_2_day6.drop_duplicates(subset=['visitoridlow'])
df_subset_1_day6['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2_day6['prop29'].replace(prop29, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)

df_subset = None

end = time.time()
print('Data subsetting finished in ' + str(round(end - start, 2)) + ' seconds')

#---------------------------------#
### DATA PROCESSING 3: SAMPLING ###
#---------------------------------#
#%%
train_samp = 3000
test_samp = 6000

#------------------#
### Sample day 0 ###
#------------------#
df_subset_day0 = pd.concat([df_subset_1_day0.sample(n=train_samp), df_subset_2_day0.sample(n=train_samp)])
df_subset_day0 = df_subset_day0.sample(frac = 1) #, random_state=148
#------------------#
### Sample day 1 ###
#------------------#
df_subset_day1 = pd.concat([df_subset_1_day1.sample(n=train_samp), df_subset_2_day1.sample(n=train_samp)])
df_subset_day1 = df_subset_day1.sample(frac = 1) #, random_state=148
#------------------#
### Sample day 2 ###
#------------------#
df_subset_day2 = pd.concat([df_subset_1_day2.sample(n=train_samp), df_subset_2_day2.sample(n=train_samp)])
df_subset_day2 = df_subset_day2.sample(frac = 1) #, random_state=148
#------------------#
### Sample day 3 ###
#------------------#
df_subset_day3 = pd.concat([df_subset_1_day3.sample(n=train_samp), df_subset_2_day3.sample(n=train_samp)])
df_subset_day3 = df_subset_day3.sample(frac = 1) #, random_state=148
#------------------#
### Sample day 4 ###
#------------------#
df_subset_day4 = pd.concat([df_subset_1_day4.sample(n=train_samp), df_subset_2_day4.sample(n=train_samp)])
df_subset_day4 = df_subset_day4.sample(frac = 1) #, random_state=148
#------------------#
### Sample day 5 ###
#------------------#
df_subset_day5 = pd.concat([df_subset_1_day5.sample(n=train_samp), df_subset_2_day5.sample(n=train_samp)])
df_subset_day5 = df_subset_day5.sample(frac = 1) #, random_state=148
#----------------------------#
### Sample day 6: test set ###
#----------------------------#
df_subset_day6 = pd.concat([df_subset_1_day6.sample(n=test_samp), df_subset_2_day6.sample(n=test_samp)])
df_subset_day6 = df_subset_day6.sample(frac = 1) #, random_state=148

#-------------------------------#
### MODEL BUILDING + TRAINING ###
#-------------------------------#
#%%
#----------------------#
### Train-test split ###
#----------------------#
df_train = pd.concat([df_subset_day0, df_subset_day1, df_subset_day2, df_subset_day3, df_subset_day4, df_subset_day5])
df_train = df_train.sample(frac = 1)
df_test = df_subset_day6.sample(frac = 1)

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
num_epochs = 1000 #500
num_obs = 512 #24 (batch_size)
adam_opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True) #0.0025
sgd_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9) #0.0025

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=len(features) - 2))
model.add(Dropout(0.01)) #0.1
#model.add(Dense(128, activation='sigmoid')) #256
#model.add(Dropout(0.1)) #0.2
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.01)) #0.1
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer=adam_opt, metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

#%%
#--------------------#
### Model Training ###
#--------------------#
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=num_epochs, batch_size=num_obs)

#-------------------------------------------------#
### Accuracy, AUC, Loss, Confusion Matrix Plots ###
#-------------------------------------------------#
sns.set()

acc_score = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
auc_score = hist.history['auc']
val_auc = hist.history['val_auc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(auc_score) + 1)

plt.plot(epochs, auc_score, '-', label='Training AUC', alpha=1)
plt.plot(epochs, val_auc, '-', label='Validation AUC', alpha=0.75)
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend(loc='lower right')
plt.plot()
plt.show()

plt.plot(epochs, acc_score, '-', label='Training Accuracy', alpha=1)
plt.plot(epochs, val_acc, '-', label='Validation Accuracy', alpha=0.75)
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()

plt.plot(epochs, loss, '-', label='Training Loss', alpha=1)
plt.plot(epochs, val_loss, '-', label='Validation Loss', alpha=0.75)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()

y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
labels = ['0', '1']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

#%%