import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

#from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

##################################################
### DATA PROCESSING ### Sampling from days 0-5 ###
##################################################

#####################
### Convert day 0 ###
#####################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_0.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_0.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day0 = pd.concat([df_subset_1.sample(n=3000), df_subset_2.sample(n=3000)])
df_subset_day0 = df_subset_day0.sample(frac = 1) #, random_state=148

print('Converted day 0')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

#####################
### Convert day 1 ###
#####################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_1.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_1.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day1 = pd.concat([df_subset_1.sample(n=3000), df_subset_2.sample(n=3000)])
df_subset_day1 = df_subset_day1.sample(frac = 1) #, random_state=148

print('Converted day 1')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

#####################
### Convert day 2 ###
#####################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_2.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_2.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day2 = pd.concat([df_subset_1.sample(n=3000), df_subset_2.sample(n=3000)])
df_subset_day2 = df_subset_day2.sample(frac = 1) #, random_state=148

print('Converted day 2')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

#####################
### Convert day 3 ###
#####################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_3.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_3.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day3 = pd.concat([df_subset_1.sample(n=3000), df_subset_2.sample(n=3000)])
df_subset_day3 = df_subset_day3.sample(frac = 1) #, random_state=148

print('Converted day 3')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

#####################
### Convert day 4 ###
#####################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_4.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_4.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day4 = pd.concat([df_subset_1.sample(n=3000), df_subset_2.sample(n=3000)])
df_subset_day4 = df_subset_day4.sample(frac = 1) #, random_state=148

print('Converted day 4')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

#####################
### Convert day 5 ###
#####################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_5.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_5.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day5 = pd.concat([df_subset_1.sample(n=3000), df_subset_2.sample(n=3000)])
df_subset_day5 = df_subset_day5.sample(frac = 1) #, random_state=148

print('Converted day 5')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

###############################
### Convert day 6: test set ###
###############################
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_6.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_6.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

### Create data frame with subset of variables ###
# columns = ['evar23', 'checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 
#            'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor'] # Subset with 'dummy' variables
columns = ['evar23', 'checkoutthankyouflag', 'addonsymal', 'post_evar22', 'cdedspomodel', 'evar83', 'prop29', 'visitnumber', 'visitpagenum']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)

df_subset = df_all.iloc[::-1]
df_subset = df_subset.iloc[1: , :]
df_subset = df_subset.dropna()
df_subset = df_subset[df_subset['evar83'] != '']
df_subset_1 = df_subset.loc[df_subset['checkoutthankyouflag'] == 0]
df_subset_2 = df_subset.loc[df_subset['checkoutthankyouflag'] == 1]
df_subset_1 = df_subset_1[~df_subset_1['evar23'].isin(df_subset_2['evar23'].unique())]
df_subset_1 = df_subset_1.drop_duplicates(subset=['evar23'])
df_subset_2 = df_subset_2.drop_duplicates(subset=['evar23'])
df_subset_1['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
df_subset_2['prop29'].replace(['FSApproved', 'Mature', 'Prospect', 'Emerging', 'New', 'FSRetry', 'Graduate-FS', 'FreshStart', 'FSGrad', 'FSCleanup', 'Cash'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], inplace=True)
#df_subset = pd.concat([df_subset_1, df_subset_2])
#df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
df_subset_day6 = pd.concat([df_subset_1.sample(n=3600), df_subset_2.sample(n=3600)])
df_subset_day6 = df_subset_day6.sample(frac = 1) #, random_state=148

print('Converted day 6')
tar = None
df_all = None
df_subset = None
df_subset_1 = None
df_subset_2 = None

#######################################################################
### MODEL BUILDING ### Splitting data and setting up neural network ###
#######################################################################

### Test + train sets ###
df_train = pd.concat([df_subset_day0, df_subset_day1, df_subset_day2, df_subset_day3, df_subset_day4, df_subset_day5])
df_train = df_train.sample(frac = 1)
df_test = df_subset_day6.sample(frac = 1)

x_train = df_train.drop(['evar23', 'checkoutthankyouflag'], axis=1)
x_train = np.asarray(x_train).astype('float32')
y_train = df_train['checkoutthankyouflag']
y_train = np.asarray(y_train).astype('float32')

x_test = df_test.drop(['evar23', 'checkoutthankyouflag'], axis=1)
x_test = np.asarray(x_test).astype('float32')
y_test = df_test['checkoutthankyouflag']
y_test = np.asarray(y_test).astype('float32')

### Model set up ###
opt_alg = keras.optimizers.Adam(learning_rate=0.05)

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=7))
model.add(Dropout(0.1))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer=opt_alg, metrics=['accuracy', keras.metrics.AUC(name='auc')])
model.summary()

############################################################
### MODEL TRAINING ### Learning curve + confusion matrix ###
############################################################
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, batch_size=500)

### Plot training and val accuracy ###
sns.set()

auc_score = hist.history['auc']
val_auc = hist.history['val_auc']
acc_score = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(auc_score) + 1)

plt.plot(epochs, auc_score, '-', label='Training AUC')
plt.plot(epochs, val_auc, '-', label='Validation AUC')
plt.title('Training and Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend(loc='lower right')
plt.plot()
plt.show()

plt.plot(epochs, acc_score, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, '-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()

### Confusion matrix ###
y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
labels = ['0', '1']

sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()