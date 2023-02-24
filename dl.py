import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

### Convert .tar to Pandas data frame ###
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
df_subset = pd.concat([df_subset_1.sample(n=len(df_subset_2.index)), df_subset_2]) #, random_state=148
#df_subset = pd.concat([df_subset_1.sample(n=10000), df_subset_2.sample(n=10000)])
df_subset = df_subset.sample(frac = 1) #, random_state=148

### Set up neural network ###
x = df_subset.drop(['evar23', 'checkoutthankyouflag'], axis=1)
x = np.asarray(x).astype('float32')
y = df_subset['checkoutthankyouflag']
y = np.asarray(y).astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y) #, random_state=148

model = Sequential() 
model.add(Dense(128, activation='sigmoid', input_dim=7))
model.add(Dropout(0.1))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.AUC(name='auc')]) #metrics=['accuracy']
model.summary()

### Neural network training ###
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=500, batch_size=250)

### Plot training and val accuracy ###
sns.set()

acc = hist.history['auc'] #hist.history['accuracy']
val = hist.history['val_auc'] #hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training AUC') #'Training Accuracy'
plt.plot(epochs, val, ':', label='Validation AUC') #'Training Accuracy'
plt.title('Training and Validation AUC') #'Training and Validation Accuracy'
plt.xlabel('Epoch')
plt.ylabel('AUC') #'Accuracy'
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