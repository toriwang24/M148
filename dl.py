import tarfile
import pandas as pd
import pyarrow.parquet as pq
import numpy as np

# Convert .tar to Pandas data frame
pd.set_option('display.max_columns', None)
tar = tarfile.open("hitdata7days_0.tar.gz","r:gz")
tar.extractall()
tar = tarfile.open("hitdata7days_0.tar.gz","r:gz")
tn = tar.next()
tn = tar.next()
pq.read_schema(tn.name)
df = pd.read_parquet(tn.name)

# Create data frame with subset of variables
columns = ['evar23','checkoutthankyouflag', 'visitnumber', 'visitpagenum', 'newvisit', 'hourlyvisitor', 'dailyvisitor', 'monthlyvisitor', 'yearlyvisitor']
dfs = []
for member in tar:
    if member.isreg():
        df_temp = pd.read_parquet(member.name,columns = columns)
        dfs.append(df_temp)

df_all = pd.concat(dfs)
df_subset = df_all.drop_duplicates(subset=['evar23'])
df_subset = df_subset.iloc[1: , :]

# Import packages for deep learning
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set up neural network
x = df_subset.drop(['evar23', 'checkoutthankyouflag'], axis=1)
x = np.asarray(x).astype('float32')
y = df_subset['checkoutthankyouflag']
y = np.asarray(y).astype('float32')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=148)

model = Sequential() 
model.add(Dense(128, activation='relu', input_dim=7))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.summary()

# Neural network training
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)

# Plot training and val accuracy
sns.set()

acc = hist.history['accuracy']
val = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)
 
plt.plot(epochs, acc, '-', label='Training accuracy')
plt.plot(epochs, val, ':', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()

# Confusion matrix
y_predicted = model.predict(x_test) > 0.5
mat = confusion_matrix(y_test, y_predicted)
labels = ['1', '0']
 
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)
 
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()