import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = pd.read_csv("diabets-after-CD.csv")

data.info()

data.isnull().sum()
data.corr()

data = data[["Glucose", "BMI", "Age", "Outcome"]]

print(data[["Glucose", "BMI", "Age", "Outcome"]])

print(len(data.columns))
X = data.values[:, :3]
y = data.values[:, 3]

# 5 hidden layer
# create model
model1 = Sequential()
model1.add(Dense(32, input_dim=3, activation='relu'))
model1.add(Dense(16, activation='relu'))
model1.add(Dense(8, activation='relu'))
model1.add(Dense(4, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))
# Compile model
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model1.fit(X, y, validation_split=0.33, epochs=75, batch_size=64, verbose=0)
print(history.history.keys())

# val_acc
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predicts = pd.DataFrame(model1.predict(X))
predicts.columns=["tyle"]
predicts["nhan"] = y
print(predicts[predicts.nhan == 1].sort_values(by=['tyle']))
print(predicts[predicts.nhan == 0].sort_values(by=['tyle']))