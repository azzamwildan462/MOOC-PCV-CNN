from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Mendefinisikan data set
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[1], [0], [0], [0]])

# Mendefinisikan Model Keras
model = Sequential()
model.add(Dense(100, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model Keras.
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Fit model ke data set fit the keras model on the dataset
model.fit(X, Y, epochs=1000, batch_size=10)
# Mengevaluasi model
_, akurasi = model.evaluate(X, Y)
print('Akurasi: %.2f' % (akurasi*100))

# Memprediksi kembali vektor Input X
prediksi = model.predict(X)
# Menbulatkan Hasil Presiksi
HasilPrediksi = [round(x[0]) for x in prediksi]
print('Prediksi ', HasilPrediksi)
