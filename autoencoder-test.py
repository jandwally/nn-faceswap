from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import pickle

''' a simple autoencoder '''
def new_autoencoder(x_train, x_test):

	# Input
	shape = x_train.shape[1:-1]
	input_image = Input(shape=shape)

	# Encoder
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_image)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	# Decoder
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

	# Make and train autoencoder
	autoencoder = Model(input_image, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.fit(x_train, x_train,
		epochs=10, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

	return autoencoder



def mnist_test():

	# Training data
	(x_train, _), (x_test, _) = mnist.load_data()

	# Adding an extra channel dimension, convert to floats
	x_train = np.expand_dims(x_train, 3).astype('float32') / 255.0
	x_test = np.expand_dims(x_test, 3).astype('float32') / 255.0
	print("x_train.shape: ", x_train.shape)
	print("x_test.shape: ", x_test.shape)

	#autoenc = new_autoencoder(x_train, x_test)
	#pickle.dump(autoenc, open("autoenc.p", "wb"))
	autoenc = pickle.load(open("autoenc.p", "rb"))

	# Predict on the first ten
	n = 10
	samples = x_train[0:n]
	predicted = autoenc.predict(samples)

	#print(samples[0])
	#print(predicted[0])

	# Display the results
	for i in range(n):
	    # display original
	    ax = plt.subplot(2, n, i + 1)
	    plt.imshow(samples[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # display reconstruction
	    ax = plt.subplot(2, n, i + n + 1)
	    plt.imshow(predicted[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()








