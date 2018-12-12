from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
import pickle


# constants
EPOCHS = 200
BATCH_SIZE = 128


''' a simple autoencoder '''
def new_autoencoder(x_train, x_test):

	# Input
	shape = x_train.shape[1:]
	print(x_train.shape)
	print(shape)
	input_image = Input(shape=shape)

	# Encoder
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_image)
	# x = MaxPooling2D((2, 2), padding='same')(x)
	# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	print("encoded shape:", encoded.shape)

	# Decoder
	index_of_first_decoder_layer = 6
	first_decoder_layer = Conv2D(8, (3, 3), activation='relu', padding='same')
	x = first_decoder_layer(encoded)
	x = UpSampling2D((2, 2))(x)
	# x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
	# x = UpSampling2D((2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
	print("decoded shape:", decoded.shape)

	# Make autoencoder
	autoencoder = Model(inputs=[input_image], outputs=[decoded])

	# Encoded representation
	s = x_train.shape[1] / 4
	encoding_dim = (s, s, 8)
	encoder = Model(inputs=[input_image], outputs=[encoded])

	print(len(encoder.layers))
	print(len(autoencoder.layers))

	# Create a decoder model
	encoded_input = Input(shape=encoding_dim)
	x = first_decoder_layer(encoded_input)
	for i in range(index_of_first_decoder_layer, len(autoencoder.layers)):
		x = autoencoder.layers[i](x)			 #(there's probably an easier way to do this)
	decoder = Model(inputs=[encoded_input], outputs=[x])

	# Compile and train autoencoder
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))

	return autoencoder, encoder, decoder