
import cv2
import numpy as np
import pickle

from face_detection import *
#from autoencoders import *
import pickle

def swap():

	source_filename = "videos/Easy/FrankUnderwood.mp4"
	replacement_filename = "videos/Easy/MrRobot.mp4"

	# Open source video
	source_video = cv2.VideoCapture(source_filename)
	replacement_video = cv2.VideoCapture(replacement_filename)

	''' Face detection '''

	# Detect faces in the source video

	print("Detecting faces from souce video...")
	#source_faces = detect_all_faces(source_video)
	#pickle.dump(source_faces, open("source_faces.p", "wb"))
	source_faces = pickle.load(open("source_faces.p", "rb"))
	num_src_face = len(source_faces)
	print("Found {0} faces!".format(num_src_face))

	# Detect faces in the replacement video

	print("Detecting face to replace from replacement video...")
	#replacement_faces = detect_all_faces(replacement_video)
	#pickle.dump(replacement_faces, open("replacement_faces.p", "wb"))
	replacement_faces = pickle.load(open("replacement_faces.p", "rb"))
	num_tgt_face = len(replacement_faces)
	print("Found {0} faces!".format(num_tgt_face))

	''' Autoencoders '''

	# Resize all the images to training size, and convert to floats

	s = 96
	source_array = np.zeros((num_src_face, s, s, 3)).astype('float32')
	for f in range(num_src_face):
		curr_face = cv2.resize(source_faces[f], (s, s))
		source_array[f,:,:,:] = curr_face.astype('float32') / 255.0
	
	with open("new_source.p", "wb") as f:
    	 pickle.dump(source_array, f)

	target_array = np.zeros((num_tgt_face, s, s, 3)).astype('float32')
	for f in range(num_tgt_face):
		curr_face = cv2.resize(replacement_faces[f], (s, s))
		target_array[f,:,:,:] = curr_face.astype('float32') / 255.0

	# Train autoencoders on the source and replacement faces

	part = np.random.permutation(np.arange(0, source_array.shape[0]))

	source_array = source_array[part]
	src_train = source_array[0:180]
	src_test = source_array[180:-1]
	#src_autoencoder_data = new_autoencoder(src_train, src_test)
	#pickle.dump(src_autoencoder_data, open("src_autoencoder_data.p", "wb"))
	src_autoencoder_data = pickle.load(open("src_autoencoder_data.p", "rb"))
	src_autoenc, src_enc, src_dec = src_autoencoder_data

	target_array = target_array[part]
	tgt_train = target_array[0:180]
	tgt_test = target_array[180:-1]
	#tgt_autoencoder_data = new_autoencoder(tgt_train, tgt_test)
	#pickle.dump(tgt_autoencoder_data, open("tgt_autoencoder_data.p", "wb"))
	tgt_autoencoder_data = pickle.load(open("tgt_autoencoder_data.p", "rb"))
	tgt_autoenc, tgt_enc, tgt_dec = tgt_autoencoder_data

	#testeroo
	n = 10
	src_samples = source_array[0:10*n:10]
	tgt_samples = target_array[0:10*n:10]
	predicted1 = src_autoenc.predict(src_samples)
	predicted2 = tgt_autoenc.predict(tgt_samples)

	# test encoding into itself
	encoded_step = src_enc.predict(src_samples)
	test1 = src_dec.predict(encoded_step)

	# test src->encoder->tgt
	test2 = tgt_dec.predict(src_enc.predict(src_samples))
	# and the other way
	test3 = src_dec.predict(tgt_enc.predict(tgt_samples))

	# Display the results
	for i in range(n):
	    cv2.imshow("source samples", cv2.resize(src_samples[i].reshape(s, s, 3), (480, 480)))
	    cv2.imshow("predicted src", cv2.resize(predicted1[i].reshape(s, s, 3), (480, 480)))
	    cv2.imshow("target samples", cv2.resize(tgt_samples[i].reshape(s, s, 3), (480, 480)))
	    cv2.imshow("predicted tgt", cv2.resize(predicted2[i].reshape(s, s, 3), (480, 480)))
	    cv2.imshow("test1", cv2.resize(test1[i].reshape(s, s, 3), (480, 480)))
	    cv2.imshow("test2", cv2.resize(test2[i].reshape(s, s, 3), (480, 480)))
	    cv2.imshow("test3", cv2.resize(test3[i].reshape(s, s, 3), (480, 480)))
	    cv2.waitKey(0)

swap()