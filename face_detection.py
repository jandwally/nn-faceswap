'''
	Modified from open source code at:
	https://github.com/shantnu/FaceDetect
'''

import cv2

def face_detect(cascade, image):

	# Take the image, convert to gray
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = cascade.detectMultiScale(
	    gray,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.CASCADE_SCALE_IMAGE
	)
	#print("Found {0} faces!".format(len(faces)))
	return faces


'''
	INPUT: video - video to detect faces in
	OUTPUT: all_faces - a list of slices of all the faces found in the video
'''
def detect_all_faces(video):

	# Initialize stuff for face detection
	cascPath = "haarcascade_frontalface_default.xml"
	cascade = cv2.CascadeClassifier(cascPath)

	all_faces = []

	while True:

		# Try to read in a frame
		success, frame = video.read()
		if not success:
			break

		# Detect a face in the current frame
		faces = face_detect(cascade, frame)
		x, y, w, h = faces[0]

		# Save this face
		face_slice = frame[y : y+h, x : x+w, :]
		all_faces.append(face_slice)

		# Draw a rectangle around the faces
		# copied_image = frame.copy()
		# for (x, y, w, h) in faces:
		#     cv2.rectangle(copied_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

		# cv2.imshow("Faces found", copied_image)
		# cv2.waitKey(0)

		# cv2.imshow("Faces slice", face_slice)
		# cv2.waitKey(0)

	return all_faces