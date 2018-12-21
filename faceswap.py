
import cv2
import numpy as np
import face_recognition
import pickle

from morphing import *
# import morph_tri_deniz as morph1
# import morph_tri_john as morph2


# helper func
def landmarks_to_array(landmarks):

    coords = []

    for d in landmarks:
        for key, value in d.items():
            for coord in value:
                temp = np.array(coord)
                coords.append(temp)

    array = np.array(coords)
    return array


# This part assumes we have the autoencoders trained
def swap(source_filename, replacement_filename, output_filename, which_frame):

    # Open video to replace faces on, and output video
    replacement_video = cv2.VideoCapture(replacement_filename)
    w = int(replacement_video.get(3))
    h = int(replacement_video.get(4))
    fps = replacement_video.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

    # Sample a source face
    source_video = cv2.VideoCapture(source_filename)
    for x in range(which_frame):
        source_video.read()
    success, first_frame = source_video.read()
    new_face_landmarks = face_recognition.face_landmarks(first_frame)
    new_face_coords = landmarks_to_array(new_face_landmarks)

    count = 0
    while True:
        print("\nIteration:", count)
        # if (count == 24): break

        # Get the next frame
        success, input_frame = replacement_video.read()
        if not success:
            print("Finished: frame", count)
            break

        ''' Detect features, and morph output face '''

        # Get facial landmarks
        print("Getting landmarks...")
        old_face_landmarks = face_recognition.face_landmarks(input_frame)
        old_face_coords = landmarks_to_array(old_face_landmarks)

        ''' Transform source face, to map it to the replacement frame '''

        # Pad the source face to make it the right shape
        new_shape = (max(input_frame.shape[0], first_frame.shape[0]), max(input_frame.shape[1], first_frame.shape[1]), 3)
        new_face = np.zeros(new_shape, dtype=first_frame.dtype)
        new_face[0 : first_frame.shape[0], 0 : first_frame.shape[1], :] = first_frame

        # Do image morphing (call code from project 2A)
        print("Warping face...")
        warped_face = morph_face(new_face, input_frame, new_face_coords, old_face_coords)
        #warped_face = morph1.morph_tri(new_face, input_frame, new_face_coords, old_face_coords, [1], [0])[0]
        #warped_face2 = morph2.morph_tri(new_face, input_frame, new_face_coords, old_face_coords, np.array([1]), np.array([0]))[0]

        ''' Stitch face and blend to get output frame '''

        # Compute location of center by averaging and facial landmarks
        max_y, max_x = np.max(old_face_coords[:,1]), np.max(old_face_coords[:,0])
        min_y, min_x = np.min(old_face_coords[:,1]), np.min(old_face_coords[:,0])
        center = (int((max_x + min_x)/2), int((max_y + min_y)/2))
        # print("y:", min_y, "->", max_y)
        # print("x:", min_x, "->", max_x)

        # Crop the warped face, create an all white mask
        warped_face = warped_face[min_y : max_y, min_x : max_x, :]
        mask = 255 * np.ones(warped_face.shape, warped_face.dtype)
        mask[np.where(np.sum(warped_face, axis=2) == 0)] = np.array([0,0,0])

        # Seamlessly clone src into dst and put the results in output
        print("Cloning...")
        output_frame = cv2.seamlessClone(
            np.uint8(warped_face),
            np.uint8(input_frame),
            np.uint8(mask),
            center, cv2.NORMAL_CLONE)

        # Save frame
        print("Writing output frame...")
        output_video.write(output_frame)
        count = count + 1

    print("Done!")
    source_video.release()
    replacement_video.release()
    output_video.release()



# source_filename = "videos/Easy/MrRobot.mp4"
# replacement_filename = "videos/Easy/FrankUnderwood.mp4"
# output_filename = "out.avi"
# which_frame = 60

# swap(source_filename, replacement_filename, output_filename, which_frame)