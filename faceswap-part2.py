
import cv2
import numpy as np
import face_recognition

from face_detection import *
import morph_tri_deniz as morph1
import morph_tri_john as morph2
import pickle


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
def second_part():

    replacement_filename = "videos/Easy/FrankUnderwood.mp4"
    source_filename = "videos/Easy/MrRobot.mp4"

    # Open video to replace faces on, and output video
    replacement_video = cv2.VideoCapture(replacement_filename)
    w = int(replacement_video.get(3))
    h = int(replacement_video.get(4))
    fps = replacement_video.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter("out.avi",cv2.VideoWriter_fourcc(*'MJPG'), fps, (w,h))

    source_video = cv2.VideoCapture(source_filename)
    success, test_frame = source_video.read()

    count = 0
    while True:
        if count == 2:
            break
        print("\nIteration:", count)

        # Get the next frame
        success, input_frame = replacement_video.read()
        if not success:
            break

        ''' Encode face, and decode input into source face '''
        new_face = test_frame

        ''' Detect features, and morph output face '''

        # Get facial landmarks
        print("Getting landmarks...")
        old_face_landmarks = face_recognition.face_landmarks(input_frame)
        new_face_landmarks = face_recognition.face_landmarks(new_face)
        old_face_coords = landmarks_to_array(old_face_landmarks)
        new_face_coords = landmarks_to_array(new_face_landmarks)

        # Do image morphing (call code from project 2A)
        print("Warping face...")
        warped_face = morph1.morph_tri(new_face, input_frame, new_face_coords, old_face_coords, [1], [0])[0]
        # warped_face2 = morph2.morph_tri(new_face, input_frame, new_face_coords, old_face_coords, np.array([1]), np.array([0]))[0]

        # cv2.imshow("im1", new_face)
        # cv2.imshow("im2", input_frame)
        # cv2.imshow("Warped face test", warped_face)

        ''' Stitch face and blend to get output frame '''

        # Create an all white mask
        mask = 255 * warped_face

        # The location of the center of the src in the dst
        max_x, max_y = np.max(old_face_coords[:,1]), np.max(old_face_coords[:,0])
        min_x, min_y = np.min(old_face_coords[:,1]), np.min(old_face_coords[:,0])
        center = (int((max_y + min_y)/2), int((max_x + min_x)/2))

        # Seamlessly clone src into dst and put the results in output
        print("Cloning...")
        output_frame = cv2.seamlessClone(warped_face, input_frame, mask, center, cv2.NORMAL_CLONE)

        # Save frame
        print("Writing output frame...")
        output_video.write(output_frame)
        count = count + 1

    print("Done!")
    replacement_video.release()
    output_video.release()


second_part()