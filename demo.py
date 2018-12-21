
from faceswap import swap


# videos
frank = "videos/Easy/FrankUnderwood.mp4"
robot = "videos/Easy/MrRobot.mp4"
jon = "videos/Easy/JonSnow.mp4"
rosso1 = "videos/Medium/LucianoRosso1.mp4"
rosso2 = "videos/Medium/LucianoRosso2.mp4"
rosso3 = "videos/Medium/LucianoRosso3.mp4"
joker = "videos/Hard/Joker.mp4"


''' ARGUMENTS: '''

# Filename of the source video: the face to swap, and which frame to use as the source face
source_filename = robot
which_frame = 0

# Filename of the target video: where faces will be replaced
replacement_filename = frank

# What to save the output as
output_filename = "out.avi"


print("FACE SWAPPING\n")

print("     Source image :", source_filename)
print("Replacement image :", replacement_filename)
print("  Saving image as :", output_filename)

swap(source_filename, replacement_filename, output_filename, which_frame)