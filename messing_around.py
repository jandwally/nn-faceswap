import pickle
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=2)
source = pickle.load(open("source_faces.p", "rb"))
target = pickle.load(open("replacement_faces.p", "rb"))


count = 0
for row in ax:
    count = count + 1
    for col in row:
        if count == 1:
            col.imshow(source[0])
        else:
            col.imshow(target[0])

plt.show()


n = 2

#ax = plt.subplot(2, n)
#plt.imshow(source[0])
#plt.show()
