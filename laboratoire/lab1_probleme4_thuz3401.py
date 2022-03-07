# S5 APP4
# Labo1

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import math
from zplane import zplane

# Probleme 4

plt.gray()
img = mpimg.imread('imagecouleur.png')

# On fait des loops dans ce probleme et pour lAPP

rows, columns = np.shape(img)
print(rows)
print(columns)

new_img = np.zeros([int(columns / 2), int(2 * rows)])

new_rows, new_columns = np.shape(img)
print(new_rows)
print(new_columns)

for row in range(0, rows):
    for column in range(0, columns):
        new_img[int(column / 2)][int(2 * row)] = img[column][row]

plt.imshow(new_img)
plt.show()
