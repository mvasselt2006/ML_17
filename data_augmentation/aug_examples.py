# =============================================================================
# example use cases of the augmentation functions
# =============================================================================

from aug_methods import *

import matplotlib.pyplot as plt
import numpy as np

# load data
with open('../data.txt', 'r') as file:
    dataset = np.array([list(map(int, line.split())) for line in file])

# =========================================================
# example use of rotating a 3 by 90 degrees clockwise
# =========================================================

# change image from row-heavy vector to 16x15 matrix
image = dataset[600].reshape(16, 15)

# rotato potato, angle in degrees
angle = 90
modified_image = rotate_clockwise(image, angle)

# plots
plt.figure(figsize=(6, 3))

# plot original
plt.subplot(1, 2, 1)
plt.title("unchanged")
plt.imshow(image, cmap='gray')
plt.axis('off')

# plot rotated
plt.subplot(1, 2, 2)
plt.title("rotated")
plt.imshow(modified_image, cmap='gray')
plt.axis('off')

plt.show()

# =========================================================
# example use of reflecting an 8 vertically
# =========================================================

# change image from row-heavy vector to 16x15 matrix
image = dataset[1600].reshape(16, 15)

# reflecto patronum
modified_image = reflect_vertical(image)

# plots
plt.figure(figsize=(6, 3))

# plot original
plt.subplot(1, 2, 1)
plt.title("unchanged")
plt.imshow(image, cmap='gray')
plt.axis('off')

# plot rotated
plt.subplot(1, 2, 2)
plt.title("reflected")
plt.imshow(modified_image, cmap='gray')
plt.axis('off')

plt.show()

# =========================================================
# example use of shearing a 9
# =========================================================

# change image from row-heavy vector to 16x15 matrix
image = dataset[1900].reshape(16, 15)

# reflecto patronum
modified_image = shear(image, 0.7, 'horizontal')

# plots
plt.figure(figsize=(6, 3))

# plot original
plt.subplot(1, 2, 1)
plt.title("unchanged")
plt.imshow(image, cmap='gray')
plt.axis('off')

# plot rotated
plt.subplot(1, 2, 2)
plt.title("sheared")
plt.imshow(modified_image, cmap='gray')
plt.axis('off')

plt.show()

# =========================================================
# example use of translating a 2 
# =========================================================

# change image from row-heavy vector to 16x15 matrix
image = dataset[400].reshape(16, 15)

# reflecto patronum
modified_image = translate(image, 1, 0)

# plots
plt.figure(figsize=(6, 3))

# plot original
plt.subplot(1, 2, 1)
plt.title("unchanged")
plt.imshow(image, cmap='gray')
plt.axis('off')

# plot rotated
plt.subplot(1, 2, 2)
plt.title("translated")
plt.imshow(modified_image, cmap='gray')
plt.axis('off')

plt.show()

# =========================================================
# example use of adding noise to a 7
# =========================================================

# change image from row-heavy vector to 16x15 matrix
image = dataset[1500].reshape(16, 15)

# reflecto patronum
modified_image = add_noise(image, 0.6)

# plots
plt.figure(figsize=(6, 3))

# plot original
plt.subplot(1, 2, 1)
plt.title("unchanged")
plt.imshow(image, cmap='gray')
plt.axis('off')

# plot rotated
plt.subplot(1, 2, 2)
plt.title("noise added")
plt.imshow(modified_image, cmap='gray')
plt.axis('off')

plt.show()

# =========================================================
# example use of amplifying greyscale intensities
# =========================================================

# change image from row-heavy vector to 16x15 matrix
image = dataset[300].reshape(16, 15)

# reflecto patronum
modified_image = amplify_greyscale(image, 1.5)

# plots
plt.figure(figsize=(6, 3))

# plot original
plt.subplot(1, 2, 1)
plt.title("unchanged")
plt.imshow(image, cmap='gray')
plt.axis('off')

# plot rotated
plt.subplot(1, 2, 2)
plt.title("greyscales amplified")
plt.imshow(modified_image, cmap='gray')
plt.axis('off')

plt.show()
