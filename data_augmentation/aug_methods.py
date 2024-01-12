# =============================================================================
# the 'image' input or each function must be the row-heavy 240x1 representation
# =============================================================================

from PIL import Image
import numpy as np
from skimage.transform import AffineTransform, warp

# =====================================
# rotation
# =====================================

def rotate_clockwise(image, angle):

    # convert the image to type uint8 then convert to PIL image
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    
    # rotato potato then convert PIL image back to numpy array
    rotated_image = image.rotate(-angle)
    output_image = np.array(rotated_image)
    
    return output_image

def rotate_anti_clockwise(image, angle):
    return rotate_clockwise(image, -angle)

# =====================================
# reflection
# =====================================

def reflect_horizontal(image):
    return np.flip(image, axis=1)

def reflect_vertical(image):
    return np.flip(image, axis=0)

# =====================================
# shearing: useless, from what i can see
# =====================================

def shear(image, shear_intensity, direction):

    if direction == 'horizontal':
        transform = AffineTransform(shear=shear_intensity)
    elif direction == 'veritcal':
        transform = AffineTransform(shear=shear_intensity).inverse

    output_image = warp(image, transform, mode='wrap')

    return output_image

# =====================================
# translation
# =====================================

def translate(image, x, y):

    transform = AffineTransform(translation=(x, y))
    output_image = warp(image, transform, mode='wrap')

    return output_image

# =====================================
# adding noise
# =====================================

def add_noise(image, noise_level):

    gaussian_noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + gaussian_noise
    output_image = np.clip(noisy_image, 0, 6)

    return output_image

# =====================================
# changing greyscale intensities
# =====================================

def amplify_greyscale(image, intensity_factor):

    intensity_scaled_image = image * intensity_factor
    output_image = np.clip(intensity_scaled_image, 0, 6)

    return output_image
