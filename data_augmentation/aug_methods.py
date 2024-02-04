# =============================================================================
# the 'image' input or each function must be the row-heavy 240x1 representation
# =============================================================================

from PIL import Image
import numpy as np
from skimage.transform import AffineTransform, warp
import pandas as pd


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


if __name__ =="__main__":
    # load data frames
    # load data frames
    train_df = pd.read_csv("raw_train.csv")
    
    # split training data into features and label
    train_images = train_df.iloc[:, :-1]
    train_labels = train_df.iloc[:, -1]

    train_images = train_images.to_numpy()
    train_labels = train_labels.to_numpy()
    train_data= train_df.to_numpy()

    #Check rotations against eachother
    labels1 =train_labels
    labels2 = np.zeros((2000,))
    labels2[:1000,]=train_labels
    labels2[1000:,]=train_labels
    rot = [25]
    for x in rot:
        print(x)
        rot1 = np.zeros((4000,240))
        for i  in range(1000):
            in1,in2 = np.random.randint(1,x,2)
            rot1[i,:]= rotate_clockwise(train_images[i], in1).reshape(240)
            rot1[i+1000,:]= rotate_anti_clockwise(train_images[i], in1).reshape(240)
            rot1[i+2000,:]= rotate_clockwise(train_images[i], in2).reshape(240)
            rot1[i+3000,:]= rotate_anti_clockwise(train_images[i], in2).reshape(240)
        datafile= np.zeros((5000,241))
        datafile[:1000,:]= train_data
        datafile[1000:,:-1]= rot1
        datafile[1000:3000,-1]= labels2
        datafile[3000:,-1]= labels2
        datafile= pd.DataFrame(datafile)
        datafile.to_csv(f"rot_upto{x}.csv",index=False)

    






