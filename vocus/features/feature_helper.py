import numpy as np
import cv2 
from .pyramide import PyramideType,PyramideItter

def center_surround_pyramide(image,sigma:int,p_type:PyramideType):
    """
    -- img -- the image to perform the pyramide on
    -- kwargs -- kwargs for center_soround
    -- returns list of images in scale 2 to 4 with center sourund applied
    -- where index 0 is s2 and index 2 is s4
    """
    return_list = []
    for i, simg in enumerate(
        PyramideItter(
            img=image,
            layers=5,
            p_type=p_type
        )
    ):
        #check if simg is a scale we need
        if i > 1:
            #apply center suround
            scaled_center_suround_img = center_suround(simg,sigma)
            return_list.append(scaled_center_suround_img)
    return return_list

def center_suround(image, sigma:int):
    """
    image -- scaled image (should be grayscale / one color)
    sigma -- type: int -- size of comparison kernel
    """
    assert len(image.shape) == 2, "image has wrong shape"

    #create image to store the values in
    result = np.zeros(shape=image.shape,dtype=np.int32) #uint8 isnt big enough / we might have negative numbers
    #itterate over each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #calculate mean of souounding pixels
            surround = np.mean(
                image[
                    max(0, i - sigma) : min(image.shape[0] - 1, i + sigma),
                    max(0, j - sigma) : min(image.shape[1] - 1, j + sigma)
                ]
            )
            #set corresponding pixel in result
            result[i,j] = max((np.int32(image[i,j]) - np.int32(surround)),0)
    return result

def across_scale_add(images:list):
    """
    across scale operation as describes in the paper
    -- images : list of cv2 images (where index 0 is the biggest on (as in _center_surround_pyramide))
    returns one image
    """
    same_size_images = []
    target_size = (images[0].shape[1],images[0].shape[0])
    for i, simg in enumerate(images):
        same_size_images.append(cv2.resize(simg, target_size,interpolation= cv2.INTER_NEAREST))
    result = np.array(same_size_images[0],dtype=np.int) # convert to np.int because int8 is to small
    for i in range(1,len(same_size_images)):
        result = np.add(result,same_size_images[i])
    return result
    

def scale_up(image):
    """
    Scales the image up
    uses cv2 resize for now...
    img -- cv2 image to scale up
    returns -- upscaled image (twice the size)
    """
    width = int(image.shape[1] * 2)
    height = int(image.shape[0] * 2)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

def create_orientation_map(img, theta):
    """
    -- img scaled grayscale image to apply gabor filters on
    -- theta angle in radiens to use in gabor filter
    retruns a feature map for a single scale with a gabor filter applied
    """
    #there might be other values that work better...
    g_kernel = cv2.getGaborKernel((6, 6), 10, theta, 5, 0, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
    return filtered_img
    
def create_orientation_feature_maps(img,theta):
    """
    -- img the s0 scale image in grayscale
    -- theta the angle to use in the gabor fiters
    returns a across scale added feature map for scales s2 - s4 with a given theta
    """
    return_list = []
    for i, simg in enumerate(
        PyramideItter(
            img=img,
            layers=5,
            p_type=PyramideType.LAPLACIAN
        )
    ):
        #check if simg is a scale we need
        if i > 1:
            #apply gabor filters
            scaled_center_suround_img = create_orientation_map(simg,theta=theta)
            return_list.append(scaled_center_suround_img)
    return return_list

def get_center_surround_pyramide_feature_map(channel):
    """
    returns a feature map that is computed by the center surround pyramide
    with scales s2-s4 and sigma 3 and 7
    """
    f_3 = across_scale_add(center_surround_pyramide(channel,3,PyramideType.GAUSSIAN))
    f_7 = across_scale_add(center_surround_pyramide(channel,7,PyramideType.GAUSSIAN))
    return across_scale_add([f_3, f_7])

def deg_in_rad(degree):
    """
    converts degree to radiens
    """
    radiant = 2*np.pi/360 * degree
    return radiant

def min_max_scale(img, _range=(0.,1.)):
    """
    min max scale function over all axis
    """
    scaled_img = (img - np.min(img) / np.max(img) - np.min(img))
    in_range = (scaled_img * (_range[1] - _range[0])) + _range[0]
    return in_range
