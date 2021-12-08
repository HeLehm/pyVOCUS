import numpy as np
import cv2
import math

def calculate_msr(saliency_map, threshold = 0.25,**kwargs):
    """
    calculates and returns msr saliency_map
    Parameters
    -------
    saliency_map : np.array
        the saliency map
    threshold : float
        threshold as descirbed in the paper
        default: 0.25
    **kwargs: dict
        not used
    Returns
    ------
    msr: np.array dtype = uint8
        the msr as bitmap
    """
    _seed = np.max(saliency_map)
    _seed_index = np.unravel_index(
        np.argmax(saliency_map, axis=None),
        saliency_map.shape
    )

    _25p_seed_lower_bound = _seed - (float(_seed) * threshold)
    #map where 1 = pixel value is more higher than seed - (25% of seed)
    in_range_map = (saliency_map >= _25p_seed_lower_bound)

    seed_list = [_seed_index]
    NEIGHBORS = [
        (-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),
    ]

    return_map = np.zeros(
        shape=saliency_map.shape,
        dtype=np.uint8
    )

    while len(seed_list):
        currentPoint = seed_list.pop(0)
        for neighbor in NEIGHBORS:
            n_index = (
                currentPoint[0] + neighbor[0],
                currentPoint[1] + neighbor[1]
            )
            #skip to next loop if index is out of range
            if (n_index[0] >= in_range_map.shape[0]
                or n_index[0] < 0
                or n_index[1] >= in_range_map.shape[1]
                or n_index[1] < 0):
                continue
            #if value is in exepted range
            if in_range_map[n_index[0],n_index[1]] and return_map[n_index[0],n_index[1]] == 0:
                return_map[n_index[0],n_index[1]] = 1
                seed_list.append(n_index)

    
    return return_map.astype(np.uint8)

def get_FOA(fao_s, threshold = 0.25, dilation_kernel = (30,30)):
    """
    Parameters
    -------
    fao_s : np.array
        old salience map
    threshold : float
        used in calculate msr
        default : 0.25
    dilation_kernel : (int,int)
        the dilation kernel to dilate the msr
    RETURNS
    -------
    msr,new fao_s
        msr : most salien region as grayscale map
        new fao_s : old fao_s with the msr dilated and set to zero
    """
    #most salient region
    msr = calculate_msr(fao_s,threshold = threshold)
    #dilation
    kernel = np.ones(dilation_kernel, np.uint8)
    ior = cv2.dilate(msr, kernel, iterations=1)
    #return msr and new _FAOS
    return msr, np.where(ior > 0, 0,fao_s)

def uniqueness_weight(img, threshold = 0.5):
    """
    Parameters
    -------
    img: np.array
        the feature map to be weighted
    threshold : float
        threshold as described in the paper as a decimal
    Returns
    ------
    weight: float
        (paper page:65 - 66)
    """
    #avoid divison by 0
    if np.max(img) == np.min(img): return 0
    #threshold as apixel value
    thresh_pixel_val =  ((np.max(img) - np.min(img)) *  threshold) + np.min(img)
    #m in paper
    _m = np.count_nonzero(img > thresh_pixel_val)
    _sqrt_m = math.sqrt(_m)
    #calculate W(X)
    weight = img.size / _sqrt_m
    return weight