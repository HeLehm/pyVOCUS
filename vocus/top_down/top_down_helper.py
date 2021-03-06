
from util import plot_imgs
from ..helper import calculate_msr
import numpy as np
import numpy.ma as ma
import cv2

def learn_weights(imgs,regions,bottom_up_class,**kwargs):
    """
    learn weights for multiple image
    Parameter
    ------------
    imgs: list[image or any other input data as np.array]
        list of images 
        note: need corresponding region
    regions: tuple
        list of manual region of interest as slice
        form: [(y1, y2, x1, x2),...]
    bottom_up_class: subclass of bottom_up_part
        will be used to compute bottom_up_saliency_map
    kwargs:
        additional argumennts for:
        -calculate_msr
        -initialisaton of bottom up part --> so t will need img keyword in most cases
    Returns
    ------------
    np.array of weights
        where index 0 = feature 0 and index len(weights) - 1 = the last conspicous map
    """
    assert len(imgs) == len(regions), "Not the same number of images passed as images"
    weights = []
    for i, img in enumerate(imgs):
        _weights_i = _learn(regions[i],bottom_up_class,img=img,**kwargs)
        weights.append(_weights_i)

    #average weights
    weights = np.array(weights)
    weight_prod = np.prod(weights,axis=0)
    return np.power(weight_prod,1./float(len(imgs)))

def _learn(region_slices_,bottom_up_class,**kwargs):
    """
    NOTE: Probably needs img as keyword
    learn weights for single image
    Parameter
    ------------
    region_slices : tuple
        manual region of interest as slice
        form: (y1, y2, x1, x2)
    bottom_up_class: subclass of bottom_up_part
        will be used to compute bottom_up_saliency_map
    kwargs:
        additional argumennts for:
        -calculate_msr
        -initialisaton of bottom up part
    Returns
    ------------
    np.array of weights
        array of weights
    """
    bottom_up_instance = bottom_up_class(**kwargs)
    #get saliency map and crop to roi
    saliency_map = bottom_up_instance.get_saliency_map()
    #convert slices to fit s2
    #this has to be modiefied if bottom_up_instance s ustom class that doesnt use s2 scale saliency_map
    region_slices= tuple([int(d / 4) for d in list(region_slices_)]) 
    
    #get roi
    roi = saliency_map[region_slices[0]:region_slices[1],region_slices[2]:region_slices[3]]

    #create msr that has same shape as saliency_map, feature_maps and con_maps
    msr = np.zeros(shape=saliency_map.shape).astype(np.bool)
    msr[region_slices[0]:region_slices[1],region_slices[2]:region_slices[3]] = calculate_msr(saliency_map=roi,**kwargs).astype(np.bool)

    #get features to weigh
    feature_maps = bottom_up_instance.get_feature_maps()
    con_maps = bottom_up_instance.get_conspicous_maps()

    weights  = []
    for xi in feature_maps + con_maps:
        mi_msr = ma.masked_array(xi,mask= msr).mean()
        mi_img = ma.masked_array(xi,mask= np.invert(msr)).mean()
        if mi_img == 0.:
            weights.append(mi_msr)
            continue
        weight_i  = mi_msr / mi_img
        weights.append(weight_i)

    return np.array(weights)


def search_with_weights(weights, _t, bottom_up_class, **kwargs):
    """
    Parameter
    --------
    weights : np.array
        weights as descibed in the paper
        can be computed by _learn or learnweights function
    _t: float 0...1
        weight of the top down map
    bottom_up_class : bottom_up_part subclass
        NOT AN INSTANCE!
    Returns
    -------
    top down saliency_map: np.array
        as descibed in the paper
    """
    bottom_up_instance = bottom_up_class(**kwargs)
    bu_saliency_map = bottom_up_instance.get_saliency_map()
    td_saliency_map = compute_top_down_saliency_map(bottom_up_instance,weights)
    saliency_map = _t * td_saliency_map  + (1. - _t) * bu_saliency_map
    return saliency_map

def compute_top_down_saliency_map(bottom_up_instance, weights):
    """
    Parameter
    --------
    bottom_up_instance : bottom_up_part
        instance of a bottom_up_part sub class with data 
    weights : np.array
        trained weights
    Returns
    -------
    top_down_saliency map : np.array
    """
    feature_maps = bottom_up_instance.get_feature_maps()
    con_maps = bottom_up_instance.get_conspicous_maps()

    excitation_map_list = []
    inhibition_map_list = []

    for i, _map in enumerate(feature_maps + con_maps):
        if weights[i] == 0.:
            continue
        if weights[i] > 1.:
            excitation_map_list.append(weights[i] * _map)
        elif weights[i] < 1.:
            inhibition_map_list.append((1. / weights[i]) * _map)

    #create excitation_map
    excitation_map = sum(excitation_map_list) #all the featuremaps and conmaps should have the same scales...

    #create inhibition_map
    inhibition_map = sum(inhibition_map_list)

    td_saliency_map = excitation_map - inhibition_map
    td_saliency_map = np.where(td_saliency_map < 0, 0, td_saliency_map)
    
    return td_saliency_map