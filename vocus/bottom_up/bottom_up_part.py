#Feature classes
from ..features.feature import Feature
from ..features.color import Color
from ..features.intensity import Intensity
from ..features.orientation import Orientation

#helper functions
from ..helper import uniqueness_weight, get_FAO

#other libs
import numpy as np


class bottom_up_part():
    """
    the bottom up part as descibed in the paper
    """
    def __init__(self,feature_types,**kwargs) -> None:
        """
        -- feature_types: list of subclasses of feature class
        -- **kwargs will be passed to all feature maps __init__ function
        NOTE: when using these standard Feature kwwags should have 'img' keyword
        """
        self._feature_instances : list(Feature) = []
        for feature_type in feature_types:
            self._feature_instances.append(
                feature_type(**kwargs)
            )
        self._S = None
        self._FOA_S = None

    def get_saliency_map(self):
        """
        returns the saliency map
        stored if it has been calculated before, else: calculate
        """
        if self._S is not None:
            return self._S
        return self._calculate_saliency_map()

    def get_conspicous_maps(self):
        return [
            x.get_conspicous_map() for x in self._feature_instances
        ]
    
    def get_feature_maps(self):
        return [
            f_map for f_maps in [
                f.get_feature_maps() for f in self._feature_instances
            ] for f_map in f_maps
        ]

    def _calculate_saliency_map(self):
        """
        calculates, stores and retruns the saliency map
        """
        conspicous_maps = [
            x.get_conspicous_map() for x in self._feature_instances
        ]
        weighted_sals = [
            x * uniqueness_weight(x) for x in conspicous_maps
        ]
        #find global max
        self._S =  np.add(*weighted_sals)
        return self._S


    def get_next_FOA(self, threshold = 0.25, dilation_kernel = (40,40)):
        """
        returns next FOA as MSR grayscale img
        muatates self._FOA_S
        """
        if self._FOA_S is None:
            self._FOA_S = self._S.copy()
        msr, self._FOA_S = get_FAO(
            self._FOA_S,
            threshold=threshold,
            dilation_kernel=dilation_kernel
        )
        return msr

class bottom_up_part_default(bottom_up_part):
    """
    default bottom up part with Intensity,Color,Orientation as features
    """
    def __init__(self,img, **kwargs) -> None:
        super().__init__(
            img = img,
            feature_types = [Intensity, Color, Orientation],
            **kwargs
        )