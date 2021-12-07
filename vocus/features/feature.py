from .feature_helper import across_scale_add, min_max_scale
from ..helper import uniqueness_weight

import numpy as np

class Feature:
    """
    Superclass for Features
    """
    def __init__(self,img) -> None:
        self._conspicous_map = None

    def get_feature_maps(self) -> list:
        """
        Subclass should implement this and return a list of its feature maps
        """
        pass

    def get_conspicous_map(self):
        """
        returns the conspicous map for this feature
        """
        if self._conspicous_map is not None:
            return self._conspicous_map
        #find global max
        _max = max(
            [
                np.max(x) for x in self.get_feature_maps()
            ]
        )
        #apply uniqueness weights
        _weighted_feature_maps = [uniqueness_weight(x) * x for x in self.get_feature_maps()]
        #across_scale_add and normalize
        self._conspicous_map =  min_max_scale(
            across_scale_add(_weighted_feature_maps).astype(np.int),
            _range=(0,_max)
        ).astype(np.int)
        return self._conspicous_map
    