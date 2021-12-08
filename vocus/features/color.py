from .feature import Feature
from .feature_helper import get_center_surround_pyramide_feature_map

import numpy as np
import math

class Color(Feature):
    def __init__(self, img) -> None:
        super().__init__()
        _channel_b = img[:,:,0]
        _channel_g = img[:,:,1]
        _channel_r = img[:,:,2]
        _channel_y = self._get_yellow_channel(img)

        self._feature_b = get_center_surround_pyramide_feature_map(_channel_b)
        self._feature_g = get_center_surround_pyramide_feature_map(_channel_g)
        self._feature_r = get_center_surround_pyramide_feature_map(_channel_r)
        self._feature_y = get_center_surround_pyramide_feature_map(_channel_y)

    def _get_yellow_channel(self,lab_img):
        #TODO: Optimize
        _shape = lab_img.shape[:2]
        _r = np.array([255,127])
        y_channel = np.zeros(shape=_shape,dtype=np.uint8)
        for i in range(lab_img.shape[0]):
            for j in range(lab_img.shape[1]):
                _pixel_values = lab_img[i,j,1:]
                y_channel[i,j] = math.sqrt(
                    (_pixel_values[0] - _r[0]) ** 2
                    + (_pixel_values[1] - _r[1]) ** 2
                )
        return y_channel

    def get_feature_maps(self) -> list:
        return [
            self._feature_b,
            self._feature_g,
            self._feature_r,
            self._feature_y
        ]
