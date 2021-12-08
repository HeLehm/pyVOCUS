"""
Python implementation of: Intensity Feature

Underlying paper:
TY  - BOOK
AU  - Frintrop, Simone
AU  - Hertzberg, J.
PY  - 2006/01/01
SN  - 978-3-540-32759-2
T1  - VOCUS: A Visual Attention System for Object Detection and Goal-Directed Search
VL  - 2
DO  - 10.1007/11682110
JO  - Fraunhofer IAIS

autor (of this file) Hergen Lehmann
"""
import cv2
from .feature_helper import get_center_surround_pyramide_feature_map
from .feature import Feature


class Intensity(Feature):
    """
    The Intesnity feature
    produces I'' I' and I
    """
    def __init__(self, img) -> None:
        """
        -- img -- cv2 img in bgr mode
        """
        super().__init__() #does nothing so far...
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #on, off
        self._feature_maps = [
            get_center_surround_pyramide_feature_map(gray_img),
            get_center_surround_pyramide_feature_map(~gray_img)
        ]

    def get_feature_maps(self):
        """
        returns I'_on and I
        returns [on_map,off_map]
        """
        return self._feature_maps
