import cv2
from .feature import Feature
from .feature_helper import create_orientation_feature_maps, across_scale_add,deg_in_rad

class Orientation(Feature):
    """
    The Orientation feature
    produces O'' O' and O
    """
    def __init__(self, img) -> None:
        super().__init__()
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self._theta_0 = across_scale_add(create_orientation_feature_maps(gray_img,0))
        self._theta_45 = across_scale_add(create_orientation_feature_maps(gray_img,deg_in_rad(45)))
        self._theta_90 = across_scale_add(create_orientation_feature_maps(gray_img,deg_in_rad(90)))
        self._theta_135 = across_scale_add(create_orientation_feature_maps(gray_img,deg_in_rad(135)))

    def get_feature_maps(self) -> list:
        """
        returns [theta_0, theta_45, theta_90, theta_135]
        """
        return [
            self._theta_0,
            self._theta_45,
            self._theta_90,
            self._theta_135
        ]
