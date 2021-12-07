from enum import Enum
from typing import Iterator
import cv2


class PyramideType(Enum):
    """
    Enum to controll how the Pyrmide scales the image down
    """
    GAUSSIAN = 0
    LAPLACIAN = 1

class PyramideItter(Iterator):
    """
    Itterator that yields rescaled image at each scale
    yiels current layer of the pyramide
    """

    #memory
    __slots__ = '_img', '_layers', '_p_type', '_last_layer', '_i'

    def __init__(
        self,
        img,
        layers:int=4,
        p_type:PyramideType=PyramideType.GAUSSIAN
    ) -> None:
        """
        img -- cv2 image of any type
        layers -- height of the pyramide
        """
        self._img = img
        #store layer count
        self._layers = layers
        #gaussian or laplacian
        self._p_type = p_type
        #store the last result
        self._last_layer = None
        #count itteration
        self._i = 0

    def __next__(self):
        #increase itteration counter
        self._i += 1

        #check if done
        if self._i > self._layers:
            raise StopIteration

        #return a copy of the image in first itteration
        if self._last_layer is None:
            self._last_layer = self._img.copy()
            #return the image if mode is gaussian
            #if mode is laplaccian do the laplaccian calculations
            if self._p_type is PyramideType.GAUSSIAN:
                return self._last_layer

        #else: downscale image acordingly
        if self._p_type is PyramideType.GAUSSIAN:
            #simply use cv2 method
            self._last_layer = cv2.pyrDown(self._last_layer)

        elif self._p_type is PyramideType.LAPLACIAN:
            #use cv2 method and do laplaccian calculation
            #Code used as reference:
            #https://stackoverflow.com/questions/59103350/opencv-laplacian-pyramid-size-not-correct
            target_size = (self._last_layer.shape[1], self._last_layer.shape[0])
            layer_copy = self._last_layer.copy()
            self._last_layer = cv2.pyrDown(self._last_layer)
            gaussian_expanded = cv2.pyrUp(self._last_layer, dstsize=target_size)
            return cv2.subtract(layer_copy, gaussian_expanded)

        return self._last_layer

    def __len__(self):
        return self._layers


if __name__ == "__main__":
    TEST_IMG_PATH = "/Users/hanslehmann/Desktop/rocky.png"
    test_img = cv2.imread(TEST_IMG_PATH)
    for i,scale_img in enumerate(PyramideItter(test_img,p_type=PyramideType.LAPLACIAN)):
        cv2.imshow(f"scale {i}",scale_img)
        cv2.waitKey()
