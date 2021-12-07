from ..bottom_up.bottom_up_part import bottom_up_part_default
from ..helper import get_FAO
from .top_down_helper import learn_weights, search_with_weights

class TopDownExtension:
    def __init__(self,bottom_up_class, weights=None) -> None:
        self._weights = weights
        self._FAO_S = None
        self._bottom_up_class = bottom_up_class

    def reset_FAO(self):
        self._FAO_S = None
    
    def set_weights(self, new_weights):
        self._weights = new_weights

    def get_wieghts(self):
        return self._weights

    def learn(self,imgs,regions, **kwargs):
        self._weights = learn_weights(imgs,regions,self._bottom_up_class, **kwargs)

    def search(self, t, **kwargs):
        """
        creates and stores a fao as descibed in the paper in search mode
        NOTE: use function get_next_FAO to show results
        """
        assert self._weights is not None, "No Weights..."
        self._FAO_S = search_with_weights(
            _t = t,
            bottom_up_class = self._bottom_up_class,
            **kwargs
        )

    def get_next_FAO(self, threshold = 0.25, dilation_kernel = (30,30)):
        """
        returns next FOA as MSR grayscale img
        muatates self._FOA_S
        """
        msr, self._FOA_S = get_FAO(
            self._FOA_S,
            threshold=threshold,
            dilation_kernel=dilation_kernel
        )
        return msr
        

class TopDownExtensionDefault(TopDownExtension):
    def __init__(self, weights=None) -> None:
        super().__init__(bottom_up_class= bottom_up_part_default,weights=weights)
        
        
        
