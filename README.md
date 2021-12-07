# pyVOCUS
unofficial python implementation of 'VOCUS: A Visual Attention System for Object Detection and Goal-Directed Search'[[1]](#1).

## Usage
### Bottom-up
```python
import cv2
from vocus.bottom_up.bottom_up_part import bottom_up_part_default
img = cv2.imread(...)
bottom_up_instance = bottom_up_part_default(img)
saliency_map = bottom_up_instance.get_saliency_map()
```
### Top-down
wip...
### Sensor Fusion
You can write your own features by createing a custom Feature-subclass (vocus/features/feature.py) and a custom bottom_up_part-subclass (vocus/bottom_up_part.py).

## References
<a id="1">[1]</a> 
Frintrop, Simone & Hertzberg, J.. (2006). VOCUS: A Visual Attention System for Object Detection and Goal-Directed Search. 10.1007/11682110. 
