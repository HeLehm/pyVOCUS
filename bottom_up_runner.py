from util import load_rgb_img, plot_imgs, get_dot_img_test_path
from vocus.bottom_up.bottom_up_part import bottom_up_part_default

if __name__ == "__main__":
    TEST_PATH = get_dot_img_test_path()
    img = load_rgb_img(TEST_PATH)
    _b =  bottom_up_part_default(img)

    for i in range(5):
        _s = _b.get_saliency_map()
        _mrs = _b.get_next_FOA()
        _sf = _b._FOA_S
        plot_imgs([_s,_sf,_mrs])
