from util import load_img, plot_imgs,  get_top_down_train_imgs_and_regions,get_top_down_test_imgpath_and_region
from vocus.top_down.top_down_extension import TopDownExtensionDefault

if __name__ == "__main__":
    #get train img
    img_paths, img_regions = get_top_down_train_imgs_and_regions()
    imgs = [load_img(x) for x in img_paths]

    #init class and lear weights
    _td = TopDownExtensionDefault()
    _td.learn(imgs, img_regions)

    #get test img
    _testpath, _ = get_top_down_test_imgpath_and_region()
    test_img = load_img(_testpath)

    #get prediction
    _td.search(test_img, 1.)
    msr = _td.get_next_FOA()

    #plt results
    plot_imgs([test_img,msr])

    
