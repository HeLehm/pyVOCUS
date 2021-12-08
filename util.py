import cv2
import matplotlib.pyplot as plt
import os

def plot_con_feat(feature):
    plot_imgs(feature.get_feature_maps() + [feature.get_conspicous_map()])

def plot_imgs(imgs,title = "",save_path=None):
    if not isinstance(imgs,list):
        plt.imshow(imgs, cmap='gray',interpolation='none')
        if save_path is not None:
            print("save")
            plt.savefig(save_path,bbox_inches='tight')
        plt.show()
        
        return
    fig, axs = plt.subplots(len(imgs))
    fig.suptitle(title) 
    for i,img in enumerate(imgs):
        axs[i].imshow(img, cmap='gray',interpolation='none')
    if save_path is not None:
        print("save")
        plt.savefig(save_path,bbox_inches='tight')
    plt.show()
    
    

def load_img(path):
    #load
    img = cv2.imread(path)
    return img

def _get_test_dir():
    return os.path.join(
        os.path.dirname(__file__),
        "test_files"
    )

def get_dot_img_test_path():
    return os.path.join(
        _get_test_dir(),
        "vocus_dots.png"
    )


def get_top_down_train_imgs_and_regions():
    _test_dir = _get_test_dir()
    _filenames = ["train_top_down_1.png"]
    regions = [(200,300,200,300)]
    paths = [os.path.join(_test_dir,x)for x in _filenames]
    return paths,regions

def get_top_down_test_imgpath_and_region():
    _test_dir = _get_test_dir()
    region = (100,200,0,100)
    path = os.path.join(_test_dir,"test_top_down_1.png")
    return path,region