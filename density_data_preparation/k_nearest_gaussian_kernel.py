import scipy.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
from matplotlib import pyplot as plt

def gaussian_filter_density(img,points):
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of current image: ",img_shape,". Totally need generate ",len(points),"gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances = tree.query(points, k=4)

    print ('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += gaussian_filter(pt2d, sigma, mode='constant')
    print ('done.')
    return density

if __name__=="__main__":
    root = '/path/to/dataset'
    
    part_A_train = os.path.join(root,'train_data','images')
    part_A_test = os.path.join(root,'test_data','images')
    path_sets = [part_A_train,part_A_test]
    
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
        img= plt.imread(img_path)
        k = np.zeros((img.shape[0],img.shape[1]))
        points = mat["image_info"][0,0][0,0][0] 
        k = gaussian_filter_density(img,points)
        np.save(img_path.replace('.jpg','.npy').replace('images','ground_truth'), k)