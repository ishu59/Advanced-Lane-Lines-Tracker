import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os, glob
import random
import pickle

try:
    from cv2 import cv2
except:
    import cv2

random.seed(5)

# pickle_save_path = './caliberation_data.pkl'
pickle_save_path = './camera_cal/caliberation_data.pkl'


def get_images_path_list(image_folder=None):
    if image_folder is None:
        image_folder = "./camera_cal"
    images_name = []
    for file in os.listdir(image_folder):
        if file.endswith(".jpg"):
            file_path = os.path.join(image_folder, file)
            # print(file_path)
            images_name.append(file_path)
    return images_name


def get_obj_img_pts(nx=9, ny=6, image_folder=None):
    obj_pt = np.zeros((nx * ny, 3), np.float32)
    obj_pt[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    object_points = []
    image_points = []
    images_name = get_images_path_list(image_folder)
    for im_path in images_name:
        img = mpimg.imread(im_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            object_points.append(obj_pt)
            image_points.append(corners)
    return object_points, image_points


def draw_corners(img, nx, ny):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img, cmap='gray')
    return img


def caliberate(object_points=None, image_points=None, img_path=None):
    if object_points is None or image_points is None:
        object_points, image_points = get_obj_img_pts()
    if img_path is None:
        images_list = get_images_path_list()
        img_path = random.choice(images_list)
    test_img = cv2.imread(img_path)
    img_size = (test_img.shape[1], test_img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points,
                                                       image_points,
                                                       img_size, None, None)
    data = {}
    data['mtx'] = mtx
    data['dist'] = dist
    data['objp'] = object_points
    data['imgp'] = image_points
    pickle.dump(data, open(pickle_save_path, 'wb'))
    print(f"data saved to pickle file: {pickle_save_path}")
    return data


def disp_compare_img(img, undist, index=0):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=10)
    name = f'output_images/caliber_undist_img/image_{index}.png'
    plt.savefig(fname=name)
    plt.show()



def caliberation_data():
    if os.path.exists(pickle_save_path):
        data = pickle.load(open(pickle_save_path, 'rb'))
        mtx = data['mtx']
        dist = data['dist']
    else:
        data = caliberate()
        mtx = data['mtx']
        dist = data['dist']
    return mtx, dist


def caliberate_from_pickle(img=None, disp=False):
    data = pickle.load(open(pickle_save_path, 'rb'))
    mtx = data['mtx']
    dist = data['dist']
    if img is None:
        imgp = './camera_cal/calibration4.jpg'
        img = cv2.imread(imgp)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if disp:
        disp_compare_img(img, undist)
    return undist


if __name__ == '__main__':
    # get_obj_img_pts()
    # caliberate(img_path='../test_images/test_image_dist.jpg')
    # caliberate(img_path='../camera_cal/calibration2.jpg')
    # caliberation_data()
    caliberate_from_pickle()
