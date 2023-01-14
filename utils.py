import skimage
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import shutil
import pathlib
from tqdm import tqdm
from sklearn.decomposition import PCA

################################################## BUILD DATASET ####################################################

def read_file(filename):
    return [line.split('\n')[0] for line in open(filename).readlines()]

def read_landmarks(filename):
    landmarks = [ line.split('\n')[0].split(" ")[:2] for line in open(filename).readlines()][:]
    return np.float32(landmarks)

def get_bbox(landmarks):
    _min_x, _min_y =  np.min(landmarks, axis = 0)
    _max_x, _max_y =  np.max(landmarks, axis = 0)
    height = _max_y - _min_y
    width  = _max_x - _min_x
    # +30% of the face width and height
    x_min, y_min , x_max , y_max = _min_x - 0.15*width,  _min_y - 0.15*height,  _max_x + 0.15*width, _max_y + 0.15*height
    return np.int0([x_min, y_min , x_max , y_max])



## Loop throught the datasets

def build_trainset(dataset_path, trainset, images_train, landmarks_train):
    im_train = trainset / "images"
    ld_train = trainset / "landmarks"
    source_train = trainset / "source_train.txt"
    # create directories
    im_train.mkdir()
    ld_train.mkdir()
    source_train.touch()
    source_file = open(source_train, "w")
    for ind, image_filename, landmarks_filename in zip(tqdm(range(0, len(images_train)), ncols = 100, desc ="Build train\t"), images_train, landmarks_train):
        # image
        image = skimage.io.imread( dataset_path / image_filename)
        shape  = image.shape
        # landmarks
        landmarks = read_landmarks( dataset_path / landmarks_filename)
        ld = landmarks
        # bbox
        x_min, y_min , x_max , y_max = get_bbox(landmarks)
        # resize bbox
        x_min, y_min , x_max , y_max = max(x_min, 0), max(y_min, 0), min(x_max, shape[1]), min(y_max, shape[0])
        width_bbox = x_max - x_min 
        height_bbox = y_max - y_min
        # crop and resize to 128 x 128
        crop = skimage.transform.resize(image[y_min : y_max, x_min : x_max], (128,128))
        # rescale landmarks
        landmarks = (landmarks - [x_min, y_min])@[[128/width_bbox, 0], [0, 128/height_bbox]] 
        # Save image and landmarks
        skimage.io.imsave( im_train / f"image_train_{ind:04}.png",(crop/ crop.max() * 255).astype(np.uint8))
        np.save( ld_train / f"landmarks_train_{ind:04}.npy", landmarks)
        # write in source file
        source_file.write(f'{image_filename}\t{landmarks_filename}\t{ind:04}\n')
    source_file.close()

def build_testset(dataset_path, testset, images_test, landmarks_test):
    im_test = testset / "images"
    ld_test = testset / "landmarks"
    source_test = testset / "source_test.txt"
    # create directories
    im_test.mkdir()
    ld_test.mkdir()
    source_test.touch()
    source_file = open(source_test, "w")
    for ind, image_filename, landmarks_filename in zip(tqdm(range(0, len(images_test)), ncols = 100, desc ="Build test\t"), images_test, landmarks_test):
        # image
        image = skimage.io.imread( dataset_path / image_filename)
        shape  = image.shape
        # landmarks
        landmarks = read_landmarks( dataset_path / landmarks_filename)
        ld = landmarks
        # bbox
        x_min, y_min , x_max , y_max = get_bbox(landmarks)
        # resize bbox
        x_min, y_min , x_max , y_max = max(x_min, 0), max(y_min, 0), min(x_max, shape[1]), min(y_max, shape[0])
        width_bbox = x_max - x_min 
        height_bbox = y_max - y_min
        # crop and resize to 128 x 128
        crop = skimage.transform.resize(image[y_min : y_max, x_min : x_max], (128,128))
        # rescale landmarks
        landmarks = (landmarks - [x_min, y_min])@[[128/width_bbox, 0], [0, 128/height_bbox]] 
        # Save image and landmarks
        skimage.io.imsave( im_test / f"image_test_{ind:04}.png",(crop/ crop.max() * 255).astype(np.uint8))
        np.save( ld_test / f"landmarks_test_{ind:04}.npy", landmarks)
        # write in source file
        source_file.write(f'{image_filename}\t{landmarks_filename}\t{ind:04}\n')
    source_file.close()
    

def build_dataset(dataset_path):
    # Create dataset directory
    dataset  = pathlib.Path(r'dataset/')
    if  os.path.exists(dataset) :
        shutil.rmtree(dataset)
    dataset.mkdir()

    # Read train images & landmarks
    images_paths = read_file(dataset_path / '300w_train_images.txt')
    landmarks_paths = read_file(dataset_path / '300w_train_landmarks.txt')

    trainset = dataset / "trainset"
    trainset.mkdir()
    build_trainset(dataset_path, trainset, images_paths, landmarks_paths)

    images_paths = read_file(dataset_path / 'helen_testset.txt') + read_file( dataset_path / 'ibug.txt') + read_file(dataset_path / 'lfpw_testset.txt')
    landmarks_paths = read_file(dataset_path / 'helen_testset_landmarks.txt') + read_file(dataset_path / 'ibug_landmarks.txt') + read_file(dataset_path / 'lfpw_testset_landmarks.txt')

    testset = dataset / "testset"
    testset.mkdir()
    build_testset(dataset_path, testset, images_paths, landmarks_paths)
    
    # checksum
    checksum = dataset / "checksum.txt"
    checksum.touch()
    checksum_ = open(checksum, "w")
    checksum_.write("On est des tubes on est pas des pots!")
    checksum_.close()
    
######################################################################################################################
        
def plot_samples(images, landmarks, n = 5):
    """ show images of the dataset with their annotated landmarks"""
    # Figure params
    max_faces_per_row = 5
    rows, columns = n//max_faces_per_row + 1 , min(n, max_faces_per_row)
    figure_width  = min(4*columns, 20)
    figure_height = 4*rows
    fig = plt.figure(figsize=(figure_width, figure_height))
    fig.tight_layout()
    # Choose 'n' random people in the set
    _ind = np.random.choice(len(images), n, replace = False)
    for i, ind in enumerate(_ind):
        ax =  fig.add_subplot(rows, columns, i + 1)
        # plot image
        ax.imshow(images[ind])
        # plot landmarks
        ldks = landmarks[ind]
        plt.scatter(*ldks.transpose(), s = 10)
        ax.axis(False)
    plt.show()
    
def plot_samples_regression(images, landmarks, _landmarks, delta , n = 5):
    """ Show the regression result"""
    max_faces_per_row = 5
    rows, columns = n//max_faces_per_row + 1 , min(n, max_faces_per_row)
    figure_width  = min(4*columns, 20)
    figure_height = 4*rows
    _ind = np.random.choice(len(images), n, replace = False)
    fig = plt.figure(figsize=(figure_width, figure_height))
    fig.tight_layout()
    for i, ind in enumerate(_ind):
        ldks = landmarks[ind].reshape(-1, 2).T
        _ldks = _landmarks[ind].reshape(-1, 2).T
        u, v = delta.T[ind].reshape(-1, 2).T
        ting = [_ldks[0] + u, _ldks[1] - v]
        ax =  fig.add_subplot(rows, columns, i + 1)
        ax.imshow(images[ind])
        plt.scatter(*ldks, s = 18, marker = '+', color = 'red')
        plt.scatter(*_ldks, s = 8, color = '#00ff7f')
        # invert the v component to addapt the inverted 'y' axis
        plt.quiver(*_ldks, u, -v, width = 5e-3)
        plt.axis('off')
    plt.show()
    

# create mean translation and rescaling
def create_transformation(mean_landmarks, n = 10, rotation = 10):
    """ Data augmentations """
    #  random translations on both axis
    translations = list(zip(40*np.random.rand(n) - 20, 40*np.random.rand(n) - 20 ))
    
    #  random rescaling    on both axis
    rescaling    = list(zip(0.40*np.random.rand(n) + 0.8, 0.4*np.random.rand(n) + 0.8 ))
    
    #  random rotations +/- 10°
    ang = rotation
    ang = ang / 180.0 * np.pi
    rotations    = list(2 * ang* np.random.rand(n)  - ang)
    # compute regression
    ld = []
    for s, t, r in zip(rescaling, translations, rotations):
        # if we disable the rotation parameters, we set 'r' to dont change the rotation matrix 
        l = mean_landmarks@[[s[0] * np.cos(r), - s[0] * np.sin(r) ], [ s[1] * np.sin(r), s[1] *  np.cos(r)]] + [t[0], t[1]]
        ld.append(l)
    return np.float32(ld)


def to_keypoints(landmark, size = 20) :
    """ Turn the landmarks coordinates into a Keypoint list"""
    return [ cv.KeyPoint(*ld, size = size) for ld in landmark] 



def testing(images, landmarks, mean_model, A0, R0):
    """ Test the learned regressor on the test set"""
    TESTSET_SIZE = len(images)
    # transform to keypoints
    _keypts = to_keypoints(mean_model)
    # extract descriptors
    descriptors = []
    for ind in tqdm(range(0,  TESTSET_SIZE), ncols = 100, desc ="Extract desc. ") :
        sift = cv.SIFT_create()
        kp, des = sift.compute(images[ ind ], _keypts)
        descriptors.append( des.flatten() )
    # project descriptors
    descriptors = A0 @ np.array(descriptors).T
    # Apply regressor
    Y0 = np.vstack(( descriptors, np.ones((1, TESTSET_SIZE))))
    return R0 @ Y0