#!/usr/bin/env python
# coding: utf-8

"""
for subset in `seq 0 9`
do
python -W ignore infinite_generator_3D.py \
--fold $subset \
--scale 32 \
--data /mnt/dataset/shared/zongwei/LUNA16 \
--save generated_cubes
done
"""

# In[1]:


import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import sys
import random
import numpy as np
from tqdm import tqdm
from optparse import OptionParser
from skimage.transform import resize
from scipy import ndimage
from skimage import measure
import copy

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--fold", dest="fold", help="fold of subset", default=None, type="int")
parser.add_option("--input_rows", dest="input_rows", help="input rows", default=96, type="int") #was 64
parser.add_option("--input_cols", dest="input_cols", help="input cols", default=96, type="int") #was 64
parser.add_option("--input_deps", dest="input_deps", help="input deps", default=32, type="int") #was 32
parser.add_option("--data", dest="data", help="the directory of LUNA16 dataset", default=None, type="string")
parser.add_option("--save", dest="save", help="the directory of processed 3D cubes", default=None, type="string")
parser.add_option("--scale", dest="scale", help="scale of the generator", default=32, type="int")
(options, args) = parser.parse_args()
fold = options.fold
#fold = 0

seed = 1
random.seed(seed)

assert options.data is not None
assert options.save is not None
assert options.fold >= 0 and options.fold <= 9

if not os.path.exists(options.save):
    os.makedirs(options.save)


class setup_config():
    hu_max = 250.0
    hu_min = -250.0

    def __init__(self,
                 input_rows=None,
                 input_cols=None,
                 input_deps=None,
                 scale=None,
                 DATA_DIR=None,
                 train_fold=[0, 1, 2, 3, 4],
                 valid_fold=[5, 6],
                 test_fold=[7, 8, 9],
                 ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold
        self.test_fold = test_fold

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = setup_config(input_rows=options.input_rows,
                      input_cols=options.input_cols,
                      input_deps=options.input_deps,
                      scale=options.scale,
                      DATA_DIR=options.data,
                      )
config.display()

# cuts away the dead regions around the body that hold no information
def generate_selectable_area(config, img_array):
    # Cutoff outliers and normalize
    img_array[img_array < config.hu_min] = config.hu_min
    img_array[img_array > config.hu_max] = config.hu_max
    img_array = 1.0 * (img_array - config.hu_min) / (config.hu_max - config.hu_min)

    # Take a deep copy to leave original img_array intact
    temp = copy.deepcopy(img_array)

    # Smoothen out to get rid of stray pixels and lines
    temp = ndimage.morphology.grey_erosion(temp, size=(5, 5, 5))
    temp = ndimage.morphology.grey_erosion(temp, size=(5, 5, 5))

    # Label pixels to group them together
    labels = ndimage.label(temp)[0]

    # Select the largest label (which is the body)
    properties = measure.regionprops(labels, img_array)
    largest_label = -1
    largest_area = 0
    for index in range(labels.max()):
        area = getattr(properties[index], 'area')
        if (area > largest_area):
            largest_label = index
            largest_area = area

    # Select the slices that contain the largest label (which is the body)
    obj = ndimage.find_objects(labels)[largest_label]
    print('='*10+' Selecting slices '+'='*10)
    print(obj)

    # make sure selectable area is not too small to generate cubes from
    if not (obj[0].stop - obj[0].start > config.input_rows):
        if obj[0].start + config.input_rows + 1 < img_array.shape[0]:
            obj = (slice(obj[0].start, obj[0].start + config.input_rows + 1), obj[1], obj[2])
        else:
            obj = (slice(obj[0].stop - config.input_rows - 1, obj[0].stop), obj[1], obj[2])
    if not (obj[1].stop - obj[1].start > config.input_cols):
        if obj[1].start + config.input_rows + 1 < img_array.shape[1]:
            obj = (obj[0], slice(obj[1].start, obj[1].start + config.input_cols + 1), obj[2])
        else:
            obj = (obj[0],slice(obj[1].stop - config.input_cols - 1, obj[1].stop),obj[2])
    if not (obj[2].stop - obj[2].start > config.input_deps):
        if obj[2].start + config.input_deps + 1 < img_array.shape[2]:
            obj = (obj[0], obj[1], slice(obj[2].start, obj[2].start + config.input_deps + 1))
        else:
            obj = (obj[0], obj[1], slice(obj[2].stop - config.input_deps - 1, obj[2].stop))

    selected_area = img_array[obj]
    print('original shape: '+str(img_array.shape))
    print('cropped shape: '+str(selected_area.shape))
    return selected_area

# generate cubes
def random_generator(config, img_array):
    img_array = generate_selectable_area(config, img_array)
    size_x, size_y, size_z = img_array.shape
    if size_z - config.input_deps < 0:
        return None

    slice_set = np.zeros((config.scale, config.input_rows, config.input_cols, config.input_deps), dtype=float)

    for num_pair in range(config.scale):
        start_x = random.randint(0, size_x - config.input_rows - 1)
        start_y = random.randint(0, size_y - config.input_cols - 1)
        start_z = random.randint(0, size_z - config.input_deps - 1)

        crop_window = img_array[start_x: start_x + config.input_rows,
                      start_y: start_y + config.input_cols,
                      start_z: start_z + config.input_deps]

        slice_set[num_pair] = crop_window

    return np.array(slice_set)

def get_self_learning_data(fold, config):
    slice_set = []
    for index_subset in fold:
        subset_path = os.path.join(config.DATA_DIR, "subset" + str(index_subset))
        file_list = os.listdir(subset_path)

        for img_file in tqdm(file_list):
            print(' ')
            print('name of scan: '+ img_file)
            img_array = np.load(os.path.join(subset_path,img_file))
            x = random_generator(config, img_array)
            if x is not None:
                slice_set.extend(x)

    return np.array(slice_set)


print(">> Fold {}".format(fold))
cube = get_self_learning_data([fold], config)
print("cube: {} | {:.2f} ~ {:.2f}".format(cube.shape, np.min(cube), np.max(cube)))
np.save(os.path.join(options.save,
                     "bat_" + str(config.scale) +
                     "_s" +
                     "_" + str(config.input_rows) +
                     "x" + str(config.input_cols) +
                     "x" + str(config.input_deps) +
                     "_" + str(fold) + ".npy"),
        cube,
        )