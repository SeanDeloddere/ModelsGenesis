from os import listdir, mkdir, makedirs
from os.path import join, isdir, exists
import numpy as np

raw_folder = 'MV_raw/Task111'
npy_folder = 'MV_npys/Task111'

if not exists(npy_folder):
    makedirs(npy_folder)

i = 0
for filename in listdir(raw_folder):
    image = np.load(join(raw_folder,filename), allow_pickle=True)["arr_0"]
    # image = image.transpose(2,1,0) ALREADY TRANSPOSED IN npz2nifti!!
    subset = "subset"+str(i%10)
    destination = join(npy_folder, subset)
    if not isdir(destination):
        mkdir(destination)
    np.save(join(destination, filename[:-4]),image)
    i += 1