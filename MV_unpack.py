from os import listdir, mkdir
from os.path import join, isdir
import numpy as np

raw_folder = 'MV_raw/Task105'
npy_folder = 'MV_npys/Task105'

i = 0
for filename in listdir(raw_folder):
    image = np.load(join(raw_folder,filename), allow_pickle=True)["arr_0"]
    image = image.transpose(2,1,0)
    subset = "subset"+str(i%10)
    destination = join(npy_folder, subset)
    if not isdir(destination):
        mkdir(destination)
    np.save(join(destination, filename[:-4]),image)
    i += 1