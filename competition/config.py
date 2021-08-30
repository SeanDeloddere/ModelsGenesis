import os
import shutil

class models_genesis_config:
    DATA_DIR = "/home/sean/ModelsGenesis/generated_cubes/Task111"
    nb_epoch = 1000
    patience = 5
    lr = 1e-1
    #train_fold = [0]
    train_fold=[0,1,2,3,4,5,6]
    #valid_fold=[7]
    valid_fold=[7,8,9]
    #test_fold=[7,8,9]
    hu_max = 250.0 #250 for MV data
    hu_min = -250.0 #250 for MV data
    def __init__(self, 
                 note="",
                 data_augmentation=True,
                 input_rows=96,
                 input_cols=96,
                 input_deps=32,
                 batch_size= 2, #any higher GPU goes OOM
                 weights=None,
                 nb_class=7,
                 nonlinear_rate=0.9, # was 0.9
                 paint_rate=0.9,
                 outpaint_rate=0.8,
                 rotation_rate=0.0,
                 flip_rate=0.4,
                 local_rate=0.5, # was 0.5
                 verbose=1,
                 scale=32,
                ):
        self.exp_name = "Task111_2"
        self.data_augmentation = data_augmentation
        self.input_rows, self.input_cols = input_rows, input_cols
        self.input_deps = input_deps
        self.batch_size = batch_size
        self.verbose = verbose
        self.nonlinear_rate = nonlinear_rate
        self.paint_rate = paint_rate
        self.outpaint_rate = outpaint_rate
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.rotation_rate = rotation_rate
        self.flip_rate = flip_rate
        self.local_rate = local_rate
        self.nb_class = nb_class
        self.scale = scale
        self.weights = weights

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")