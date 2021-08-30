import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils_own import *
from config import models_genesis_config
from torch import nn
import torch
import sys

sys.path.append('/home/sean/nnUNet')

from nnunet.training.learning_rate.poly_lr import poly_lr
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
#from torchsummary import summary
import random
import copy
from scipy.special import comb

from torch.optim import lr_scheduler
from optparse import OptionParser

#test

from torch.utils.data import Dataset, DataLoader

save_path = '/home/sean/ModelsGenesis/pretext_results/test'

print("torch = {}".format(torch.__version__))

seed = 3
random.seed(seed)
model_path = "pretrained_weights/Task111"
if not os.path.exists(model_path):
    os.makedirs(model_path)
logs_path = os.path.join(model_path, "Logs")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

config = models_genesis_config()
config.display()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('='*10+'device'+'='*10)
print(device)


class MVDataset(Dataset):
    def __init__(self, fold):
        self.dataname = os.path.join(config.DATA_DIR, "bat_" + str(config.scale) + "_s_"+str(config.input_rows)+"x"+str(config.input_cols)+"x"+str(config.input_deps)+"_")
        self.fold = fold
        foldname = self.dataname + str(self.fold) + '.npy'
        print("loading in fold "+str(self.fold))
        self.data = np.load(foldname)
        self.len = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        x,y = generate_pair(img, config, index)
        xc, yc = x.copy(), y.copy()
        return xc, yc

    def __len__(self):
        return self.len

# ################### configuration for model (MV)
num_input_channels=1
base_num_features = 32
num_classes = 7
net_num_pool_op_kernel_sizes=[[1, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2],[1, 2, 2]]
net_conv_kernel_sizes = [[1, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3],[3, 3, 3]]
net_numpool=len(net_num_pool_op_kernel_sizes)
conv_per_stage = 2
conv_op = nn.Conv3d
dropout_op = nn.Dropout3d
norm_op = nn.InstanceNorm3d
norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
# ##############################

model = Generic_UNet(num_input_channels, base_num_features, num_classes,
                                    net_numpool,
                                    conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

#print(model)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.patience * 0.6), gamma=0.5)
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000
intial_epoch =0
num_epoch_no_improvement = 0
sys.stdout.flush()
mean = -775.8457
std = 251.9326
if config.weights != None:
    checkpoint=torch.load(config.weights)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    intial_epoch=checkpoint['epoch']
    print("Loading weights from ",config.weights)
sys.stdout.flush()
for epoch in range(intial_epoch,config.nb_epoch):
    scheduler.step(epoch)
    model.train()
    print('current lr', optimizer.param_groups[0]['lr'])
    for fold in config.train_fold:
        train_dataset = MVDataset(fold)
        train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
        for i, data in enumerate(train_loader, 0):
            image, gt = data
            image = np.multiply(image,2000)-1000
            image = (image-mean)/std
            gt = np.repeat(gt,num_classes,axis=1)
            image=image.to(device, dtype=torch.float)
            gt=gt.to(device, dtype=torch.float)
            pred=model(image)
            pred=torch.sigmoid(pred)
            loss = criterion(pred,gt)

            # === Save ===
            if (i == 0):
                np_image = image.cpu().numpy()
                np_gt = gt.cpu().numpy()
                np_pred = pred.cpu().detach().numpy()
                np.save(os.path.join(save_path, 'mg_image_' + str(epoch)), np_image)
                np.save(os.path.join(save_path, 'mg_gt_' + str(epoch)), np_gt)
                np.save(os.path.join(save_path,'mg_pred_' + str(epoch)), np_pred)
            # ============

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (i + 1) % 5 ==0:
                print('Epoch [{}/{}], Fold [{}/{}], Step [{}/{}], Loss: {:.6f}'
                    .format(epoch + 1, config.nb_epoch, fold +1, len(config.train_fold), i + 1, len(train_loader.dataset)//config.batch_size, np.average(train_losses)))
                sys.stdout.flush()

    model.eval()
    print("validating....")
    with torch.no_grad():
        for fold in config.valid_fold:
            valid_dataset = MVDataset(fold)
            valid_loader = DataLoader(dataset=valid_dataset, batch_size=6, shuffle=True)
            for i, data in enumerate(valid_loader, 0):
                image, gt = data
                image = np.multiply(image,2000)-1000
                image = (image-mean)/std
                gt = np.repeat(gt,num_classes,axis=1)
                image= image.to(device, dtype=torch.float)
                gt= gt.to(device, dtype=torch.float)
                pred=model(image)
                pred=torch.sigmoid(pred)
                loss = criterion(pred,gt)
                valid_losses.append(loss.item())
                if (i + 1) % 5 == 0:
                    print('Validating: Epoch [{}/{}], Fold [{}/{}], Step [{}/{}], Loss: {:.6f}'
                          .format(epoch + 1, config.nb_epoch, fold+1 - len(config.train_fold), len(config.valid_fold), i + 1,
                                  len(valid_loader.dataset) // 6, np.average(valid_losses)))
                    sys.stdout.flush()

    #logging
    train_loss=np.average(train_losses)
    valid_loss=np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch+1,valid_loss,train_loss))
    train_losses=[]
    valid_losses=[]
    if valid_loss < best_loss:
        print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        #save model
        torch.save({
           'epoch':epoch + 1,
           'state_dict' : model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()
        },os.path.join(model_path, config.exp_name+".model"))
        print("Saving model ",os.path.join(model_path, config.exp_name+".model"))

    else:
        print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    
    if num_epoch_no_improvement == config.patience:
        print("Early Stopping")
        break
    sys.stdout.flush()
