import os
import shutil
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim
import sklearn.metrics as metrics
from PIL import Image

from tensorboardX import SummaryWriter
from datetime import datetime

from utils import *
from dataloader import *
from HyperMM import *

def run(args):
    seed = args.seed
    torch.manual_seed(seed)

    log_root_folder = "./logs/"
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)
    
    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    if args.transform == 1 :
        transform_vgg16 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_vgg16 = None


    train_dataset = PairedDataset(args.path_data,
                                phase="train", transform=transform_vgg16)
    validation_dataset = PairedDataset(args.path_data,  
                                    phase="val", transform=transform_vgg16)
    test_dataset = PairedDataset(args.path_data, 
                              phase="test", transform=transform_vgg16)

    train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)

    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=-True, num_workers=4, drop_last=False)

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)


    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    """
    Pre training phase:
    """   

    print("Pre-training starts: ")

    feature_extractor = HyperMMPretrain()
    feature_extractor.to(device)

    optimizer_pretrain = optim.Adam(feature_extractor.parameters(), lr=args.lr_pretrain, weight_decay=0.0005)

    best_train_loss = float('inf')

    num_epochs_pretrain = args.epochs_pretrain
    iteration_change_loss = 0
    if bool(args.early_stopping):
        patience = args.patience
    else:
        patience = None

    t_start_training_pretrain = time.time()

    crit_pretrain1 = torch.nn.MSELoss()
    crit_pretrain2 = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs_pretrain):        
        t_start = time.time()

        train_loss = pretrain_model(
            feature_extractor, train_loader, epoch, num_epochs_pretrain, crit_pretrain1, crit_pretrain2, optimizer_pretrain, device)
            
        t_end = time.time()
        delta = t_end - t_start

        print("train loss : {0} | elapsed time {1} s".format(
            train_loss, delta))
        
        iteration_change_loss += 1
        print('-' * 30)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping of pre-training after {0} iterations without the decrease of the loss'.
                format(iteration_change_loss))
            break
    
    t_end_training_pretrain = time.time()
    print('Pre-training took {} s/{} min'.format(t_end_training_pretrain - t_start_training_pretrain, (t_end_training_pretrain - t_start_training_pretrain)/60))


    """
    Main training pase:
    """   
   
    print("Main training starts : ")

    net = HyperMMNet(feature_extractor=feature_extractor)
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    if bool(args.early_stopping):
        patience = args.patience
    else:
        patience = None
    log_every = args.log_every

    t_start_training = time.time()

    crit = torch.nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)
        
        t_start = time.time()

        train_loss, train_auc = train_model(
            net, train_loader, epoch, num_epochs, crit, optimizer, writer, device)

        val_loss, val_auc = evaluate_model(
            net, validation_loader, epoch, num_epochs, crit, writer, current_lr, device)

        
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()
            
        t_end = time.time()
        delta = t_end - t_start

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))


        iteration_change_loss += 1
        print('-' * 30)

        if (val_auc > best_val_auc) and (epoch >= 5):
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = "model_{}_val_auc_{:0.4f}_train_auc_{:0.4f}_epoch_{}.pth".format(args.prefix_name, val_auc, train_auc, epoch+1)
                for f in os.listdir('./models/'):
                    if args.prefix_name in f:
                        os.remove("./models/{}".format(f))
                torch.save(net, "./models/{}".format(file_name))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                format(iteration_change_loss))
            break

    if args.save_model == 0:
        torch.save(net, "./models/model_{}_val_auc_{:0.4f}_train_auc_{:0.4f}_final.pth".format(args.prefix_name, val_auc, train_auc)) 

    t_end_training = time.time()
    print('Training took {} s/ {} min'.format(t_end_training - t_start_training, (t_end_training - t_start_training)/60))

    """
    Testing phase
    """
    models_list = os.listdir("./models/")
    model_name = list(filter(lambda name: args.prefix_name in name, models_list))[0]
    model_path = "./models/{}".format(model_name)
    testing_model = torch.load(model_path)

    test_preds, test_trues = test_model(testing_model, test_loader, device)

    res = dict.fromkeys(['acc', 'auc', 'f1', 'precision', 'recall'])
    res['acc'] = metrics.accuracy_score(test_trues, np.round(test_preds))
    res['auc'] = metrics.roc_auc_score(test_trues, np.round(test_preds))
    res['f1'] = metrics.f1_score(test_trues, np.round(test_preds))
    res['precision'] = metrics.precision_score(test_trues, np.round(test_preds))
    res['recall'] = metrics.recall_score(test_trues, np.round(test_preds))

    res_df = pd.DataFrame(res, index=[0])
    res_df.to_csv('./results/{}.csv'.format(args.prefix_name))

    print('Test accuracy for HyperMM : ', metrics.accuracy_score(test_trues, np.round(test_preds)))

def parse_arguments():
     parser = argparse.ArgumentParser()
     parser.add_argument('--path_data', type=str, default="/data/chaptouk/ADNI_clean/pp")
     parser.add_argument('--prefix_name', type=str, required=True)
     parser.add_argument('--transform', type=int, choices=[0,1], default=1)
     parser.add_argument('--lr_scheduler', type=str, default='plateau', choices=['plateau', 'step'])
     parser.add_argument('--epochs', type=int, default=100)
     parser.add_argument('--epochs_pretrain', type=int, default=50)
     parser.add_argument('--lr', type=float, default=1e-4)
     parser.add_argument('--lr_pretrain', type=float, default=1e-4)
     parser.add_argument('--gamma', type=float, default=0.5)
     parser.add_argument('--flush_history', type=int, choices=[0,1], default=1)
     parser.add_argument('--save_model', type=int, choices=[0,1], default=1)
     parser.add_argument('--patience', type=int, default=10)
     parser.add_argument('--early_stopping', type=int, default=1, choices=[0,1])
     parser.add_argument('--log_every', type=int, default=100)
     parser.add_argument('--seed', type=int, default=7)
     args = parser.parse_args()
     return args

if __name__ == "__main__":
    args = parse_arguments()
    run(args)