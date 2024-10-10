import torch
import torch.nn as nn
import torch.optim as optim

import sklearn.metrics as metrics
import numpy as np

def pretrain_model(model, train_loader, epoch, num_epochs, crit1, crit2, optimizer, device, log_every=100):
    _ = model.train()

    model.to(device)
    losses = []

    for i, tuple_data in enumerate(train_loader):
        optimizer.zero_grad()

        data = tuple_data[:-1]
        label = tuple_data[-1].to(device)

        for d in data:
            img = d[0].to(device)
            c = d[1].to(device)
        
            _, input, output, pred = model.forward(img, c)

            mse = crit1(input, output)
            ce = crit2(pred, label)

            loss = mse + ce
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)

            if (i % log_every == 0) & (i > 0):
                print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]\
                        | avg train loss : {4} '''.
                    format(
                        epoch + 1,
                        num_epochs,
                        i,
                        len(train_loader),
                        np.round(np.mean(losses), 4),
                    )
                    )
    train_loss_epoch = np.round(np.mean(losses), 4)
    return train_loss_epoch

def train_model(model, train_loader, epoch, num_epochs, crit, optimizer, writer, device, log_every=100):
    _ = model.train()

    model.to(device)
    y_preds = []
    y_trues = []
    losses = []

    for i, tuple_data in enumerate(train_loader):
        optimizer.zero_grad()

        label = tuple_data[-1].to(device)
        data = tuple_data[:-1]

        for d in data:
            d[0] = d[0].to(device)
            d[1] = d[1].to(device)
            prediction = model.forward([d])
            loss = crit(prediction, label)
            loss_value = loss.item()
            losses.append(loss_value)
            probas = torch.sigmoid(prediction)
            y_trues.append(int(label[0]))
            y_preds.append(probas[0].item())
        
        prediction = model.forward(data)

        loss = crit(prediction, label)
        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)
        y_preds.append(probas[0].item())
        y_trues.append(int(label[0]))

        loss.backward()
        optimizer.step()

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]\
                    | avg train loss : {4} | train auc : {5}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4)
                  )
                  )
    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)
    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, crit, writer, current_lr, device, log_every=20):
    _ = model.eval()
    
    #for param in model.pretrained_model.parameters():
        #param.requires_grad = False

    model.to(device)

    y_trues = []
    y_preds = []
    losses = []

    with torch.no_grad():
        for i, tuple_data in enumerate(val_loader):

            label = tuple_data[-1].to(device)
            data = tuple_data[:-1]

            for d in data:
                d[0] = d[0].to(device)
                d[1] = d[1].to(device)

                prediction = model.forward([d])
                loss = crit(prediction, label)
                loss_value = loss.item()
                losses.append(loss_value)
                probas = torch.sigmoid(prediction)
                y_trues.append(int(label[0]))
                y_preds.append(probas[0].item())

            
            prediction = model.forward(data)

            loss = crit(prediction, label)
            loss_value = loss.item()
            losses.append(loss_value)

            probas = torch.sigmoid(prediction)
            y_trues.append(int(label[0]))
            y_preds.append(probas[0].item())

            try:
                auc = metrics.roc_auc_score(y_trues, y_preds)
            except:
                auc = 0.5

            writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
            writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)
            
            if (i % log_every == 0) & (i > 0):
                print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss: {4}  | val auc : {5}'''.
                    format(
                        epoch + 1,
                        num_epochs,
                        i,
                        len(val_loader),
                        np.round(np.mean(losses), 4),
                        np.round(auc, 4)
                    )
                    )

    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    return val_loss_epoch, val_auc_epoch

def test_model(model, test_loader, device):
    _ = model.eval()
    model.to(device)

    y_trues = []
    y_preds = []

    with torch.no_grad():
        for i, tuple_data in enumerate(test_loader):

            label = tuple_data[-1].to(device)
            data = tuple_data[:-1]

            for d in data:
                d[0] = d[0].to(device)
                d[1] = d[1].to(device)
            
            prediction = model.forward(data)
            probas = torch.sigmoid(prediction)

            y_trues.append(int(label[0]))
            y_preds.append(probas[0].item())            

    return y_preds, y_trues

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']