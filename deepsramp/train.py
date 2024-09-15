from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, classification_report
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from . import utils

def train_loop(dataloader, model, device, loss_fn, optimizer, scheduler, epoch):
    size = len(dataloader.dataset)
    scaler = GradScaler('cuda')
    
    for batch, (X, y) in enumerate(dataloader):
        X = [x.to(device) for x in X]
        y = y.to(device)
        # y = [i.to(device) for i in y]
        
        # with autocast():
        model.train()
        pred = model(*X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        scheduler.step(epoch + batch / (size // 128) )

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X[0])
            print(f"loss: {loss:>7f}  [{current:>6d}/{size:>6d}]", end=' ')

def test_loop(dataloader, model, device, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    preds = []
    ys = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = [x.to(device) for x in X]
            y = y.to(device)

            model.eval()
            pred = model(*X)
            test_loss += loss_fn(pred, y).item()
            
            preds += (torch.sigmoid(pred[:, -1]).tolist())
            ys += (y[:, -1].tolist())

    test_loss /= num_batches
    fpr, tpr, throc = roc_curve(ys, preds)
    aucc = auc(fpr, tpr)
    reca, prec, thprc = precision_recall_curve(ys, preds)
    auprcc = auc(prec, reca)
    mcc = matthews_corrcoef(ys, torch.tensor(preds) > 0.5)
    report = classification_report(ys, torch.tensor(preds) > 0.5, output_dict=True)
    
    print(f"[Test] ACC: {(100*report['accuracy']):>0.1f}%, Loss: {test_loss:>8f}, AUROC: {aucc:>.4f}, AUPRC: {auprcc:>.4f}, Size: {len(ys)}, Ratio: {(sum(ys)/len(ys)):.5f}")
    return test_loss, aucc, auprcc, fpr, tpr, throc, reca, prec, thprc, mcc, report

def train(model, loss_fn, optimizer, scheduler, train_dl, test_dl, epochs=15, device='cpu', ei=[0, 1]):
    for t in range(epochs):
        print(f"[Epoch {t+1:02d}]", end=' ')
        train_loop(train_dl, model, device, loss_fn, optimizer, scheduler, t)
        print()
    # return test_loop(test_dl, model, device, loss_fn, ei)

def pred_loop(dataloader, model, device='cpu'):
    preds = []
    ys = []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = [x.to(device) for x in X]
            y = y.to(device)
            model.eval()
            pred = model(*X)
            preds += (torch.sigmoid(pred[:, half_length, -1]).tolist())
            ys += (y[:, half_length, -1].tolist())
    return preds, ys


def nano_test_loop(dataloader, model, device, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct, size = 0, 0, 0

    preds = []
    ys = []
    with torch.no_grad():
        for X, y in dataloader:
            X = [x.to(device) for x in X]
            y = y.to(device)

            model.eval()
            pred = model(*X)
            test_loss += loss_fn(pred, y).item()
            # status = ((pred > 0.5) == y).type(torch.float)
            # correct += status.sum().item()
            # size += len(status)

            preds += (pred[0].tolist())
            ys += (y.tolist())

    test_loss /= num_batches
    # correct /= size
    fpr, tpr, throc = roc_curve(ys, preds)
    aucc = auc(fpr, tpr)
    recall, prec, thprc = precision_recall_curve(ys, preds)
    auprcc = auc(prec, recall)
    acc = accuracy_score(ys, torch.tensor(preds) > 0.5)
    rs = recall_score(ys, torch.tensor(preds) > 0.5)
    ps = precision_score(ys, torch.tensor(preds) > 0.5)
    print(f"[Test] ACC: {(100*acc):>0.1f}%, Loss: {test_loss:>8f}, AUROC: {aucc:>.4f}, AUPRC: {auprcc:>.4f}, Recall: {rs:>.4f}, Precision: {ps:>.4f}, Size: {len(ys)}")
    return correct, test_loss, aucc, auprcc, fpr, tpr, throc, recall, prec, thprc, preds, ys

def nano_pred_loop(dataloader, model, device='cpu'):
    preds = []
    ys = []
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = [x.to(device) for x in X]
            y = y.to(device)
            model.eval()
            pred = model(*X)
            preds += (pred[0].tolist())
            ys += y.tolist()
    return preds, ys


def dp_test_loop(dataloader, model, device, loss_fn):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    preds = []
    ys = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = [x.to(device) for x in X]
            y = y.to(device)

            model.eval()
            pred = model(*X)
            test_loss += loss_fn(pred, y).item()
            
            preds += (torch.sigmoid(pred).tolist())
            ys += (y[:, -1].tolist())

    test_loss /= num_batches
    fpr, tpr, throc = roc_curve(ys, preds)
    aucc = auc(fpr, tpr)
    reca, prec, thprc = precision_recall_curve(ys, preds)
    auprcc = auc(prec, reca)
    mcc = matthews_corrcoef(ys, torch.tensor(preds) > 0.5)
    report = classification_report(ys, torch.tensor(preds) > 0.5, output_dict=True)
    
    print(f"[Test] ACC: {(100*report['accuracy']):>0.1f}%, Loss: {test_loss:>8f}, AUROC: {aucc:>.4f}, AUPRC: {auprcc:>.4f}, Size: {len(ys)}, Ratio: {(sum(ys)/len(ys)):.5f}")
    return test_loss, aucc, auprcc, fpr, tpr, throc, reca, prec, thprc, mcc, report