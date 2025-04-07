import os
import time
import torch
from tqdm import tqdm

def Eval_mae(pred, target, method='improve', dataset='PCSOD', cuda=True):
    print('eval[MAE]:{} dataset with {} method.'.format(dataset, method))
    with torch.no_grad():
        if cuda:
            pred = pred.cuda()
            target = target.cuda()
        mea = torch.abs(pred - target).mean()
        return mea.item()


def adap_fmeasure(pred, target, method='improve', dataset='PCSOD', cuda=True):
    print('eval[adaptive FMeasure]:{} dataset with {} method.'.format(dataset, method))
    beta2 = 0.3
    with torch.no_grad():
        if cuda:
            pred = pred.cuda()
            target = target.cuda()
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        pred = (pred >= th).float()
        tp = (pred * target).sum()
        prec, recall = tp / (pred.sum() + 1e-20), tp / (target.sum() + 1e-20)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0
        return f_score.item()


def Eval_fmeasure(pred, target, method='improve', dataset='PCSOD', cuda=True):
    print('eval[mean FMeasure and max FMeasure]:{} dataset with {} method.'.format(dataset, method))
    beta2 = 0.3

    with torch.no_grad():
        if cuda:
            pred = pred.cuda()
            gt = target.cuda()
        prec, recall = _eval_pr(pred, gt, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        return f_score.mean().item()


def adap_Emeasure(pred, target, method='improve', dataset='PCSOD', cuda=True):
    print('eval[adaptive EMeasure]:{} dataset with {} method.'.format(dataset, method))
    with torch.no_grad():
        if cuda:
            pred = pred.cuda()
            target = target.cuda()
        th = 2 * pred.mean()
        if th > 1:
            th = 1
        y_pred_th = (pred >= th).float()
        fm = y_pred_th - y_pred_th.mean()
        target = target - target.mean()
        align_matrix = 2 * target * fm / (target * target + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4

        avg_e = torch.sum(enhanced) / (target.numel() - 1 + 1e-20)
        return avg_e.item()


def Eval_Emeasure(pred, target, method='improve', dataset='PCSOD', cuda=True):
    print('eval[mean EMeasure and max EMeasure]:{} dataset with {} method.'.format(dataset, method))
    with torch.no_grad():
        scores = 0
        if cuda:
            pred = pred.cuda()
            target = target.cuda()
        scores += _eval_e(pred, target, 255)
        return scores.mean().item()


def Eval_IoU(pred, target, method='improve', dataset='PCSOD', cuda=True):
    print('eval[IoU]:{} dataset with {} method.'.format(dataset, method))
    with torch.no_grad():
        scores = 0
        if cuda:
            pred = pred.cuda()
            target = target.cuda()
        scores += _eval_iou(pred, target, 255)
        return scores.mean().item()


def LOG(self, output):
    with open(self.logfile, 'a') as f:
        f.write(output)


def _eval_e(y_pred, y, num, cuda=True):
    if cuda:
        score = torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        score = torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_pred_th = (y_pred >= thlist[i]).float()
        fm = y_pred_th - y_pred_th.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
    return score


def _eval_pr(y_pred, y, num, cuda=True):
    if cuda:
        prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall


def _eval_iou(pred, gt, num, cuda=True):
    if cuda:
        score = torch.zeros(num).cuda()
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
    else:
        score = torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        pred_th = (pred >= thlist[i]).float()
        score[i] = torch.sum((pred_th == 1) & (gt == 1)) / torch.sum((pred_th == 1) | (gt == 1))

    return score
