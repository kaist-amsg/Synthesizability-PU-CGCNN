import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import pickle
import csv
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from cgcnn.data_PU_learning import CIFData
from cgcnn.data_PU_learning import collate_pool, bootstrap_aggregating
from cgcnn.model_PU_learning import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal gated neural networks')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--modeldir', type=str, metavar='N',
                    help='Model directory')
parser.add_argument('--bag', default=50, type=int, metavar='N',
                    help='Bagging size of PU training')
parser.add_argument('--graph', type=str, metavar='N',
                    help='Folder name for preloaded crystal graph files')
parser.add_argument('--cifs', type=str, metavar='N',
                    help='Folder name containing cif files and id_prop.csv file')

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()


def preload(preload_folder, id_prop_file):
    data = []
    with open(id_prop_file) as g:
        reader = csv.reader(g)
        cif_list = [row[0] for row in reader]

    for cif_id in cif_list:
        with open(preload_folder+'/'+cif_id+'.pickle', 'rb') as f:
            data.append(pickle.load(f))

    return data


def main():
    global args, model_args, best_mae_error 

    model_dir = args.modeldir
    graph_dir = args.graph
    idprop_file_path = os.path.join(args.cifs, 'id_prop.csv')

    # Loop for all models
    for i in tqdm(range(1, args.bag+1)):
        collate_fn = collate_pool
        dataset_test = preload(preload_folder = graph_dir, id_prop_file = idprop_file_path)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)

        modelpath = os.path.join(model_dir, 'checkpoint_bag_'+str(i)+'.pth.tar')

        if os.path.isfile(modelpath):
            model_checkpoint = torch.load(modelpath,
                                          map_location=lambda storage, loc: storage)
            model_args = argparse.Namespace(**model_checkpoint['args'])
        else:
            print("=> no model params found at '{}'".format(modelpath))

        # build model
        structures, _, _ = dataset_test[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True)


        if args.cuda:
            model.cuda()

        # define loss func and optimizer
        criterion = nn.NLLLoss()

        normalizer = Normalizer(torch.zeros(3))

        # optionally resume from a checkpoint
        if os.path.isfile(modelpath):
            checkpoint = torch.load(modelpath,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            normalizer.load_state_dict(checkpoint['normalizer'])
        else:
            print("=> no model found at '{}'".format(modelpath))

        validate(test_loader, model, criterion, normalizer, i, test=True)


    bootstrap_aggregating(args.bag, prediction=True)


def validate(val_loader, model, criterion, normalizer, modelnum, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])

        target_normed = target.view(-1).long()
        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if test:
            test_pred = torch.exp(output.data.cpu())
            test_target = target
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target, test=False)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not test and i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                  'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                  'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   accu=accuracies, prec=precisions, recall=recalls,
                   f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results_prediction_'+str(modelnum)+'.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
        print(' {star} Recall(TPR) {recall.avg:.3f}'.format(star=star_label,
                                                 recall=recalls))
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
    return auc_scores.avg, recalls.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target, test):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        if not test:
            auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    if test:
        return accuracy, precision, recall, fscore
    else:
        return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()

