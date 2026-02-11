import argparse
import os
import shutil
import time
import random
import warnings
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from data_utils import get_datasets
from sklearn.metrics import roc_auc_score
from models.sfocus import sfocus18
from PIL import Image
from datetime import datetime
import torch.nn.functional as F
import cv2
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', default='CheXpert' , help='ImageNet, CheXpert, NIH, MIMIC')
parser.add_argument('--ngpu', default= 1, type=int, metavar='G', help='number of gpus to use')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default= 40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--milestones', type=int, default=[50,100], nargs='+', help='LR decay milestones')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default= False, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS', help='input batch size for training (default: 64)')
parser.add_argument("--prefix", default="Result", type=str, required=False, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate',default=False, dest='evaluate', action='store_true', help='evaluation only')
best_prec1 = 0

global result_dir
global probs 
global gt    
global k
global best_validation_score
best_validation_score = 0

def main():


    global args, best_prec1
    global viz, train_lot, test_lot
    args = parser.parse_args()
    print ("args", args)
    base_path = args.base_path

    now = datetime.now()
    result_dir = os.path.join(base_path, "{}_{}H".format(now.date(), str(now.hour)))
    os.makedirs(result_dir, exist_ok=True)
    c = open(result_dir + "/config.txt", "w")
    c.write("plus: {}, depth: {}, dataset: {}, epochs: {}, lr: {}, momentum: {},  weight-decay: {}, seed: {}".format(args.plus, str(args.depth), args.dataset, str(args.epochs), str(args.lr), str(args.momentum),str(args.weight_decay), str(args.seed)))
    open(result_dir + "/validation_performance.txt", "w")
    open(result_dir + "/test_performance.txt", "w")


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    
    train_dataset, val_dataset, test_dataset, num_classes, unorm = get_datasets(args.dataset)
    model = sfocus18(args.dataset, args.model, num_classes, depth = args.depth, pretrained=False, plus=args.plus)

    if args.dataset == 'ImageNet':
        criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(base_path, '2024-03-14_14H/model.pth')))

    cudnn.benchmark = True

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True)
    if args.evaluate:
        validate(test_loader, model, criterion, unorm, -1, PATH)
        return
    PATH = os.path.join('./checkpoints/SF', args.dataset, args.prefix)
    os.makedirs(PATH, exist_ok=True)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    
    for epoch in range(args.start_epoch, args.epochs):

        if epoch == 0:
            prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, result_dir)
            prec1 = test(test_loader, model, criterion, unorm, epoch, PATH, result_dir)
        
        train(train_loader, val_loader, test_loader, model, criterion, optimizer, epoch, result_dir, PATH, unorm)

        prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, result_dir)
        prec1 = test(test_loader, model, criterion, unorm, epoch, PATH, result_dir)
        
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        

def train(train_loader,val_loader, test_loader, model, criterion, criterion2, optimizer, epoch, dir, PATH, unorm, mask_model = None):

    global result_dir
    global probs 
    global gt
    global k

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_loader_examples_num = len(train_loader.dataset)
    sigmoid =  nn.Sigmoid()

    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader):
        
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        
        target = target.cuda()
        inputs = inputs.cuda()
            
        output, _ = model(inputs, target)
        loss = criterion(output, target) + criterion2(sigmoid(output),target)
    
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        
        if i % 2000 == 0 and i != 0:
            tempk = k
            tempprobs = probs
            tempgt = gt
            prec1 = validate(val_loader, model, criterion, unorm, epoch, PATH, dir)
            prec1 = test(test_loader, model, criterion, unorm, epoch, PATH, dir)
            k = tempk
            probs = tempprobs
            gt = tempgt
            model.train()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    
    if args.dataset == 'ImageNet':
        wandb.log({
        "Epoch":epoch,
        "Train loss":losses.avg,
        "Train Top 1 ACC":top1.avg,
        "Train Top 5 ACC":top5.avg,
    })  

def vis_heatmaps(hmaps, inputs, unnorm, epoch, path):
    f_shape = hmaps[0].shape
    i_shape = inputs[0].shape
    img_tensors = []
    for idx, image in enumerate(inputs):
        hmap = hmaps[idx]
        if f_shape[0] == 1:
            hmap = torch.cat((hmap, torch.zeros(2, f_shape[1], f_shape[2])))
        hmap = (transforms.ToPILImage()(hmap)).resize((i_shape[1], i_shape[2]))
        pil_image = transforms.ToPILImage()(torch.clamp(unnorm(image), 0, 1))
        res = Image.blend(pil_image, hmap, 0.5)
        img_tensors.append(transforms.ToTensor()(res))
    save_image(img_tensors, '{}/{}.png'.format(path, epoch), nrow=8)

def validate(val_loader, model, criterion, unorm, epoch, PATH, dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    global probs 
    global gt
    global k
    global best_validation_score

    val_loader_examples_num = len(val_loader.dataset)

    model.eval()
    end = time.time()
    for i, (inputs, target) in enumerate(val_loader):
        cnt += 1
        target = target.cuda()
        inputs = inputs.cuda()

        output, saliency_maps  = model(inputs, target)
        loss = criterion(output, target)
        
        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if args.dataset == 'ImageNet':
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        wandb.log({
            "Valid loss":losses.avg,
            "Valid Top 1 ACC":top1.avg,
            "Valid Top 5 ACC":top5.avg,
        })   
        f = open(dir + "/performance.txt", "a")
        f.write(str(top1.avg.item()) + "\n")
        f.close()

    return top1.avg

def test(test_loader, model, criterion, unorm, epoch, PATH, dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    test_loader_examples_num = len(test_loader.dataset)

    model.eval()
    end = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        cnt += 1
        target = target.cuda()
        inputs = inputs.cuda()

        output, saliency_maps  = model(inputs, target)
        loss = criterion(output, target)

        prec1, prec5 = accuracy(args.dataset, output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    if args.dataset == 'ImageNet':
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5)) 
        f = open(dir + "/performance.txt", "a")
        f.write(str(top1.avg.item()) + "\n")
        f.close()

    return top1.avg

def get_distance(saliency_maps):
    
    cosine_similarity = 0

    for i in range(4):

        grad1 = get_cam(saliency_maps[0][i])
        grad2 = get_cam(saliency_maps[1][i])
        cosine_similarity += F.cosine_similarity(grad1, grad2)

    cosine_similarity /= 4

    return cosine_similarity

def measure_distance(dataloader):

    model = sfocus18(args.dataset, 1, depth = args.depth, pretrained=False, plus=args.plus)
    model.load_state_dict(torch.load(os.path.join('2024-03-14_14H/model.pth')))
    model.eval()

    all_distances = []
    all_input_pairs = []

    for i, (inputs, target) in enumerate(dataloader):
        target = target.cuda()
        inputs = inputs.cuda()

        output, saliency_maps  = model(inputs, target)
        distance = get_distance(saliency_maps)

        all_distances.append(distance)
        all_input_pairs.append(inputs)

    return all_input_pairs, all_distances

def save_checkpoint(state, is_best, path):
    filename='{}/checkpoint.pth.tar'.format(path)
    if is_best:
        torch.save(state, filename)


class AverageMeter(object):
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

def get_cam(grad_cam_map):

    grad_cam_map = grad_cam_map.unsqueeze(dim=0)
    grad_cam_map = F.interpolate(grad_cam_map, size=(150, 150), mode='bilinear', align_corners=False)
    map_min, map_max = grad_cam_map.min(), grad_cam_map.max()
    grad_cam_map = (grad_cam_map - map_min).div(map_max - map_min + 0.0000001).data

    grad_heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam_map.squeeze().cpu()), cv2.COLORMAP_JET)
    grad_heatmap = torch.from_numpy(grad_heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = grad_heatmap.split(1)
    grad_heatmap = torch.cat([r, g, b])

    return grad_heatmap


def accuracy(dataset, output, target, topk=(1,)):
    
    sigmoid = torch.nn.Sigmoid()
    res = []
    global probs 
    global gt
    global k
    
    if dataset == 'ImageNet':
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    
    elif dataset == 'CheXpert' or dataset == 'NIH':
        
        probs[k: k + output.shape[0], :] = output.cpu()
        gt[   k: k + output.shape[0], :] = target.cpu()
        k += output.shape[0] 
        
        preds = np.round(sigmoid(output).cpu().detach().numpy())
        targets = target.cpu().detach().numpy()
        test_sample_number = len(targets)* len(output[0])
        test_correct = (preds == targets).sum()
        
        res.append([test_correct / test_sample_number * 100])
        res.append([0])
    
    return res


if __name__ == '__main__':
    main()
