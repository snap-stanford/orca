import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer):
    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    ce = MarginLoss(m=-1*m)
    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        
        ((ux, ux2), _) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        labeled_len = len(target)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        output, feat = model(x)
        output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)
        
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        pos_pairs = []
        target_np = target.cpu().numpy()
        
        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx)
        
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        entropy_loss = entropy(torch.mean(prob, 0))
        
        loss = - entropy_loss + ce_loss + bce_loss

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tf_writer.add_scalar('loss/bce', bce_losses.avg, epoch)
    tf_writer.add_scalar('loss/ce', ce_losses.avg, epoch)
    tf_writer.add_scalar('loss/entropy', entropy_losses.avg, epoch)


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100
    else:
        warnings.warn('Dataset is not listed')
        return

    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.batch_size - labeled_batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    # First network intialization: pretrain the RotNet network
    model = models.resnet18(num_classes=num_classes)
    model = model.to(device)
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in range(args.epochs):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer)
        train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, mean_uncert, epoch, tf_writer)
        scheduler.step()


if __name__ == '__main__':
    main()
