import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_imagenet as datasets
import torchvision.transforms as transforms
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

def train(args, model, device, train_loader, optimizer, m, labeled_len, epoch, tf_writer):
    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    ce = MarginLoss(m=-1*m)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')

    for batch_idx, ((x, x2), combined_target, idx) in enumerate(train_loader):
        
        target = combined_target[:labeled_len]
        x, x2, target = x.to(device), x2.to(device), target.to(device)
        
        optimizer.zero_grad()
        output, feat = model(x)
        output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        prob2 = F.softmax(output2, dim=1)

        # Similarity labels
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

        # Clustering and consistency losses
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
        for batch_idx, (x, label, _) in enumerate(test_loader):
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
    print('Test overall acc {:.4f}, label acc {:.4f}, unlabel acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)
    tf_writer.add_scalar('nmi/unseen', unseen_nmi, epoch)
    tf_writer.add_scalar('uncert/test', mean_uncert, epoch)
    return mean_uncert

parser = argparse.ArgumentParser(
            description='orca',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='imagenet100', help='dataset setting')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--milestones', nargs='+', type=int, default=[30, 60])
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--dataset_root', type=str)
parser.add_argument('--exp_root', type=str, default='./results/')
parser.add_argument('--labeled-num', default=50, type=int)
parser.add_argument('--labeled-ratio', default=0.5, type=float)
parser.add_argument('--model_name', type=str, default='resnet')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--name', type=str, default='debug')

if __name__ == "__main__":

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    
    args.savedir = args.savedir + '/'

    model = models.resnet50(num_classes=args.num_classes)
    state_dict = torch.load('./pretrained/simclr_imagenet_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)

    for name, param in model.named_parameters(): 
        if 'fc' not in name and 'layer4' not in name:
            param.requires_grad = False

    model = model.to(device)
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    train_label_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet100_label_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
    train_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet100_unlabel_{}_{:.1f}.txt'.format(args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
    concat_set = datasets.ConcatDataset((train_label_set, train_unlabel_set))
    labeled_idxs = range(len(train_label_set)) 
    unlabeled_idxs = range(len(train_label_set), len(train_label_set)+len(train_unlabel_set))
    batch_sampler = datasets.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, int(args.batch_size * len(train_unlabel_set) / (len(train_label_set) + len(train_unlabel_set))))

    test_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet100_unlabel_50_0.5.txt', transform=transform_test)

    train_loader = torch.utils.data.DataLoader(concat_set, batch_sampler=batch_sampler, num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_unlabel_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in range(args.epochs):
        mean_uncert = test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer)
        train(args, model, device, train_loader, optimizer, mean_uncert, batch_sampler.primary_batch_size, epoch, tf_writer)
        scheduler.step()
