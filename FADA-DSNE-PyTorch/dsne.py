import argparse
import torch
import torch.nn as nn
import dataloader
from models import main_models
import numpy as np
import time
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import DSNE_Loss

parser=argparse.ArgumentParser()
parser.add_argument('--n_epoches',type=int,default=1000)
parser.add_argument('--n_target_samples',type=int,default=7)
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--batch_size_src_test',type=int,default=256)
parser.add_argument('--batch_size_tgt_test',type=int,default=256)
parser.add_argument('--lr',type=float,default=0.001) # learning rate
parser.add_argument('--wd',type=float,default=0.0001) # weight_decay
parser.add_argument('--gpu',type=int,default=1)
parser.add_argument('--exp',type=str,default='test')
parser.add_argument('--src',type=str,default="mnist")
parser.add_argument('--tgt',type=str,default="svhn")
parser.add_argument('--margin',type=float,default=1.0) # dsne loss margin; no loss for differences greater than margin
parser.add_argument('--alpha',type=float,default=0.1) # dsne loss lambda for aux loss
parser.add_argument('--print_freq',type=int,default=10)
parser.add_argument('--ckpt_freq',type=int,default=1)
parser.add_argument('--classes',type=int,default=10)
parser.add_argument('--input_dim',type=int,default=32)
parser.add_argument('--feature_size',type=int,default=512)
parser.add_argument('--dropout',type=float,default=0.5)
parser.add_argument('--neg_data_ratio',type=float,default=3)


opt=vars(parser.parse_args())


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.avg = 0.0
        self.count = 0

    def update(self, val, num=1):
        self.sum += val * num
        self.val = val
        self.count += num
        self.avg = self.sum / self.count if self.count!=0 else 0.0



use_cuda=True if torch.cuda.is_available() else False
device=torch.device('cuda:{}'.format(opt['gpu'])) if use_cuda else torch.device('cpu')
torch.manual_seed(1)

if use_cuda:
    torch.cuda.manual_seed(1)

def to_device_all(x_array, device):
    return [x.to(device) for x in x_array]

log_dir = "runs/{}-{}2{}-{}".format(opt['exp'], opt['src'], opt['tgt'], time.strftime("%Y-%m-%d-%H-%M"))
writer = SummaryWriter(log_dir=log_dir)
                
# prepare model
# model=main_models.InitModel()
model = main_models.LeNetPlus(
    input_dim=opt['input_dim'],
    classes=opt['classes'],
    feature_size=opt['feature_size'],
    dropout=opt['dropout']
    )
model.to(device)
    
image_size = opt['input_dim']
# prepare dataset
src_test_dataset = dataloader.get_dataset(opt['src'], train=False, size=image_size)
tgt_test_dataset = dataloader.get_dataset(opt['tgt'], train=False, size=image_size)
src_test_dataloader = DataLoader(dataset=src_test_dataset, batch_size=opt['batch_size_src_test'], shuffle=False)
tgt_test_dataloader = DataLoader(dataset=tgt_test_dataset, batch_size=opt['batch_size_tgt_test'], shuffle=False)

trs_dataset = dataloader.get_dataset(opt['src'], train=True, size=image_size)
trt_dataset = dataloader.get_dataset(opt['tgt'], train=True, size=image_size)
sampled_trt_dataset = dataloader.sampling_dataset(trt_dataset, opt['n_target_samples'])
train_dataset = dataloader.PairDataset(
    trs_dataset, sampled_trt_dataset, ratio=opt['neg_data_ratio']) # ratio=0: all neg; #neg/#pos=ratio
train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], shuffle=True, drop_last=True)

# optimizer
optimizer = torch.optim.Adam(
        params=list(model.parameters()),
        lr=opt['lr'],
        weight_decay=opt['wd']
    )

# train val
alpha = opt['alpha']
best_test_tgt_acc = 0

for epoch in range(opt['n_epoches']):

    loss_meters = {
        "train_src": AverageMeter(),
        "train_pred_src": AverageMeter(),
        "train_dsne_src": AverageMeter(),
        "train_tgt": AverageMeter(),
        "train_pred_tgt": AverageMeter(),
        "train_dsne_tgt": AverageMeter(),
    }
    acc_meters = {
        "acc_src_src": AverageMeter(),
        "acc_tgt_src": AverageMeter(),
        "acc_src_tgt": AverageMeter(),
        "acc_tgt_tgt": AverageMeter(),
    }

    
    # train
    model.train()
    num_iters = len(train_dataloader)
    for iter_idx, (xs, ys, xt, yt, _) in enumerate(train_dataloader):
        
        def train(xs, ys, xt, yt, target=True):
            postfix = 'tgt' if target else 'src'
            

            optimizer.zero_grad()

            cri_dsne = DSNE_Loss()
            cri_pred = nn.CrossEntropyLoss()

            fts, ys_pred = model(xs)
            ftt, yt_pred = model(xt)

            loss_dsne = cri_dsne(fts, ys, ftt, yt)
            loss_pred = cri_pred(ys_pred, ys)
            loss = (1-alpha) * loss_pred + alpha * loss_dsne

            loss.backward()
            optimizer.step()
            
            _, ys_predicted = torch.max(ys_pred, 1)
            ys_acc = (ys_predicted == ys).sum().item()
            _, yt_predicted = torch.max(yt_pred, 1)
            yt_acc = (yt_predicted == yt).sum().item()
            
            writer.add_scalar('Loss/train_'+postfix, loss.item(), iter_idx+len(train_dataloader)*epoch)
            writer.add_scalar('Loss/train_pred_'+postfix, loss_pred.item(), iter_idx+len(train_dataloader)*epoch)
            writer.add_scalar('Loss/train_dsne_'+postfix, loss_dsne.item(), iter_idx+len(train_dataloader)*epoch)
            loss_meters["train_"+postfix].update(loss.item(), xs.size(0))
            loss_meters["train_pred_"+postfix].update(loss_pred.item(), xs.size(0))
            loss_meters["train_dsne_"+postfix].update(loss_dsne.item(), xs.size(0))
            writer.add_scalar('Loss/ac_train_'+postfix, loss_meters["train_"+postfix].val, 
                              iter_idx+len(train_dataloader)*epoch)
            writer.add_scalar('Loss/ac_train_pred_'+postfix, loss_meters["train_pred_"+postfix].val,
                              iter_idx+len(train_dataloader)*epoch)
            writer.add_scalar('Loss/ac_train_dsne_'+postfix, loss_meters["train_dsne_"+postfix].val,
                              iter_idx+len(train_dataloader)*epoch)
             
            acc_meters["acc_src_"+postfix].update(ys_acc/xs.size(0), xs.size(0)) 
            acc_meters["acc_tgt_"+postfix].update(yt_acc/xs.size(0), xs.size(0))
            writer.add_scalar('Acc/acc_src_'+postfix, acc_meters["acc_src_"+postfix].val, iter_idx+len(train_dataloader)*epoch)
            writer.add_scalar('Acc/acc_tgt_'+postfix, acc_meters["acc_tgt_"+postfix].val, iter_idx+len(train_dataloader)*epoch)
            
    
            ########### visualization
            # img_batch = np.zeros((16, 3, image_size, image_size))
            # for i in range(8):
            #     img_batch[i] = xs[i].data.cpu()
            #     img_batch[i+8] = xt[i].data.cpu()
            # 
            # y_batch = np.zeros((2, 8))
            # y_batch[0] = ys[:8].data.cpu()
            # y_batch[1] = yt[:8].data.cpu()
            # writer.add_images('my_image_batch', img_batch, float(iter_idx)/len(train_dataloader)+epoch)
            # writer.add_text('my_image_batch_text', str(y_batch), float(iter_idx)/len(train_dataloader)+epoch)

            if iter_idx % opt['print_freq'] == 0:
                print("Epoch {} [{}/{}] Training Loss {}: {:.5} (pred: {:.5}, dsne: {:.5}), {}: {:.4}, {}: {:.4}".format(
                    epoch, iter_idx, num_iters, postfix, 
                    loss_meters["train_"+postfix].val, 
                    loss_meters["train_pred_"+postfix].val,
                    loss_meters["train_dsne_"+postfix].val,
                    "acc_src_"+postfix, acc_meters["acc_src_"+postfix].val,
                    'acc_tgt_'+postfix, acc_meters["acc_tgt_"+postfix].val,
                ))
        
        xs, ys, xt, yt = to_device_all([xs, ys, xt, yt], device)
        train(xs, ys, xt, yt, target=False)
        train(xt, yt, xs, ys, target=True)

                
    # test on src_test
    model.eval()
    acc = 0
    total = 0
    for x, y in src_test_dataloader:
        x, y = to_device_all([x, y], device)
        _, y_pred = model(x)
        _, y_predicted = torch.max(y_pred, 1)
        acc += (y_predicted == y).sum().item()
        total += y.size(0)
    accuracy = acc/total*100.0
    writer.add_scalar('Acc/src_test', accuracy, epoch)
    print("---- Epoch [{}/{}] src_test accuracy: {:.4}% ----".format(epoch, opt['n_epoches'], accuracy))
        
    # test on tgt_tst
    acc = 0
    total = 0
    for x, y in tgt_test_dataloader:
        x, y = to_device_all([x, y], device)
        _, y_pred = model(x)
        _, y_predicted = torch.max(y_pred, 1)
        acc += (y_predicted == y).sum().item()
        total += y.size(0)
    accuracy = acc/total*100.0
    writer.add_scalar('Acc/tgt_test', accuracy, epoch)
    print("---- Epoch [{}/{}] tgt_test accuracy: {:.4}% ----".format(epoch, opt['n_epoches'], accuracy))
    
    if epoch + 1 % opt['ckpt_freq']:
        model.save(os.path.join(log_dir, "{}.pth".format(epoch+1)))
        model.save(os.path.join(log_dir, "lastest.pth"))
    
    if best_test_tgt_acc < accuracy:
        best_test_tgt_acc = accuracy
        model.save(os.path.join(log_dir, "best.pth"))
        
    
        
        

















