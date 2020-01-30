import os
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from datasets import HelenDataset, CelebDataset
from nets import FSRNet
from utils import write_log, _normalize, _denormalize


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='celeba', type=str)
args = parser.parse_args()

# load dataset
if args.dataset == 'celeba':
    trn_dataset = CelebDataset(mode='train')
    val_dataset = CelebDataset(mode='test')
elif args.dataset == 'helen':
    trn_dataset = HelenDataset(mode='train')
    val_dataset = HelenDataset(mode='test')
else:
    print('not implemented')
    exit()

trn_dloader = torch.utils.data.DataLoader(dataset=trn_dataset, batch_size=14, shuffle=True)
val_dloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
hmaps_ch, pmaps_ch = trn_dataset.num_channels()

# load network
net = FSRNet(hmaps_ch, pmaps_ch)
net = nn.DataParallel(net)
net = net.cuda()

# settings
learning_rate = 2.5e-4
criterion = nn.MSELoss()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

# outputs
output_dir = './outputs_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
checkp_dir = os.path.join(output_dir, '_checkpoints')
logtxt_dir = os.path.join(output_dir, 'log.txt')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkp_dir, exist_ok=True)

# train & valid
num_epoch = 100
for epoch_idx in range(1, num_epoch + 1):

    # train
    loss1s = []; loss2s = []; loss3s = []; losses = []
    for batch_idx, (image_lr, image_hr, hmaps, pmaps) in enumerate(trn_dloader, start=1):

        image_lr = torch.from_numpy(_normalize(image_lr)).float().cuda()
        image_hr = torch.from_numpy(_normalize(image_hr)).float().cuda()

        b1 = (len(np.array(hmaps)[0]) != 1)
        b2 = (len(np.array(pmaps)[0]) != 1)

        if b1 and b2:
            hmaps = torch.from_numpy(_normalize(hmaps)).float().cuda()
            pmaps = torch.from_numpy(_normalize(pmaps)).float().cuda()
            image_pr = torch.cat((hmaps, pmaps), 1)
        elif b1:
            image_pr = torch.from_numpy(_normalize(hmaps)).float().cuda()
        elif b2:
            image_pr = torch.from_numpy(_normalize(pmaps)).float().cuda()

        y_c, prs, out = net(image_lr)
        loss1 = criterion(y_c, image_hr)
        loss2 = criterion(out, image_hr)
        loss3 = criterion(prs, image_pr)
        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss1s.append(loss1.data)
        loss2s.append(loss2.data)
        loss3s.append(loss3.data)
        losses.append(loss.data)

        if batch_idx % 300 == 0:
            _loss = sum(losses) / len(losses)
            _loss1 = sum(loss1s) / len(loss1s)
            _loss2 = sum(loss2s) / len(loss2s)
            _loss3 = sum(loss3s) / len(loss3s)
            log_str = ''
            log_str += '[%3d/%3d]' % (epoch_idx, num_epoch)
            log_str += '[%5d/%5d]' % (batch_idx, len(trn_dloader))
            log_str += '\t%.4f\t%.4f\t%.4f\t%.4f' % (_loss, _loss1, _loss2, _loss3)
            write_log(logtxt_dir, log_str)
            print(log_str)
            loss1s = []; loss2s = []; loss3s = []; losses = []

    # valid
    PSNRs = []; SSIMs = []
    for batch_idx, (image_lr, image_hr, _, _) in enumerate(val_dloader, start=1):

        image_lr = torch.from_numpy(_normalize(image_lr)).float().cuda()
        image_hr = torch.from_numpy(_normalize(image_hr)).float().cuda()

        with torch.no_grad():
            y_c, prs, out = net(image_lr)

        real = _denormalize(image_hr)[0].astype('uint8')
        pred = _denormalize(out)[0].astype('uint8')
        psnr = compare_psnr(real, pred)
        ssim = compare_ssim(real, pred, multichannel=True)
        PSNRs.append(psnr)
        SSIMs.append(ssim)

        if batch_idx == 1:

            _dir = os.path.join(output_dir, '%03d' % epoch_idx)
            os.makedirs(_dir, exist_ok=True)

            y_c = _denormalize(y_c)[0].astype('uint8')
            out = _denormalize(out)[0].astype('uint8')
            prs = _denormalize(prs)[0].astype('uint8')

            y_c_img = Image.fromarray(y_c)
            out_img = Image.fromarray(out)
            pms_img = Image.new('RGB', (64*pmaps_ch, 64))
            for i in range(pmaps_ch):
                pms_img.paste(Image.fromarray(prs[:,:,hmaps_ch+i].astype('uint8')), (64*i, 0))

            y_c_img.save(os.path.join(_dir, '%d_lrimg_pred.jpg' % batch_idx))
            out_img.save(os.path.join(_dir, '%d_hrimg_pred.jpg' % batch_idx))
            pms_img.save(os.path.join(_dir, '%d_pmaps_pred.jpg' % batch_idx))


    mean_psnr = sum(PSNRs) / len(PSNRs)
    mean_ssim = sum(SSIMs) / len(SSIMs)
    log_str = '* Mean PSNR = %.4f, Mean SSIM = %.4f' % (mean_psnr, mean_ssim)
    write_log(logtxt_dir, log_str)
    print(log_str)

    file_name = os.path.join(checkp_dir, '%03d_psnr%.2f.pth' % (epoch_idx, mean_psnr))
    torch.save({'epoch': epoch_idx,
                'hmaps_ch': hmaps_ch,
                'pmaps_ch': pmaps_ch,
                'state_dict': net.module.state_dict(),
                'optimizer': optimizer.state_dict()}, file_name)
