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
from nets import FSRNet, Discriminator, FeatureExtractor
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

# load networks
G = FSRNet(hmaps_ch, pmaps_ch)
G = nn.DataParallel(G)
G = G.cuda()

D = Discriminator(input_shape=(3, 128, 128))
D = nn.DataParallel(D)
D = D.cuda()

F = FeatureExtractor().cuda()
F.eval()

# settings
a = 1
b = 1
r_c = 1e-3
r_p = 1e-1
learning_rate = 2.5e-4
criterion_MSE = nn.MSELoss()
criterion_BCE = nn.BCELoss()
optimizer_G = optim.RMSprop(G.parameters(), lr=learning_rate)
optimizer_D = optim.RMSprop(D.parameters(), lr=learning_rate)

# outputs
output_dir = './outputsGAN_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
checkp_dir = os.path.join(output_dir, '_checkpoints')
logtxt_dir = os.path.join(output_dir, 'log.txt')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkp_dir, exist_ok=True)

# train & valid
num_epoch = 100
for epoch_idx in range(1, num_epoch + 1):

    # train
    losses_D = []; losses_G = []
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

        # get real and fake inputs
        real = image_hr
        y_c, prs, fake = G(image_lr)

        real_label = torch.from_numpy(np.ones((real.size(0), 1, 8, 8))).float().cuda()
        fake_label = torch.from_numpy(np.zeros((real.size(0), 1, 8, 8))).float().cuda()
        real_label.requires_grad = False
        fake_label.requires_grad = False

        # train discriminator
        D.zero_grad()
        loss_c1 = criterion_BCE(D(real), real_label)
        loss_c2 = criterion_BCE(D(fake.detach()), fake_label)
        loss_c = r_c * (loss_c1 + loss_c2)
        loss_c.backward()
        losses_D.append(loss_c.data)
        optimizer_D.step()

        # train generator
        G.zero_grad()
        loss_f1 = criterion_MSE(y_c, real)
        loss_f2 = a * criterion_MSE(fake, real)
        loss_f3 = b * criterion_MSE(prs, image_pr)
        loss_f = loss_f1 + loss_f2 + loss_f3
        loss_p = r_p * criterion_MSE(F(fake), F(real).detach())

        generator_content_loss = loss_f + loss_p
        generator_adversarial_loss = r_c * criterion_BCE(D(fake), real_label)
        generator_total_loss = generator_content_loss + generator_adversarial_loss
        generator_total_loss.backward()
        losses_G.append(generator_total_loss.data)
        optimizer_G.step()

        if batch_idx % 300 == 0:
            _lossD = sum(losses_D) / len(losses_D)
            _lossG = sum(losses_G) / len(losses_G)
            log_str = ''
            log_str += '[%3d/%3d]' % (epoch_idx, num_epoch)
            log_str += '[%3d/%3d]' % (batch_idx, len(trn_dloader))
            log_str += '\t%.4f\t%.4f' % (_lossD, _lossG)
            write_log(logtxt_dir, log_str)
            print(log_str)
            losses_D = []
            losses_G = []

    # valid
    PSNRs = []; SSIMs = []
    for batch_idx, (image_lr, image_hr, _, _) in enumerate(val_dloader, start=1):

        image_lr = torch.from_numpy(_normalize(image_lr)).float().cuda()
        image_hr = torch.from_numpy(_normalize(image_hr)).float().cuda()

        with torch.no_grad():
            y_c, prs, out = G(image_lr)

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
                'state_dict': G.module.state_dict(),
                'optimizer': optimizer_G.state_dict(),
                'state_dict_disc': D.module.state_dict(),
                'optimizer_disc': optimizer_D.state_dict()}, file_name)
