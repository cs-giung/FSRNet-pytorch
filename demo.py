import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

import torch
import torch.backends.cudnn as cudnn

from datasets import HelenDataset, CelebDataset
from nets import FSRNet
from utils import _normalize, _denormalize


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='celeba', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--checkpoint', required=True, type=str)
args = parser.parse_args()

# load dataset
if args.dataset == 'celeba':
    dataset = CelebDataset(mode='test')
elif args.dataset == 'helen':
    dataset = HelenDataset(mode='test')
else:
    print('not implemented')
    exit()

# load network
hmaps_ch, pmaps_ch = dataset.num_channels()
net = FSRNet(hmaps_ch, pmaps_ch).to(args.device)

# load weights
checkpoint = torch.load(args.checkpoint)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['state_dict_G']
net.load_state_dict(state_dict)

# predict
(image_lr, image_hr, _, pmaps) = dataset[0]
x = _normalize(image_lr[np.newaxis])
x = torch.from_numpy(x).float().to(args.device)
with torch.no_grad():
    y_c, prs, out = net(x)
y_c = _denormalize(y_c.cpu())[0].astype('uint8')
prs = _denormalize(prs.cpu())[0].astype('uint8')
out = _denormalize(out.cpu())[0].astype('uint8')

# plot images
fig = plt.figure(figsize=(10,3))

fig.add_subplot(1, 4, 1)
plt.title('Target')
plt.imshow(image_hr, vmin=0, vmax=255)

fig.add_subplot(1, 4, 2)
plt.title('Input (Bicubic)')
_img = image_lr
plt.imshow(_img, vmin=0, vmax=255)
psnr = compare_psnr(image_hr, _img)
ssim = compare_ssim(image_hr, _img, multichannel=True)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim))

fig.add_subplot(1, 4, 3)
plt.title('FSRNet(y_c)')
_img = y_c
plt.imshow(_img, vmin=0, vmax=255)
psnr = compare_psnr(image_hr, _img)
ssim = compare_ssim(image_hr, _img, multichannel=True)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim))

fig.add_subplot(1, 4, 4)
plt.title('FSRNet(out)')
_img = out
plt.imshow(_img, vmin=0, vmax=255)
psnr = compare_psnr(image_hr, _img)
ssim = compare_ssim(image_hr, _img, multichannel=True)
plt.xlabel('PSNR=%.2f\nSSIM=%.4f' % (psnr, ssim))

plt.tight_layout()
plt.show()

# plot parsing-maps
if len(pmaps) != 1:
    fig = plt.figure(figsize=(10,3))
    for i in range(pmaps_ch):
        fig.add_subplot(2, pmaps_ch, i+1)
        _img = pmaps[:,:,i]
        plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
    for i in range(pmaps_ch):
        fig.add_subplot(2, pmaps_ch, i+1+pmaps_ch)
        _img = prs[:,:,i]
        plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
    plt.show()
