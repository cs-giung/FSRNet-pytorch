import argparse
import numpy as np
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
parser.add_argument('--checkpoint', default=None, type=str)
args = parser.parse_args()

# load dataset
if args.dataset == 'celeba':
    dataset = CelebDataset(mode='test')
elif args.dataset == 'helen':
    dataset = HelenDataset(mode='test')
else:
    print('not implemented')
    exit()

# evaluation
psnr_lst = []
ssim_lst = []
mode_str = ''

if args.checkpoint is None:
    # BICUBIC
    for idx, (image_lr, image_hr, _, _) in enumerate(dataset):
        psnr_lst.append(compare_psnr(image_hr, image_lr))
        ssim_lst.append(compare_ssim(image_hr, image_lr, multichannel=True))
        mode_str = 'BICUBIC'

else:
    # load network
    hmaps_ch, pmaps_ch = dataset.num_channels()
    net = FSRNet(hmaps_ch, pmaps_ch).to(args.device)

    # load weights
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['state_dict_G']
    net.load_state_dict(state_dict)

    # predict
    PSNRs = 0
    for idx, (image_lr, image_hr, _, _) in enumerate(dataset):
        
        x = _normalize(image_lr[np.newaxis])
        x = torch.from_numpy(x).float().to(args.device)
        with torch.no_grad():
            _, _, out = net(x)
        out = _denormalize(out.cpu())[0].astype('uint8')

        psnr_lst.append(compare_psnr(image_hr, out))
        ssim_lst.append(compare_ssim(image_hr, out, multichannel=True))
        mode_str = args.checkpoint

print(mode_str)
print('* Mean PSNR = %.4f' % (sum(psnr_lst) / len(psnr_lst)))
print('* Mean SSIM = %.4f' % (sum(ssim_lst) / len(ssim_lst)))
