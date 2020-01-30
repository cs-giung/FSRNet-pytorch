import os
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.utils.data as data


HELEN_ROOT = '/home/gwnam/datasets/Helen/extracted/'


class HelenDataset(data.Dataset):
    """
    annotations
        * [0:41]    : jawline
        * [41:58]   : nose
        * [58:72]   : upper lip
        * [72:86]   : lower lip
        * [86:114]  : teeth
        * [114:134] : right eye
        * [134:154] :  left eye
        * [154:174] : right eyebrow
        * [174:194] :  left eyebrow
    """
    def __init__(self, root=HELEN_ROOT, mode='train', use_hmaps=True, use_pmaps=True,
                 size_lr=(16, 16), size_hr=(128, 128), size_maps=(64, 64)):
        self.root = root
        self.mode = mode
        self.use_hmaps = use_hmaps
        self.use_pmaps = use_pmaps
        self.size_lr = size_lr
        self.size_hr = size_hr
        self.size_maps = size_maps
        self.img_info = list()

        if self.mode == 'train':
            img_list = os.path.join(self.root, 'list_annos_trn.txt')
        elif self.mode == 'test':
            img_list = os.path.join(self.root, 'list_annos_tst.txt')

        with open(img_list, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split('\t')
                fname = str(line[0])[:-4]
                lmark = [int(e) for e in line[1:]]
                lmark = [(lmark[i], lmark[i+1]) for i in range(0, len(lmark), 2)]
                self.img_info.append({
                    'image': fname,
                    'lmark': lmark,
                })

    def num_channels(self):
        return 194, 9

    def _gaussian_k(self, x0, y0, sigma, width, height):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2)) * 255

    def _get_hmaps(self, lmark):
        hmaps = np.zeros((self.size_maps[0], self.size_maps[1], 194), dtype=np.uint8)
        for i, p in enumerate(lmark):
            hmaps[:,:,i] = self._gaussian_k(p[0], p[1], sigma=3,
                                            width=self.size_maps[0],
                                            height=self.size_maps[1])
        return hmaps

    def _get_pmaps(self, lmark):
        pmaps = np.zeros((self.size_maps[0], self.size_maps[1], 9), dtype=np.uint8)
        # (1) jawline
        _img = Image.fromarray(pmaps[:,:,0])
        draw = ImageDraw.Draw(_img)
        draw.line(lmark[0:41], 'white', 3)
        pmaps[:,:,0] = np.array(_img)
        # (2-8) nose, ..., left eyebrow
        c1 = (sum([p[0] for p in lmark[114:134]]) / len(lmark[114:134]),
              sum([p[1] for p in lmark[114:134]]) / len(lmark[114:134]))
        c2 = (sum([p[0] for p in lmark[134:154]]) / len(lmark[134:154]),
              sum([p[1] for p in lmark[134:154]]) / len(lmark[134:154]))
        ce = (int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2))
        _lst = [lmark[41:58] + [ce], lmark[58:72], lmark[72:86], lmark[86:114],
                lmark[114:134], lmark[134:154], lmark[154:174], lmark[174:194]]
        for i, ps in enumerate(_lst, start=1):
            _img = Image.fromarray(pmaps[:,:,i])
            draw = ImageDraw.Draw(_img)
            draw.polygon(ps, 'white')
            pmaps[:,:,i] = np.array(_img)
        return pmaps

    def _crop_face_coarsely(self, image, lmark):
        xs = [e[0] for e in lmark]
        ys = [e[1] for e in lmark]
        cx, cy = (sum(xs) / len(xs)), (sum(ys) / len(ys))
        w, h = (max(xs) - min(xs)), (max(ys) - min(ys))
        x1 = max(0, int(cx-w))
        y1 = max(0, int(cy-h))
        x2 = min(int(cx+w), image.size[0])
        y2 = min(int(cy+h), image.size[1])
        img = image.crop((x1, y1, x2, y2))
        return img, x1, y1

    def __getitem__(self, index):

        # it will be returned
        image_lr = None
        image_hr = None
        hmaps = np.array([-1])
        pmaps = np.array([-1])

        # load infos
        image = self.img_info[index]['image']
        image = Image.open(os.path.join(self.root, image + '.jpg')).convert('RGB')
        lmark = self.img_info[index]['lmark']

        # crop & resize
        face_img, x1, y1 = self._crop_face_coarsely(image, lmark)
        image_lr = face_img.resize(self.size_lr, Image.BICUBIC)
        image_lr = image_lr.resize(self.size_hr, Image.BICUBIC)
        image_hr = face_img.resize(self.size_hr, Image.BICUBIC)

        # hmaps & pmaps
        if self.use_hmaps or self.use_pmaps:
            fx = self.size_maps[0] / face_img.size[0]
            fy = self.size_maps[1] / face_img.size[1]
            lmark = [(int((e[0] - x1) * fx), int((e[1] - y1) * fy)) for e in lmark]

            if self.use_hmaps:
                hmaps = self._get_hmaps(lmark)
            if self.use_pmaps:
                pmaps = self._get_pmaps(lmark)

        return np.array(image_lr), np.array(image_hr), hmaps, pmaps

    def __len__(self):
        return len(self.img_info)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dset = HelenDataset(mode='test')
    for idx, (image_lr, image_hr, hmaps, pmaps) in enumerate(dset, start=1):

        if idx < 9:
            continue
        
        if idx == 10:
            break

        # lr & hr images
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(image_lr)
        fig.add_subplot(1, 2, 2)
        plt.imshow(image_hr)
        plt.show()

        # hmaps
        fig = plt.figure()
        for i in range(194):
            _img = Image.fromarray(hmaps[:,:,i])
            fig.add_subplot(14, 14, i+1)
            plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
        plt.show()

        # pmaps
        fig = plt.figure()
        for i in range(9):
            _img = Image.fromarray(pmaps[:,:,i])
            fig.add_subplot(3, 3, i+1)
            plt.imshow(_img, cmap='gray', vmin=0, vmax=255)
        plt.show()
