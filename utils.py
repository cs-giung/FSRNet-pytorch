import numpy as np
from datetime import datetime


def write_log(log_file, log_str):
    f = open(log_file, 'a')
    f.write('[{:%Y-%m-%d-%H-%M-%S}] '.format(datetime.now())
            + log_str + '\n')
    f.close()

def _normalize(image):
    # (N,W,H,C) >> (N,C,W,H)
    #  [0, 255] >> [-1, 1]
    tensor = np.array(image).transpose(0, 3, 1, 2)
    tensor = (tensor - 127.5) / 127.5
    tensor = np.clip(tensor, -1, 1)
    return tensor

def _denormalize(tensor):
    # (N,C,W,H) >> (N,W,H,C)
    #   [-1, 1] >> [0, 255]
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    image = np.clip(image, 0, 255)
    return image
