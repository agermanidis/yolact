from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

set_cfg('yolact_resnet50_config')
cudnn.benchmark = True
cudnn.fastest = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
net = Yolact()
net.load_weights('weights/yolact_resnet50_54_800000.pth')
net.eval()                        
net = net.cuda()

net.detect.use_fast_nms = True
cfg.mask_proto_debug = False

path = "cat.jpg"
frame = torch.from_numpy(cv2.imread(path)).cuda().float()
batch = FastBaseTransform()(frame.unsqueeze(0))
print(batch.shape)
preds = net(batch)
