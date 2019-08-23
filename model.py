# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from data import COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import defaultdict
import cv2
from subprocess import call 
import os

class YOLACT_MODEL():

    def __init__(self, opts):
        #concat the two files to one file 
        # if not os.path.isfile('weights/yolact_resnet50_54_800000.pth'):    
        #     script = "cat weights/a* > weights/yolact_resnet50_54_800000.pth"
        #     call(script, shell=True)

        set_cfg('yolact_resnet50_config')
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.net = Yolact()
        self.net.load_weights(opts['checkpoint'])
        print("done.")

        self.net.eval()                        
        self.net = self.net.cuda()

        self.net.detect.use_fast_nms = True
        cfg.mask_proto_debug = False
        self.color_cache = defaultdict(lambda: {})
        self.threshold = opts['threshold']
        
    # Generate an image based on some text.
    def detect(self, img):
        numpy_image = np.array(img)
        print('starting inference...')
        frame = torch.from_numpy(numpy_image).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = self.net(batch)
        print("done.")
        output_image = self.display(preds, frame, None, None,
                                     undo_transform=False, score_threshold=self.threshold)
        return output_image

    def display(self, dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, top_k = 100, score_threshold = 0.3):
        img_gpu = img / 255.0
        h, w, _ = img.shape
        
        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                            crop_masks        = True,
                                            score_threshold   = score_threshold)
            torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:top_k]

        img_gpu = img_gpu * masks[0]
            
        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
        img_numpy = (img_gpu * 255).byte().cpu().numpy()
               
        return img_numpy        