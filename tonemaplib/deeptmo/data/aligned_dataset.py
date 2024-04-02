### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
from .base_dataset import BaseDataset, get_params, get_transform, normalize
from .image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import scipy.io as ios
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from skimage import io, color

class hdr2luminance(object):
    "converts an input hdr to its luminance counterpart ndarray"
    def __call__(self,sample):
        x = sample
        y = np.zeros_like(x)
        x = 0.2959*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]
        y[:,:,0] = x
        y[:,:,1] = x
        y[:,:,2] = x
# NEED TO reshape here the  such thatit appears gray scale
#        y = y.transpose(2,1,0).astype(np.float64) #converting to HXWXC ndarray
        return y

def CIELabFunction(t):
        Ft = np.zeros_like(t)
        c1 = (6 / 29)**3
        c2 = ((29 / 6)**2)/3
        c3 = 4/29
        Ft[t >  c1] = np.power(t[t >  c1],(1 / 3))
        Ft[t <= c1] = t[t <= c1] * c2 + c3
        return Ft

def CIELabInvFunction(t):
        Ft = np.zeros_like(t)
        c1 = (6 / 29)
        c2 = ((29 / 6)**2)/3
        c3 = 4/29
        Ft[t >  c1] = np.power(t[t >  c1],3)
        Ft[t <= c1] = (t[t <= c1] - c3)/c2
        return Ft

class hdr2lch(object):
    "converts an input hdr to its luminance counterpart ndarray"
    def __call__(self,sample):
        # First convert to XYZ COLORSPACE
        x = sample
        y = np.zeros_like(x)
        imgxyz = x; imglab = x; imgLCH = x; 
        mtx = [[ 0.4124, 0.3576, 0.1805],[0.2126, 0.7152, 0.0722],[0.0193, 0.1192, 0.9505]]
        for i in range(3):
            imgxyz[:,:,i] = x[:,:,0]*mtx[i][0] + x[:,:,1]*mtx[i][1] + x[:,:,2]*mtx[i][2]
        # Now Convert To lab 
        fY = CIELabFunction(imgxyz[:,:,1])
        imglab[:,:,0] = 116 * fY - 16
        imglab[:,:,1] = 500 * (CIELabFunction(imgxyz[:,:,0]) - fY)
        imglab[:,:,2] = 200 * (fY - CIELabFunction(imgxyz[:,:,2]) ) 

        # Now convert to LCH space
        imgLCH[:,:,0] = imglab[:,:,0]
        imgLCH[:,:,1] = np.sqrt(np.power(imglab[:,:,1],2) + np.power(imglab[:,:,2],2))
        rad_to_deg = 180/3.14

        tmp = np.arctan2(imglab[:,:,2], imglab[:,:,1]) * rad_to_deg
        tmp[tmp < 0] = tmp[tmp < 0] + 360
        imgLCH[:,:,2] = tmp
        return imgLCH

class lch2hdr(object):
    "converts an input hdr to its luminance counterpart ndarray"
    def __call__(self,sample):
        deg_to_rad = 3.14/180

        imgLCH = sample
        imglab = np.zeros_like(imgLCH)
        rad_angle = imgLCH[..., -1]
        rad_angle[rad_angle > 180] = rad_angle[rad_angle > 180] - 360
        rad_angle = rad_angle * deg_to_rad

        # First convert from LCH space to Lab
        imglab[:,:,0] = imgLCH[:,:,0]
        imglab[:,:,1] = imgLCH[:,:,1] * np.cos(rad_angle)
        imglab[:,:,2] = imgLCH[:,:,1] * np.sin(rad_angle)

        # Convert from Lab to XYZ
        imgxyz = np.zeros_like(imglab)
        Lnorm = (imglab[:,:,0]+16)/116
        imgxyz[:,:,0] = CIELabInvFunction(Lnorm + imglab[:,:,1]/500)
        imgxyz[:,:,1] = CIELabInvFunction(Lnorm)
        imgxyz[:,:,2] = CIELabInvFunction(Lnorm - imglab[:,:,2]/200)

        # Convert from XYZ to HDR
        imghdr = np.zeros_like(imgxyz)
        mtx = np.linalg.inv([[ 0.4124, 0.3576, 0.1805],[0.2126, 0.7152, 0.0722],[0.0193, 0.1192, 0.9505]])
        for i in range(3):
            imghdr[:,:,i] = imgxyz[:,:,0]*mtx[i][0] + imgxyz[:,:,1]*mtx[i][1] + imgxyz[:,:,2]*mtx[i][2]

        return imghdr

class Changecolorspace(object):
    "converts an input hdr from gb to different color space"
    def __call__(self,sample):
        x = sample
        y = cv2.cvtColor(x, cv2.COLOR_BGR2YCR_CB)
        return y


class MinMaxNormalize(object):
    "Normalizing the ndarray between zero and one using its minimum and maximum value"
    def __call__(self,sample):
        x = sample
        xmin = x.min()
        xmax = x.max()
        x = x-xmin
        if (xmax-xmin) != 0:
            x = x/(xmax-xmin)
        
        return x

class LogNormalize(object):
    "Normalizing the ndarray between zero and one using its minimum and maximum value"
    def __call__(self,sample):


        x = sample

        x = np.log(x+1e-5)
        xmin = x.min()
        x = x-xmin

        return x

class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    """
    def __call__(self, sample):
        w, h = sample.shape[1], sample.shape[0]
        th, tw = self.finesize
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return sample[y1:y1+th, x1:x1+tw, :]

class convert(object):
    "convert to Tensor from numpy"
    def __call__(self,sample):
         x = sample
#         x = x.transpose(1,0)
         x = x.transpose(2,0,1).astype(np.float32)#converting from HXWXC to CXHXW
         return torch.from_numpy(x)

class deconvert(object):
    "convert to numpy from Tensor"
    def __call__(self,sample):
         x = sample
#         x = x.transpose(1,0)
         x = x.permute(1,2,0)#converting from CXHXW to HXWXC
         return x.cpu().numpy() 

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### label maps
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')
        self.label_paths = sorted(make_dataset(self.dir_label))
        ### real images
        if opt.isTrain:
            self.dir_image = os.path.join(opt.dataroot, opt.phase + '_img')
            self.image_paths = sorted(make_dataset(self.dir_image))

        ### instance maps
        '''
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))
        '''
        self.dataset_size = len(self.label_paths)
#        transform_list = [hdr2luminance(),MinMaxNormalize(),convert()]
        transform_list = [hdr2lch(),convert()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):

        ### label maps
        label_path = self.label_paths[index]
        label = ios.loadmat(label_path)['C']
#        label = Image.open(label_path)

        params = get_params(self.opt,(label.shape[2],label.shape[1]))
        if self.opt.label_nc == 0:
            label_tensor = self.transform(label)
            w = label_tensor.size(2)
            h = label_tensor.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
#            label_tensor = label_tensor[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
                #print('label tensor cropped sizes',label_tensor.size())


#            transform_label = get_transform(self.opt, params)
#            label_tensor = transform_label(label.convert('RGB'))
        else:
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
#            label_tensor = transform_label(label) * 255.0
        image_tensor = inst_tensor = feat_tensor = 0
        ### real images
        if self.opt.isTrain:
            image_path = self.image_paths[index]
            image_tensor= ios.loadmat(image_path)['C']
            image_tensor = self.transform(image_tensor)
#            image_tensor = image_tensor[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
   #         	image = Image.open(image_path).convert('RGB')

		#print('input tensor original size',image_tensor.size())
                #print('input tensor cropped sizes',image_tensor.size())
#		'''
#             	if (not self.opt.no_flip) and random.random() < 0.5:
#			idx = [i for i in range(image_tensor.size(2) - 1, -1, -1)]
#	            	idx = torch.LongTensor(idx)
#                    	label_tensor = label_tensor.index_select(2, idx)
#	            	image_tensor = image_tensor.index_select(2, idx)
#		'''
  #          transform_image = get_transform(self.opt, params)
  #          image_tensor = transform_image(image)

        ### if using instance maps

#        if not self.opt.no_instance:
#            inst_path = self.inst_paths[index]
#            inst = Image.open(inst_path)
#            inst_tensor = transform_label(inst)
#
#            if self.opt.load_features:
#                feat_path = self.feat_paths[index]
#                feat = Image.open(feat_path).convert('RGB')
#                norm = normalize()
#                feat_tensor = norm(transform_label(feat))

        input_dict = {'label': label_tensor, 'inst': inst_tensor, 'image': image_tensor,
                      'feat': feat_tensor, 'path': label_path}

        return input_dict

    def __len__(self):
        return len(self.label_paths)

    def name(self):
        return 'AlignedDataset'
