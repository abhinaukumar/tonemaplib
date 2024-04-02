from typing import Optional, List, Tuple
from argparse import Namespace
import os
import gdown

import torch
from torchvision import transforms
import numpy as np

from .deeptmo.models.models import create_model
from .deeptmo.data.aligned_dataset import MinMaxNormalize, convert, deconvert

from videolib import Video, Frame, standards
from videolib.standards import Standard
from videolib.cvt_color import rgb2yuv
from .tmo import TMO
from .boitard12 import Boitard12TMO


class Rana19TMO(TMO):
    '''
    Implementation of Rana's 2019 DeepTMO.

    Args:
        out_standard (Standard): Standard to which outputs must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format: Format to which output video must be written (Default: 'encoded').
        use_cuda (bool): Flag to decide whether to run inference using CUDA.
        batch_size (int): Batch size to use when tonemapping a video.
        desat (float): Desaturation parameter for color correction.

    Refs:
        Rana, A., Singh, P., Valenzise, G., Dufaux, F., Komodakis, N., & Smolic, A. (2019).
        "Deep tone mapping operator for high dynamic range images".
        IEEE Transactions on Image Processing, 29, 1285-1298.

        https://v-sense.scss.tcd.ie/code-deeptmo/
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        use_cuda: Optional[bool] = True,
        device: Optional[int] = None,
        batch_size: Optional[int] = 1,
        desat: float = 0.0
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which outputs must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            use_cuda: Flag to decide whether to run inference using CUDA (Default: True).
            device: ID of the GPU to use if using cuda. (Default: None, uses cpu or cuda:0).
            batch_size: Batch size to use when tonemapping a video (Default: 5).
            desat: Desaturation parameter for color correction (Default: 0.0).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.use_cuda = use_cuda
        if not use_cuda and device is not None:
            raise ValueError('When use_cuda is False, device must be None')
        elif use_cuda:
            self.device = 0 if device is None else device
        else:
            self.device = None
        self.batch_size = batch_size
        self.desat = desat

        if not os.path.isfile(os.path.join('deeptmo/OfficialRelease/1000_net_G.pth')):
            self._download_checkpoint()

        self._deeptmo_model = create_model(self._deeptmo_opt)
        if use_cuda:
            self._deeptmo_model = self._deeptmo_model.cuda(self.device)

        self._preprocessing_transform = transforms.Compose([MinMaxNormalize(), convert()])
        self._postprocessing_transform = transforms.Compose([deconvert()])

        self._zero_tensor = torch.Tensor([0])
        if self.use_cuda:
            self._zero_tensor = self._zero_tensor.cuda(self.device)

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'desat']

    @property
    def _deeptmo_opt(self) -> Namespace:
        '''
        Options to be passed to DeepTMO.
        '''
        opt: Namespace = Namespace()

        # Run model in inference mode
        opt.isTrain = False

        # experiment specifics
        opt.name = 'OfficialRelease'  # name of the experiment. It decides where to store samples and models
        opt.gpu_ids = ['cuda:{}'.format(self.device)] if self.use_cuda else []  # CUDA device to use
        opt.checkpoints_dir = os.path.join(os.path.dirname(__file__), 'deeptmo')
        opt.model = 'pix2pixHD'
        opt.which_epoch = '1000'
        opt.norm = 'instance'  # instance normalization or batch normalization
        opt.use_dropout = False  # do not use dropout for the generator

        # input/output sizes
        opt.label_nc = 0  # number of input image channels - 0 uses all 3 channels
        opt.output_nc = 3  # number of output image channels

        # for setting inputs
        opt.resize_or_crop = 'none'  # scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]

        # for generator
        opt.netG = 'local'  # selects model to use for netG
        opt.ngf = 32  # number of gen filters in first conv layer')
        opt.n_downsample_global = 4  # number of downsampling layers in netG
        opt.n_blocks_global = 9  # number of residual blocks in the global generator network
        opt.n_blocks_local = 3  # number of residual blocks in the local enhancer network
        opt.n_local_enhancers = 1  # 'number of local enhancers to use

        # for instance-wise features
        opt.no_instance = True  # if specified, do *not* add instance map as input
        opt.instance_feat = False  # if specified, add encoded instance features as input
        opt.label_feat = False  # if specified, add encoded label features as input

        opt.verbose = False

        return opt

    def _download_checkpoint(self):
        gdown.download_folder(id='1jG9g2032q_Cbps4HtFCauqoJRhegkC6g', output=os.path.join(os.path.dirname(__file__), 'deeptmo', 'OfficialRelease'))

    @staticmethod
    def _pad_to_multiple_of_32(img: np.ndarray) -> np.ndarray:
        '''
        Pad images so that their dimensions are multiples of 32.

        Args:
            img: Image to be padded.

        Returns:
            np.ndarray: Padded image.
        '''

        img_size = img.shape
        img_padded_size = tuple([int(np.ceil(size/32))*32 for size in img_size])
        return np.pad(img, [(0, pad_size - size) for size, pad_size in zip(img_size, img_padded_size)])

    def tonemap_frame(self, frame: Frame, limits: Optional[Tuple[float, float]] = None) -> Frame:
        '''
        Tone-map a frame using Rana's 2019 TMO.

        Args:
            frame: Frame to be tone-mapped.

        Returns:
            Frame: Tone-mapped frame.
        '''
        linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
        lum_padded = Rana19TMO._pad_to_multiple_of_32(linear_yuv_const[..., 0])
        lum_tiled = Frame._lift_to_multichannel(lum_padded)
        if limits is None:
            input_tensor = self._preprocessing_transform(lum_tiled).unsqueeze(0)
        else:
            minval, maxval = limits
            input_tensor = convert()((lum_tiled - minval) / (maxval - minval)).unsqueeze(0)

        if self.use_cuda:
            input_tensor = input_tensor.cuda(self.device)

        output_tensor = self._deeptmo_model.inference(input_tensor, self._zero_tensor).squeeze()
        lum_out = self._postprocessing_transform(output_tensor)

        lum = lum_tiled[:frame.height, :frame.width]
        lum_out = lum_out[:frame.height, :frame.width]

        temp_out_frame = Frame(frame.standard)
        temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
        out_frame = self.gamut_map(temp_out_frame)
        return out_frame

    def tonemap_video_framewise(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a video framewise using Rana's 2019 TMO.

        Args:
            video: Video object containing the input HDR video.
            out_filepath: Path to which output file must be written.
        '''
        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        temp_out_frame = Frame(video.standard)
        num_batches = int(np.ceil(video.num_frames / self.batch_size))
        input_batch = torch.zeros(self.batch_size, 3, int(np.ceil(video.height/32)*32), int(np.ceil(video.width/32)*32))
        if self.use_cuda:
            input_batch = input_batch.cuda(self.device)
        for batch_id in range(num_batches):
            # print('Processing batch: {}/{}'.format(batch_id+1, num_batches))
            actual_batch_size = min(self.batch_size, video.num_frames - batch_id*self.batch_size)

            for i in range(actual_batch_size):
                frame = video[batch_id*self.batch_size + i]
                linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
                lum_tiled = Frame._lift_to_multichannel(Rana19TMO._pad_to_multiple_of_32(linear_yuv_const[..., 0]))
                input_batch[i] = self._preprocessing_transform(lum_tiled).to(device=input_batch.device)

            output_batch = self._deeptmo_model.inference(input_batch, self._zero_tensor)

            for i in range(actual_batch_size):
                frame = video[batch_id*self.batch_size + i]
                lum = self._postprocessing_transform(input_batch[i])[:video.height, :video.width]
                lum_out = self._postprocessing_transform(output_batch[i])[:video.height, :video.width]
                temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
                out_video.append(self.gamut_map(temp_out_frame))

        out_video.close()

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Rana's 2019 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        minval = np.inf
        maxval = -np.inf
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            minval = min(linear_yuv_const[..., 0].min(), minval)
            maxval = max(linear_yuv_const[..., 0].max(), maxval)
        video.reset()

        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        temp_out_frame = Frame(video.standard)
        num_batches = int(np.ceil(video.num_frames / self.batch_size))
        input_batch = torch.zeros(self.batch_size, 3, int(np.ceil(video.height/32)*32), int(np.ceil(video.width/32)*32))
        if self.use_cuda:
            input_batch = input_batch.cuda(self.device)
        for batch_id in range(num_batches):
            # print('Processing batch: {}/{}'.format(batch_id+1, num_batches))
            actual_batch_size = min(self.batch_size, video.num_frames - batch_id*self.batch_size)

            for i in range(actual_batch_size):
                frame = video[batch_id*self.batch_size + i]
                linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
                lum_tiled = Frame._lift_to_multichannel(Rana19TMO._pad_to_multiple_of_32(linear_yuv_const[..., 0]))
                input_batch[i] = convert()((lum_tiled - minval) / (maxval - minval)).to(device=input_batch.device)

            output_batch = self._deeptmo_model.inference(input_batch, self._zero_tensor)

            for i in range(actual_batch_size):
                frame = video[batch_id*self.batch_size + i]
                lum = self._postprocessing_transform(input_batch[i])[:video.height, :video.width]
                lum_out = self._postprocessing_transform(output_batch[i])[:video.height, :video.width]
                temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
                out_video.append(self.gamut_map(temp_out_frame))
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max') -> None:
        '''
        Tone-map a video smoothly using Rana's 2019 TMO and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath)
