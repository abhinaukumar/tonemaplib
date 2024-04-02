from typing import Optional, Tuple, List

import numpy as np
import scipy as sp
import skimage.transform

from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.cvt_color import rgb2yuv
from .tmo import TMO
from .boitard12 import Boitard12TMO


class Durand02TMO(TMO):
    '''
    Implementation of Durand'2 TMO from 2002.

    Args:
        out_standard (Standard): Standard to which the output frames must comply.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        num_segments (int): Number of segments into which the luminance range must be partitioned.
        base_contrast (float): Target contrast (ratio of max to min) for the base layer.
        downsampling_factor (float): Factor by which image is downsampled during bilateral filtering.
        influence_function (str): Name of the influence function to use.
        desat (float): Desaturation parameter for color correction.
    '''
    params = ['video_mode', 'num_segments', 'base_contrast', 'downsampling_factor', 'influence_function', 'desat']
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        num_segments: Optional[int] = None,
        base_contrast: Optional[float] = 50,
        downsampling_factor: Optional[float] = 10,
        influence_function: Optional[str] = 'gaussian',
        desat: Optional[float] = 0.0
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must comply (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            num_segments: Number of segments into which the luminance range must be partitioned (Default: None - inferred per frame).
            base_contrast: Target contrast (ratio of max to min) for the base layer (Default: 50).
            downsampling_factor: Factor by which image is downsampled during bilateral filtering (Default: 10).
            influence_function: Name of the influence function to use (Default: 'gaussian').
            desat: Desaturation parameter for color correction (Default: 0.0).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self._valid_influence_functions = ['gaussian', 'biweight', 'huber']
        if influence_function not in self._valid_influence_functions:
            raise ValueError('Invalid influence function.')
        self.influence_function: str = influence_function
        self.num_segments: int = num_segments
        self.downsampling_factor: float = downsampling_factor
        self.base_contrast: float = base_contrast
        self.desat: float = desat
        self._influence_scale: float = 0.4
        self._spatial_scale_factor: float = 0.02

    def _eval_influence_function(self, x: np.ndarray) -> np.ndarray:
        '''
        Evaluate the influence function on inputs.

        Args:
            x: Input on which the influence function is to be computed.

        Returns:
            np.ndarray: Output of the influence function.
        '''
        x = x / self._influence_scale
        if self.influence_function == 'gaussian':
            return np.exp(-0.5 * x**2)
        elif self.influence_function == 'huber':
            return np.where(np.abs(x) <= 1, 1, 1 / x)
        elif self.influence_function == 'biweight':
            return (1 - x**2)**2

    @staticmethod
    def _interp_function(x: np.ndarray, val: float, interp_size: float) -> np.ndarray:
        '''
        Compute linear interpolation factors.

        Args:
            x: Input to which interpolation factors must be applied.
            val: Anchor value used for interpolation.
            interp_size: Size of interpolation horizon on either side of the anchor.
        '''
        return np.clip(1 - np.abs(x - val) / interp_size, 0, 1)

    def _fast_bilateral_filter(self, img: np.ndarray, minval: Optional[float] = None, maxval: Optional[float] = None) -> np.ndarray:
        '''
        Apply the fast bilateral filter.

        Args:
            img: Image to be filtered.
            minval: Minimum value to be used in the bilateral filter (Default: None).
            maxval: Maximum value to be used in the bilateral filter (Default: None).

        Returns:
            np.ndarray: Filtered image.
        '''
        if minval is None:
            minval = img.min()
        if maxval is None:
            maxval = img.max()
        if maxval - minval < 2*self._influence_scale:
            maxval = minval + max(maxval - minval, 2*self._influence_scale)

        num_segments = int(np.ceil((maxval - minval)/self._influence_scale)) if self.num_segments is None else self.num_segments
        segment_lums = np.linspace(minval, maxval, num_segments)
        segment_size = segment_lums[1] - segment_lums[0]

        img_ds = skimage.transform.rescale(img, 1/self.downsampling_factor, order=0, anti_aliasing=False)
        sigma = self._spatial_scale_factor * max(*img_ds.shape)

        img_filt = np.zeros_like(img)
        for segment_lum in segment_lums:
            influence_map = self._eval_influence_function(img_ds - segment_lum)
            norm_factor_map = sp.ndimage.gaussian_filter(influence_map, sigma)
            weighted_influence_map = sp.ndimage.gaussian_filter(influence_map * img_ds, sigma)
            img_filt_term = skimage.transform.resize(weighted_influence_map / norm_factor_map, img.shape, order=1, anti_aliasing=True)
            img_filt = img_filt + Durand02TMO._interp_function(img, segment_lum, segment_size) * img_filt_term

        return img_filt

    def _base_detail_decomposition(self, img: np.ndarray, minval: Optional[float] = None, maxval: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute the base-detail decomposition.

        Args:
            img: Image to be decomposed.
            minval: Minimum value to be used in the bilateral filter (Default: None).
            maxval: Maximum value to be used in the bilateral filter (Default: None).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Base and detail channels.
        '''
        base = self._fast_bilateral_filter(img, minval, maxval)
        detail = img - base
        return (base, detail)

    def tonemap_frame(self, frame: Frame, minval: Optional[float] = None, maxval: Optional[float] = None) -> Frame:
        '''
        Tone-map a frame using Durand's 2002 TMO.

        Args:
            frame: Frame to be tone-mapped.
            minval: Minimum value to be used in the bilateral filter (Default: None).
            maxval: Maximum value to be used in the bilateral filter (Default: None).

        Returns:
            Frame: Tone-mapped frame.
        '''
        linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
        log_lum = np.log10(linear_yuv_const[..., 0] + 1e-6)
        base, detail = self._base_detail_decomposition(log_lum, minval, maxval)

        base_tm = (base - base.max()) / (base.max() - base.min()) * np.log10(self.base_contrast)

        log_lum_out = base_tm + detail
        lum_out = Frame._lift_to_multichannel(np.clip(10**(log_lum_out) - 1e-6, 0, None))
        lum = Frame._lift_to_multichannel(frame.linear_yuv[..., 0])

        temp_out_frame = Frame(frame.standard)
        temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
        out_frame = self.gamut_map(temp_out_frame)
        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Durand's 2002 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        minval = np.inf
        maxval = -np.inf
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            log_lum = np.log10(linear_yuv_const[..., 0] + 1e-6)
            minval = min(minval, log_lum.min())
            maxval = max(maxval, log_lum.max())
        video.reset()

        out_video = Video(out_filepath, self.out_standard, 'w', video.width, video.height, self.out_format)
        for frame in video:
            out_frame = self.tonemap_frame(frame, minval, maxval)
            out_video.append(out_frame)
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max'):
        '''
        Tone-map a video smoothly using Durand's 2002 TMO and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath)
