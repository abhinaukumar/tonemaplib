from typing import Tuple, Optional, List

import numpy as np
import scipy as sp

from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.cvt_color import rgb2yuv
from .tmo import TMO
from .boitard12 import Boitard12TMO


class Reinhard02TMO(TMO):
    '''
    Implementation of Reinhard's TMO from 2002.
    Adapted from C++ implementation linked in refs.

    Args:
        out_standard (Standard): Standard to which the output frames must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        key (float): Key value to which image is mapped.
        lum_white (float): Smallest value mapped to white in 'global' mode.
        scales (int): Number of scales to use in 'local' mode.
        mode (str): One of 'global' or 'local', denoting whether to use the global or local TMO.
        desat (float): Desaturation parameter for color correction.

    Refs:
        Reinhard, E., Stark, M., Shirley, P., & Ferwerda, J. (2002, July).
        "Photographic tone reproduction for digital images."
        In Proceedings of the 29th annual conference on Computer graphics and interactive techniques (pp. 267-276).

        https://github.com/LuminanceHDR/LuminanceHDR/tree/master/src/TonemappingOperators/reinhard02
        https://github.com/banterle/HDR_Toolbox/blob/master/source_code/Tmo/ReinhardTMO.m
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        key: Optional[float] = 0.18,
        lum_white: Optional[float] = None,
        mode: Optional[str] = 'global',
        desat: Optional[float] = 0.0
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must conform (Default: sRGB)
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            key: Key value to which image is mapped (Default: 0.18)
            lum_white: Smallest value mapped to white in 'global' mode. If None, set to max luminance of the image. (Default: None)
            mode: One of 'global' or 'local', denoting whether to use the global or local TMO (Default: 'global')
            desat: Desaturation parameter for color correction (Default: 0.0).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.key: float = key
        self.lum_white: float = lum_white
        self.mode: str = mode
        self.scales: int = 8
        self.desat: float = desat
        self._eps: float = 0.05
        self._2_phi: float = float(1 << 8)
        self._sigs = np.power(1.6, np.arange(self.scales)) * 0.25

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'key', 'mode', 'desat']

    @staticmethod
    def _get_robust_minmax(data: np.ndarray) -> Tuple[float, float]:
        '''
        Return a robust estimate of the min and max value using the
        1st and 99th percentile of the input data.

        Args:
            data: Data for which robust min and max are to be computed.

        Returns:
            Tuple[float, float]: Robust min and max values.
        '''
        minval = np.percentile(data, 1)
        maxval = np.percentile(data, 99)
        return (minval, maxval)

    @staticmethod
    def _get_white_point(img_lum: np.ndarray, limits: Optional[Tuple[float, float]] = None) -> float:
        '''
        Infer white point from the given luminance image.
        Obtained from the MATLAB codebase.
        Args:
            img_lum: Input luminance image.

        Returns:
            float: Inferred white point.
        '''
        if limits is None:
            minval, maxval = Reinhard02TMO._get_robust_minmax(img_lum)
        else:
            minval, maxval = limits
        return 1.5 * maxval / (32 * minval)

    def _get_exposure(self, img_lum: np.ndarray) -> float:
        '''
        Infer exposure from the given luminance image.
        Obtained from the MATLAB codebase.
        Args:
            img_lum: Input luminance image.

        Returns:
            float: Inferred exposure.
        '''
        minval, maxval = Reinhard02TMO._get_robust_minmax(img_lum)
        log_avg = Frame._get_log_average(img_lum)

        return self._get_exposure_from_stats(minval, log_avg, maxval)

    def _get_exposure_from_stats(self, minval: float, log_avg: float, maxval: float) -> float:
        '''
        Infer exposure from the stats of a luminance image.
        Obtained from the MATLAB codebase.
        Args:
            minval: Minimum value of the range
            log_avg: Log-average of the range
            maxval: Maximum value of the range

        Returns:
            float: Inferred exposure.
        '''
        exposure = self.key * np.power(4, (2*np.log2(log_avg) - np.log2(minval) - np.log2(maxval)) / (np.log2(maxval) - np.log2(minval)))
        return exposure

    def _luminance_mapping(self, img_lum: np.ndarray, exposure: float, log_avg: Optional[float] = None) -> np.ndarray:
        '''
        Perform the "initial luminance mapping".

        Args:
            img_lum: Input "world" luminance image.
            exposure: Exposure value to which the image must be scaled.
            log_avg: Value of log-average to use (Default: None)

        Returns:
            np.ndarray: Scaled luminance image.
        '''
        if log_avg is None:
            log_avg = Frame._get_log_average(img_lum)
        img_scaled_lum = img_lum * exposure / log_avg
        return img_scaled_lum

    def _auto_dodging_and_burning(self, img_scaled_lum: np.ndarray, exposure: float) -> np.ndarray:
        '''
        Perform dodging and burning using Gaussian filtered versions of the scaled image.

        Args:
            img_scaled_lum: Scaled luminance image from luminance mapping.
            exposure: Exposure value of the original image.

        Returns:
            np.ndarray: Response function (V, in the paper)
        '''
        Vs = []
        for sig in self._sigs:
            Vs.append(sp.ndimage.gaussian_filter(img_scaled_lum, sig))

        Vscales = []
        for scale in range(self.scales - 1):
            Vscales.append((Vs[scale] - Vs[scale + 1]) / (self._2_phi * exposure / np.power(1.6, 2*scale) + Vs[scale]))

        mask = np.zeros(img_scaled_lum.shape, dtype=bool)
        img_response = np.zeros(img_scaled_lum.shape)
        for scale in range(self.scales - 1):
            cand_mask = (np.abs(Vscales[scale]) > self._eps)
            assign_mask = (~ mask) & cand_mask
            img_response[assign_mask] = Vs[scale][assign_mask]
            mask |= cand_mask
        img_response[~ mask] = Vs[-1][~ mask]

        return img_response

    def tonemap_frame(self, frame: Frame, log_avg: Optional[float] = None) -> Frame:
        '''
        Tone-map a frame using Reinhard's 2002 TMO.

        Args:
            frame: Frame to be tone-mapped
            log_avg: Log-average value to use for tone-mapping.

        Returns:
            Frame: Tone-mapped frame.
        '''
        linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
        img_lum = linear_yuv_const[..., 0]

        img_scaled_lum = self._luminance_mapping(img_lum, self.key, log_avg)
        lum_white = img_scaled_lum.max() if self.lum_white is None else self.lum_white

        if self.mode == 'global':
            img_response = img_scaled_lum
            img_disp_lum = img_scaled_lum * (1 + img_scaled_lum / lum_white**2) / (1 + img_response)
        elif self.mode == 'local':
            img_response = self._auto_dodging_and_burning(img_scaled_lum, self.key)
            img_disp_lum = img_scaled_lum / (1 + img_response)
        else:
            raise ValueError('Invalid mode. Must be \'global\' or \'local\'')

        temp_out_frame = Frame(frame.standard)
        lum_out = Frame._lift_to_multichannel(img_disp_lum)
        lum = Frame._lift_to_multichannel(img_lum)
        temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
        out_frame = self.gamut_map(temp_out_frame)
        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Reinhard's 2002 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        log_avgs = []
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            log_avgs.append(Frame._get_log_average(linear_yuv_const[..., 0]))
        video.reset()
        log_avg = Frame._get_log_average(np.array(log_avgs))

        out_video = Video(out_filepath, self.out_standard, 'w', video.width, video.height, format=self.out_format)
        for frame in video:
            out_frame = self.tonemap_frame(frame, log_avg)
            out_video.write_frame(out_frame)
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max') -> None:
        '''
        Tone-map a video smoothly using Reinhard's 2002 TMO and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath)
