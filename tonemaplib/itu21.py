from typing import Optional, Union, List

import numpy as np

from videolib.cvt_color import rgb2yuv
from videolib import Frame, Video, standards
from videolib.standards import Standard
from .tmo import TMO
from .boitard12 import Boitard12TMO


class ITU21TMO(TMO):
    '''
    Implementation of "Approach A" from ITU BT.2446.

    Args:
        out_standard (Standard): Standard to which output frames must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        peak_hdr (float): Nominal peak display of the HDR display.

    Refs:
        Report ITU-R BT.2446-1 (March 2021)
        "Methods for conversion of high dynamic range content to standard dynamic range content and vice-versa".
    '''
    params = ['video_mode', 'peak_hdr']
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        peak_hdr: Optional[float] = 1e4
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which output must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            peak_hdr: Nominal peak display of the HDR display (Default: 1e4).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.peak_hdr: float = peak_hdr
        peak_sdr = 1e2
        self._coeff_hdr = ITU21TMO._coeff_function(self.peak_hdr)
        self._coeff_sdr = ITU21TMO._coeff_function(peak_sdr)

    @staticmethod
    def _coeff_function(lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute the coefficient 'rho' as a function of luminance.

        Args:
            lum: Luminance.

        Returns:
            Union[float, np.ndarray]: Coefficient.
        '''
        return 1 + 32 * np.power(lum*1e-4, 1/2.4)

    @staticmethod
    def _exp_function(log_lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute the exponent to be applied in step 3 of tone mapping.

        Args:
            lum: Intermediate log-luminance after step 1.

        Returns:
            Union[float, np.ndarray]: Exponents.
        '''
        return np.where(
            log_lum <= 0.7399,
            1.0770 * log_lum,
            np.where(log_lum < 0.9909, -1.1510 * log_lum**2 + 2.7811 * log_lum - 0.6302, 0.5 * log_lum + 0.5)
        )

    def _tone_curve(self, lum: np.ndarray) -> np.ndarray:
        '''
        Apply the tone curve to the given input luma.

        Args:
            lum: HDR luma image.

        Returns:
            np.ndarray: Tone mapped luma image.
        '''
        log_lum = np.log(1 + (self._coeff_hdr-1)*lum) / np.log(self._coeff_hdr)
        lum_exponents = ITU21TMO._exp_function(log_lum)
        lum_sdr = (np.power(self._coeff_sdr, lum_exponents) - 1) / (self._coeff_sdr - 1)
        return lum_sdr

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tone-map a frame using "Approach A" from ITU BT.2446.

        Args:
            frame: Frame to be tone mapped.

        Returns:
            Frame: Tone mapped frame.
        '''
        rgb = frame.linear_rgb
        nonlin_rgb = np.power(rgb, 1/2.4)
        nonlin_yuv = rgb2yuv(nonlin_rgb, frame.standard, range=1)
        lum = nonlin_yuv[..., 0]

        lum_out = self._tone_curve(lum)
        scale = np.divide(lum_out, lum, out=np.zeros_like(lum), where=(lum != 0)) / 1.1
        cb_out = scale * (nonlin_rgb[..., 2] - lum) / 1.8814
        cr_out = scale * (nonlin_rgb[..., 0] - lum) / 1.4746
        lum_out = lum_out - np.clip(0.1*cr_out, 0, None)

        temp_out_frame = Frame(standards.rec_2020)
        temp_out_frame.yuv = np.stack([lum_out, cb_out + 0.5, cr_out + 0.5], axis=-1) * standards.rec_2020.range
        out_frame = self.gamut_map(temp_out_frame)
        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using "Approach A" from ITU BT.2446.

        Args:
            video: Video to be tone mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        # First pass
        meanval = 0
        for frame in video:
            rgb = frame.linear_rgb
            nonlin_rgb = np.power(rgb, 1/2.4)
            nonlin_yuv = rgb2yuv(nonlin_rgb, frame.standard, range=1)
            lum = nonlin_yuv[..., 0]
            meanval += np.log(lum).mean()
        meanval /= video.num_frames
        video.reset()

        # Second pass
        out_video = Video(out_filepath, self.out_standard, 'w', video.width, video.height, self.out_format)
        for frame in video:
            rgb = frame.linear_rgb
            nonlin_rgb = np.power(rgb, 1/2.4)
            nonlin_yuv = rgb2yuv(nonlin_rgb, frame.standard, range=1)
            lum = nonlin_yuv[..., 0]
            lum_in = np.log(lum)
            lum_in = lum_in * meanval / lum_in.mean()
            lum_in = np.exp(lum_in)

            lum_out = self._tone_curve(lum_in)
            scale = np.divide(lum_out, lum, out=np.zeros_like(lum), where=(lum != 0)) / 1.1
            cb_out = scale * (nonlin_rgb[..., 2] - lum) / 1.8814
            cr_out = scale * (nonlin_rgb[..., 0] - lum) / 1.4746
            lum_out = lum_out - np.clip(0.1*cr_out, 0, None)

            temp_out_frame = Frame(standards.rec_2020)
            temp_out_frame.yuv = np.stack([lum_out, cb_out + 0.5, cr_out + 0.5], axis=-1) * standards.rec_2020.range
            out_frame = self.gamut_map(temp_out_frame)
            out_video.append(out_frame)
        return out_frame

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max') -> None:
        '''
        Tone-map a video using "Approach A" from ITU BT.2446 and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        Returns:
            Frame: Tone-mapped frame.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath)
