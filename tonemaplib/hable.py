from typing import Optional, Union, List

import numpy as np

from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.cvt_color import rgb2yuv
from .tmo import TMO
from .boitard12 import Boitard12TMO


class HableTMO(TMO):
    '''
    Implementation of Hable's Uncharted TMO.

    Args:
        out_standard (Standard): Standard to which output must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        desat (float): Desaturation parameter for color correction.
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        desat: Optional[float] = 2e-4
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which output must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            desat: Desaturation parameter for color correction (Default: 2e-4).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.desat = desat

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'desat']

    @staticmethod
    def _hable_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Unnormalized tone curve.

        Args:
            x: Input channel value(s)

        Returns:
            Union[int, float, np.ndarray]: Transformed channel values
        '''

        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30
        return (x*(A*x + C*B) + D*E)/(x*(A*x + B) + D*F) - E/F

    def tonemap_frame(self, frame: Frame, ffmpeg_version: Optional[bool] = False, maxval: Optional[float] = None) -> Frame:
        '''
        Tone-map a frame using Hable's Uncharted TMO.

        Args:
            frame: Frame to be tone mapped.
            ffmpeg_version: Flag to use FFMPEG's version of the algorithm. (Default: False)
            maxval: Maximum value to be used when applying Hable's tone curve (Default: None)

        Returns:
            Frame: Tone mapped frame.
        '''
        temp_out_frame = Frame(frame.standard)
        linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
        lum = linear_yuv_const[..., 0]
        if ffmpeg_version:
            if self.desat > 0:
                overbright = Frame._lift_to_multichannel(np.clip(lum - self.desat, 1e-6, None) / np.clip(lum, 1e-6, None))
            else:
                overbright = 1
            rgb = overbright * frame.linear_rgb + (1 - overbright) * Frame._lift_to_multichannel(lum)
            lum = np.clip(rgb.max(-1), 1e-6, None)

        if maxval is None:
            lum_max = lum.max()
        else:
            lum_max = maxval
        lum_out = HableTMO._hable_function(lum) / HableTMO._hable_function(lum_max)

        temp_out_frame.linear_rgb = \
            frame.linear_rgb * Frame._lift_to_multichannel(np.divide(lum_out, lum, where=(lum != 0), out=np.zeros_like(lum))) if ffmpeg_version else \
            ((frame.linear_rgb / (Frame._lift_to_multichannel(lum) + 1e-10) - 1) * (1 - self.desat) + 1) * Frame._lift_to_multichannel(lum_out)
        out_frame = self.gamut_map(temp_out_frame)
        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str, ffmpeg_version: Optional[bool] = False) -> None:
        '''
        Tone-map a shot using Hable's Uncharted TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        maxval = -np.inf
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            lum = linear_yuv_const[..., 0]
            if ffmpeg_version:
                if self.desat > 0:
                    overbright = Frame._lift_to_multichannel(np.clip(lum - self.desat, 1e-6, None) / np.clip(lum, 1e-6, None))
                else:
                    overbright = 1
                rgb = overbright * frame.linear_rgb + (1 - overbright) * Frame._lift_to_multichannel(lum)
                lum = np.clip(rgb.max(-1), 1e-6, None)
            maxval = max(maxval, lum.max())
        video.reset()

        out_video = Video(out_filepath, self.out_standard, 'w', video.width, video.height, self.out_format)
        for frame in video:
            out_frame = self.tonemap_frame(frame, maxval=maxval)
            out_video.write_frame(out_frame)
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max', ffmpeg_version: Optional[bool] = True) -> None:
        '''
        Tone-map a video smoothly using Hable's Uncharted TMO and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath, ffmpeg_version=ffmpeg_version)
