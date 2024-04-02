from typing import Tuple, List, Optional, Union
import os

import numpy as np

from videolib import Video, Frame, standards
from .tmo import TMO
from .eilertsen15 import Eilertsen15TMO


class Boitard12TMO(TMO):
    '''
    Implementation of Boitard's TMO from 2012.

    Args:
        out_standard (Standard): Standard to which the output frames must conform.
        video_mode (str): Method used to tone map videos. Must be either 'framewise' or 'smooth' (Default: 'smooth').
        out_format (str): Format to which output video must be written.
        base_tmo (TMO): Base TMO on top of which Boitard12 is applied.
        scale_method (str): Scaling method used during post-processing. Must be either 'mean' or 'max'.

    Refs:
        Boitard, R., Bouatouch, K., Cozot, R., Thoreau, D., & Gruson, A. (15 October 2012)
        "Temporal coherency for video tone mapping."
        Proc. SPIE 8499, Applications of Digital Image Processing XXXV, 84990D
    '''
    params = ['video_mode', 'base_tmo', 'scale_method']
    def __init__(
        self,
        out_standard: Optional[standards.Standard] = standards.sRGB,
        out_format: Optional[str] = 'encoded',
        video_mode: str = 'smooth',
        base_tmo: TMO = None,
        scale_method: Optional[str] = 'max'
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            base_tmo: Base TMO on top of which Boitard12 is applied (Default: Eilertsen15TMO(sRGB)).
            scale_method: Scaling method used during post-processing. Must be either 'mean' or 'max' (Default: 'max').
        '''
        if base_tmo is None:
            self.base_tmo: TMO = Eilertsen15TMO(out_standard)
        else:
            self.base_tmo: TMO = base_tmo
        if out_standard != self.base_tmo.out_standard:
            raise ValueError('Base TMO must have the same out_standard as Boitard12')
        if scale_method not in ['mean', 'max']:
            raise ValueError('Invalid scale method. Must be either \'mean\' or \'max\'')
        if video_mode != 'smooth':
            raise ValueError('Invalid video mode. Must be \'smooth\'')
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.scale_method: str = scale_method

    def _first_pass(self, video: Video, temp_video: Video, **kwargs) -> Tuple[List[float], ...]:
        '''
        Conduct the first pass of tone-mapping.
        Uses base TMO to tone-map each frame independently, and records keys.

        Args:
            video: Video object corresponding to the HDR video.
            temp_video: Video object corresponding to the temporary video.

        Returns:
            Tuple[List[float], ...]: Lists containing keys of frames from the HDR video, and if 'max' scaling is used, the tonemapped LDR video.
        '''
        ldr_keys = []
        if self.scale_method == 'max':
            hdr_keys = []
        for frame in video:
            frame_tonemapped = self.base_tmo(frame, **kwargs)
            hdr_keys.append(Frame._get_log_average(frame.linear_yuv[..., 0]))
            if self.scale_method == 'max':
                ldr_keys.append(Frame._get_log_average(frame_tonemapped.linear_yuv[..., 0]))
            temp_video.write_frame(frame_tonemapped)

        if self.scale_method == 'max':
            return hdr_keys, ldr_keys
        else:
            return ldr_keys

    def _get_keys(self, video: Video) -> List[float]:
        '''
        Get the key of each frame of a video.

        Args:
            video: Video object containing the input video.

        Returns:
            List[float]: List containing keys of frames of the input video.
        '''
        return [Frame._get_log_average(frame.linear_yuv[..., 0]) for frame in video]

    def _second_pass(self, keys: Union[List[float], Tuple[List[float], ...]], temp_video: Video, out_video: Video) -> None:
        '''
        Conduct the second pass of tone-mapping.

        Args:
            keys: Keys of each frame of the HDR (and LDR, when using max scaling) video(s).
            temp_video: Video object corresponding to the temporary video.
            out_video: Video object corresponding to the output video.
        '''
        if self.scale_method == 'mean':
            ldr_keys = keys
            video_key = np.exp(np.mean(np.log(ldr_keys)))

            for frame, frame_key in zip(temp_video, ldr_keys):
                scaled_frame = Frame(self.out_standard)
                scale_factor = frame_key / (frame_key + video_key)
                scaled_frame.linear_rgb = frame.linear_rgb * scale_factor
                out_video.write_frame(scaled_frame)

        elif self.scale_method == 'max':
            hdr_keys, ldr_keys = keys
            max_hdr_key = np.max(hdr_keys)
            max_ldr_key = np.max(ldr_keys)

            for frame, hdr_key, ldr_key in zip(temp_video, hdr_keys, ldr_keys):
                scaled_frame = Frame(self.out_standard)
                scale_factor = (hdr_key * max_ldr_key) / (ldr_key * max_hdr_key)
                scaled_frame.linear_rgb = frame.linear_rgb * scale_factor
                out_video.write_frame(scaled_frame)

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tone-map a frame using Boitard12 TMO. Equivalent to tone mapping using the base TMO.

        Args:
            frame: Frame to be tone-mapped.

        Returns:
            Frame: Tone-mapped frame.
        '''
        return self.base_tmo(frame)

    def tonemap_video_smooth(self, video: Video, out_filepath: str, **kwargs) -> None:
        '''
        Tone-map a video smoothly using Boitard12 TMO.

        Args:
            video: Video object containing the input HDR video.
            out_filepath: Path to which output file must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)

        name, ext = out_filepath.rsplit('.', 1)
        temp_filepath = name + '_temp.' + ext

        temp_video = Video(temp_filepath, self.out_standard, mode='w', format=self.out_format)
        keys = self._first_pass(video, temp_video, **kwargs)
        temp_video.close()

        with Video(temp_filepath, self.out_standard, mode='r', width=video.width, height=video.height, format=self.out_format) as temp_video:
            out_video = Video(out_filepath, self.out_standard, mode='w', format=self.out_format)
            self._second_pass(keys, temp_video, out_video)
            out_video.close()
        os.remove(temp_filepath)

    def postprocess_video(self, video: Video, src_video: Video, out_filepath: str) -> None:
        '''
        Postprocess a video with respect to a source video using Boitard12 TMO.

        Args:
            video: Video object containing the input SDR video.
            src_video: Video object containing the source HDR video.
            out_filepath: Path to which output file must be written.
        '''
        ldr_keys = self._get_keys(video)
        video.reset()
        if self.scale_method == 'max':
            hdr_keys = self._get_keys(video)
            src_video.reset()
            keys = (hdr_keys, ldr_keys)
        else:
            keys = ldr_keys

        out_video = Video(out_filepath, self.out_standard, mode='w', format=self.out_format)
        self._second_pass(keys, video, out_video)
        out_video.close()

    def postprocess_video_using_keys(self, video: Video, hdr_keys: np.ndarray, out_filepath: str) -> None:
        '''
        Postprocess a video with respect to a source video using Boitard12 TMO.

        Args:
            video: Video object containing the input SDR video.
            hdr_keys: Array containing keys of each frames of the source HDR video
            out_filepath: Path to which output file must be written.
        '''
        ldr_keys = self._get_keys(video)
        video.reset()
        if self.scale_method == 'max':
            keys = (hdr_keys, ldr_keys)
        else:
            keys = ldr_keys

        out_video = Video(out_filepath, self.out_standard, mode='w', format=self.out_format)
        self._second_pass(keys, video, out_video)
        out_video.close()
