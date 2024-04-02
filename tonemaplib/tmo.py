from typing import Any, List, Dict, Union, Optional
import os

from videolib import standards
from videolib.standards import Standard
from videolib import Frame, Video


class TMO:
    '''
    Base class from which all TMOs are inherited.

    Args:
        out_standard (Standard): Standard to which the output frames must confirm.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.

    Refs:
        Report ITU-R BT.2407-0 (October 2017)
        "Colour gamut conversion from Recommendation ITU-R BT.2020 to Recommendation ITU-R BT.709".
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: Optional[str] = 'framewise',
        out_format: Optional[str] = 'encoded'
    ) -> None:
        '''
        Initializer. Must be extended by derived classes.

        Args:
            out_standard: Standard to which tone mapped frames must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: compressed).
        '''
        self._video_modes = ['framewise', 'shot', 'smooth']
        self.out_standard: Standard = out_standard
        if video_mode not in self._video_modes:
            raise ValueError(f'Invalid video mode {video_mode}')
        self.video_mode: str = video_mode
        self.out_format: str = out_format

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        raise ['video_mode']

    @property
    def params_dict(self) -> Dict[str, Any]:
        '''
        Return a dictionary containing parameter names and their values that define the TMO.
        '''
        return dict([(param, self.__dict__[param]) for param in self.params])

    def __str__(self) -> str:
        '''
        Return a string representation of the TMO based on params_dict.
        '''
        return '_'.join([self.__class__.__name__] + [key + '_' + str(self.params_dict[key]) for key in self.params_dict])

    @staticmethod
    def _assert_out_filepath_is_valid(out_filepath: str) -> bool:
        '''
        Check if path provided for output file is valid.

        Args:
            out_filepath: Path to output file.

        Returns:
            bool: True if it is a valid path.
        '''
        if not os.path.isdir(os.path.dirname(out_filepath)):
            raise ValueError('{} is not a valid file path'.format(out_filepath))

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tonemap a frame. Must be overloaded by derived classes.
        '''
        raise NotImplementedError

    def tonemap_video_framewise(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map video frame-wise using the TMO.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        for frame in video:
            out_frame = self.tonemap_frame(frame)
            out_video.write_frame(out_frame)
        out_video.close()

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map shot using the TMO.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        raise NotImplementedError

    def tonemap_video_smooth(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map video smoothly using the TMO and a smoothing mechanism.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        raise NotImplementedError

    def tonemap_video(self, video: Video, out_filepath: str, **kwargs) -> None:
        if self.video_mode == 'framewise':
            self.tonemap_video_framewise(video, out_filepath, **kwargs)
        elif self.video_mode == 'shot':
            self.tonemap_shot(video, out_filepath, **kwargs)
        elif self.video_mode == 'smooth':
            self.tonemap_video_smooth(video, out_filepath, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Union[Frame, None]:
        '''
        Make objects callable by calling the appropriate tonemap routine
        '''
        if isinstance(args[0], Video):
            return self.tonemap_video(*args, **kwargs)
        elif isinstance(args[0], Frame):
            return self.tonemap_frame(*args, **kwargs)
        else:
            raise ValueError('Expected first argument to be of type Video or Frame')

    def gamut_map(self, frame):
        '''
        Map gamut linearly using XYZ as the intermediate space.
        '''
        out_frame = Frame(self.out_standard)
        if (frame.standard.primaries != self.out_standard.primaries):
            out_frame.xyz = frame.xyz
        else:
            out_frame.linear_yuv = frame.linear_yuv
        return out_frame
