from typing import List, Optional, Tuple, Union, Dict, Any
import os

import numpy as np
import scipy as sp
import pywt

from videolib import Frame, Video, standards, cvt_color
from videolib.standards import Standard
from videolib.buffer import CircularBuffer
from .tmo import TMO
from .boitard12 import Boitard12TMO


class Shan12TMO(TMO):
    '''
    Implementation of Shan's TMO from 2012.

    Args:
        out_standard (Standard): Standard to which the output frames must confirm.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        wavelet (Union[str, Wavelet]): Wavelet basis to use.
        levels (int): Number of wavelet levels to use.
        temp_dir (str): Directory where temporary files will be written.
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        wavelet: Union[str, pywt.Wavelet] = 'haar',
        levels: int = 4,
        temp_dir: Optional[str] = '/tmp'
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must confirm (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            wavelet: Wavelet basis to use (Default: 'haar').
            levels: Number of wavelet levels to use (Default: 4).
            temp_dir: Directory where temporary files will be written (Default: 'temp').
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.temp_dir = temp_dir
        self.wavelet = wavelet if isinstance(wavelet, pywt.Wavelet) else pywt.Wavelet(wavelet)
        self.levels: int = levels

        # Define decomposition and reconstruction filters.
        self._base_decomp_lowpass_kernel: np.ndarray = np.array(self.wavelet.dec_lo)
        self._base_decomp_highpass_kernel: np.ndarray = np.array(self.wavelet.dec_hi)
        self._base_recon_lowpass_kernel: np.ndarray = np.array(self.wavelet.rec_lo)
        self._base_recon_highpass_kernel: np.ndarray = np.array(self.wavelet.rec_hi)

        # Upsample to for stationary wavelet transform a trous.
        self._lowpass_decomp_kernels: List = self._upsample(self._base_decomp_lowpass_kernel, self.levels)
        self._highpass_decomp_kernels: List = self._upsample(self._base_decomp_highpass_kernel, self.levels)
        self._lowpass_recon_kernels: List = self._upsample(self._base_recon_lowpass_kernel, self.levels)
        self._highpass_recon_kernels: List = self._upsample(self._base_recon_highpass_kernel, self.levels)

        # Define Gaussian filters.
        self._gaussian_sigmas = 1 * 2**np.arange(self.levels)
        self._gaussian_kernels = [np.exp(-0.5*np.arange(-np.ceil(3*sigma), np.ceil(3*sigma)+1)**2 / sigma**2) for sigma in self._gaussian_sigmas]
        self._gaussian_kernels = [kernel / kernel.sum() for kernel in self._gaussian_kernels]

        # Parameters defining the non-linearity
        self._eps: float = 1e-6
        self._gamma: float = 0.6
        self._level_gammas: np.ndarray = np.minimum(np.arange(self.levels)*0.05 + self._gamma, 0.9)

        self._level_mults: np.ndarray = np.array([1.0, 0.8] + [0.6]*max(self.levels-2, 0))
        self._subband_names = ['approx_lo', 'approx_hi', 'hor_lo', 'hor_hi', 'ver_lo', 'ver_hi', 'diag_lo', 'diag_hi']

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'wavelet', 'levels']

    @property
    def params_dict(self) -> Dict[str, Any]:
        '''
        Return a dictionary containing parameter names and their values that define the TMO.
        '''
        return {'video_mode': self.video_mode, 'wavelet': self.wavelet.name, 'levels': self.levels}

    @staticmethod
    def _upsample(kernel: np.ndarray, levels: int) -> List[np.ndarray]:
        '''
        Upsample the given kernel a given number of times.

        Args:
            kernel: Kernel to be upsampled.
            levels: Number of times to upsample.

        Returns:
            List[np.ndarray]: List of upsampled kernels.
        '''
        if levels == 0:
            return []
        else:
            up_kernel = np.zeros((2*kernel.shape[0]-1,))
            up_kernel[::2] = kernel
            return [kernel] + Shan12TMO._upsample(up_kernel, levels-1)

    @staticmethod
    def _temporal_filter(buf: CircularBuffer, kernel: np.ndarray) -> np.ndarray:
        '''
        Compute the output of an edge-aware temporal filter for the window represented the circular buffer.

        Args:
            buf: Circular buffer representing the convolution window.
            kernel: Kernel to be applied

        Returns:
            np.ndarray: Pixelwise temporally filtered image.
        '''
        img_filt = np.zeros_like(buf.top())
        for img, weight in zip(buf, kernel):
            img_filt = img_filt + img * weight
        return img_filt

    @staticmethod
    def _edge_aware_temporal_filter(buf: CircularBuffer, kernel: np.ndarray) -> np.ndarray:
        '''
        Compute the output of an edge-aware temporal filter for the window represented the circular buffer.

        Args:
            buf: Circular buffer representing the convolution window.
            kernel: Kernel to be applied

        Returns:
            np.ndarray: Pixelwise temporally filtered image.
        '''
        img_ref = buf.top()
        img_filt = np.zeros_like(img_ref)
        for img, weight in zip(buf, kernel):
            edge_mask = np.abs(img_ref - img) < 0.01
            img_filt = img_filt + np.where(edge_mask, img, img_ref) * weight
        return img_filt

    def _edge_aware_spatio_temp_wavelet_decomp(self, filename: str, level: int, num_frames: int, frame_width: int, frame_height: int) -> None:
        '''
        Compute a multi-level temporal edge-aware spatiotemporal wavelet decomposition. Save subbands into binary files.

        Args:
            filename: Name of video being processed. Used to infer and set temporary file paths.
            level: Wavelet level to compute (1-based indexing).
            num_frames: Number of frames in the video.
            frame_width: Width of each frame.
            frame_height: Height of each frame.
        '''
        if level == 0:
            return

        # Decompose up to the previous level to use as input for the current level.
        self._edge_aware_spatio_temp_wavelet_decomp(filename, level-1, num_frames, frame_width, frame_height)

        lowpass_kernel = self._lowpass_decomp_kernels[level-1]
        highpass_kernel = self._highpass_decomp_kernels[level-1]

        ksize = len(lowpass_kernel)
        frame_stride = frame_width * frame_height * 4  # Assuming 32-bit floats.

        # Create buffers to hold spatially filtered subbands.
        approx_buf = CircularBuffer(ksize)
        hor_buf = CircularBuffer(ksize)
        ver_buf = CircularBuffer(ksize)
        diag_buf = CircularBuffer(ksize)
        spatial_subband_bufs = [approx_buf, hor_buf, ver_buf, diag_buf]

        # Create file objects to store subbands on disk.
        input_file = open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_approx_lo.subband'.format(filename, level-1)), 'rb')
        subband_files = [open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}.subband'.format(filename, level, subband_name)), 'wb') for subband_name in self._subband_names]
        for frame_ind in range(num_frames):
            input_file.seek(int(frame_ind * frame_stride))
            img = np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
            lo_ax0 = sp.ndimage.convolve1d(img, lowpass_kernel, axis=0, mode='wrap')
            hi_ax0 = sp.ndimage.convolve1d(img, highpass_kernel, axis=0, mode='wrap')

            approx_buf.check_append(sp.ndimage.convolve1d(lo_ax0, lowpass_kernel, axis=1, mode='wrap'))
            hor_buf.check_append(sp.ndimage.convolve1d(hi_ax0, lowpass_kernel, axis=1, mode='wrap'))
            ver_buf.check_append(sp.ndimage.convolve1d(lo_ax0, highpass_kernel, axis=1, mode='wrap'))
            diag_buf.check_append(sp.ndimage.convolve1d(hi_ax0, highpass_kernel, axis=1, mode='wrap'))

            for subband_ind, (subband_name, output_file) in enumerate(zip(self._subband_names, subband_files)):
                if subband_name[-2:] == 'lo':
                    filter_kernel = lowpass_kernel
                elif subband_name[-2:] == 'hi':
                    filter_kernel = highpass_kernel
                else:
                    raise ValueError('Invalid subband name')
                self._edge_aware_temporal_filter(spatial_subband_bufs[subband_ind//2], filter_kernel).astype('float32').tofile(output_file)

        for buf in spatial_subband_bufs:
            buf.clear()

        for output_file in subband_files:
            output_file.close()

    def _spatial_wavelet_decomp(self, img: np.ndarray, level: int) -> List[Tuple[np.ndarray, ...]]:
        '''
        Compute a multi-level spatial wavelet decomposition.

        Args:
            img: Image for which the decomposition is to be computed.
            level: Wavelet level to compute (1-based indexing).

        Returns:
            List[Tuple[np.ndarray, ...]]: List of tuples, each containing subbands for one level, following PyWavelets' format.
        '''
        if level == 1:
            partial_decomp = []
            prev_approx = img
        else:
            partial_decomp = self._spatial_wavelet_decomp(img, level-1)
            prev_approx = partial_decomp[-1][0]
            partial_decomp[-1] = partial_decomp[-1][1:]

        lowpass_kernel = self._lowpass_decomp_kernels[level-1]
        highpass_kernel = self._highpass_decomp_kernels[level-1]

        lo_ax0 = sp.ndimage.convolve1d(prev_approx, lowpass_kernel, axis=0, mode='wrap')
        hi_ax0 = sp.ndimage.convolve1d(prev_approx, highpass_kernel, axis=0, mode='wrap')

        approx = sp.ndimage.convolve1d(lo_ax0, lowpass_kernel, axis=1, mode='wrap')
        hor = sp.ndimage.convolve1d(hi_ax0, lowpass_kernel, axis=1, mode='wrap')
        ver = sp.ndimage.convolve1d(lo_ax0, highpass_kernel, axis=1, mode='wrap')
        diag = sp.ndimage.convolve1d(hi_ax0, highpass_kernel, axis=1, mode='wrap')

        return partial_decomp + [(approx, hor, ver, diag)]

    def _edge_aware_spatio_temp_wavelet_recon(self, filename: str, level: int, num_frames: int, frame_width: int, frame_height: int) -> None:
        '''
        Reconstruct a multi-level temporal edge-aware spatiotemporal wavelet decomposition. Save reconstructed frames into a binary file.

        Args:
            filename: Name of video being processed. Used to infer and set temporary file paths.
            level: Wavelet level to compute (1-based indexing).
            num_frames: Number of frames in the video.
            frame_width: Width of each frame.
            frame_height: Height of each frame.
        '''
        if level == self.levels:
            return

        # Reconstruct next level to use it as the current level's approximation-low subband
        self._edge_aware_spatio_temp_wavelet_recon(filename, level+1, num_frames, frame_width, frame_height)

        lowpass_kernel = self._lowpass_recon_kernels[level]
        highpass_kernel = self._highpass_recon_kernels[level]

        ksize = len(lowpass_kernel)
        frame_stride = frame_width * frame_height * 4  # Assuming 32-bit floats.
        # Create buffer to hold spatially filtered subbands.
        buf = CircularBuffer(ksize)

        # Create file objects to read subbands from disk.
        subband_files = [
            open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}.subband'.format(filename, level+1, subband_name)), 'rb')
            for subband_name in self._subband_names
        ]

        # Create file objects to write filtered subbands to disk.
        filtered_subband_files = [
            open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}_recon_filtered.subband'.format(filename, level+1, subband_name)), 'wb')
            for subband_name in self._subband_names
        ]

        for subband_name, input_file, output_file in zip(self._subband_names, subband_files, filtered_subband_files):
            for frame_ind in range(num_frames):
                input_file.seek(int(frame_ind * frame_stride))
                buf.check_append(np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width))

                # Choose kernels along the temporal and the two spatial axes.
                if subband_name[-2:] == 'lo':
                    temp_kernel = lowpass_kernel
                elif subband_name[-2:] == 'hi':
                    temp_kernel = highpass_kernel
                else:
                    raise ValueError('Invalid subband name.')

                if 'approx' in subband_name or 'ver' in subband_name:
                    spat_kernel1 = lowpass_kernel
                elif 'hor' in subband_name or 'diag' in subband_name:
                    spat_kernel1 = highpass_kernel
                else:
                    raise ValueError('Invalid subband name.')

                if 'approx' in subband_name or 'hor' in subband_name:
                    spat_kernel2 = lowpass_kernel
                elif 'ver' in subband_name or 'diag' in subband_name:
                    spat_kernel2 = highpass_kernel
                else:
                    raise ValueError('Invalid subband name.')

                out_subband = self._edge_aware_temporal_filter(buf, temp_kernel)
                out_subband = sp.ndimage.convolve1d(sp.ndimage.convolve1d(out_subband, spat_kernel1, axis=0, mode='wrap'), spat_kernel2, axis=1, mode='wrap')
                out_subband.tofile(output_file)
            buf.clear()

        # Create file objects to read filtered subbands from disk.
        filtered_subband_files = [
            open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}_recon_filtered.subband'.format(filename, level+1, subband_name)), 'rb')
            for subband_name in self._subband_names
        ]

        output_file = open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_approx_lo.subband'.format(filename, level)), 'wb')
        for frame_ind in range(num_frames):
            recon = np.zeros((frame_height, frame_width), dtype=np.float32)
            for input_file in filtered_subband_files:
                input_file.seek(int(frame_ind * frame_stride))
                recon = recon + 0.125 * np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
            recon.tofile(output_file)
        output_file.close()

        # Remove fitered subband files.
        for subband_name in self._subband_names:
            os.remove(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}_recon_filtered.subband'.format(filename, level+1, subband_name)))

    def _spatial_wavelet_recon(self, decomp: List[Tuple[np.ndarray, ...]], level: int) -> np.ndarray:
        '''
        Reconstruct a multi-level spatial wavelet decomposition.

        Args:
            decomp: Wavelet decomposition.
            level: Wavelet level to compute (1-based indexing).

        Returns:
            np.ndarray: Reconstructed image.
        '''
        if level == self.levels:
            return decomp[-1][0]

        # Reconstruct next level to use it as the current level's approximation-low subband
        approx = self._spatial_wavelet_recon(decomp, level+1)
        hor, ver, diag = decomp[level][-3:]

        lowpass_kernel = self._lowpass_recon_kernels[level]
        highpass_kernel = self._highpass_recon_kernels[level]

        approx_filt = sp.ndimage.convolve1d(sp.ndimage.convolve1d(approx, lowpass_kernel, axis=0, mode='wrap'), lowpass_kernel, axis=1, mode='wrap')
        hor_filt = sp.ndimage.convolve1d(sp.ndimage.convolve1d(hor, highpass_kernel, axis=0, mode='wrap'), lowpass_kernel, axis=1, mode='wrap')
        ver_filt = sp.ndimage.convolve1d(sp.ndimage.convolve1d(ver, lowpass_kernel, axis=0, mode='wrap'), highpass_kernel, axis=1, mode='wrap')
        diag_filt = sp.ndimage.convolve1d(sp.ndimage.convolve1d(diag, highpass_kernel, axis=0, mode='wrap'), highpass_kernel, axis=1, mode='wrap')

        return 0.25*(approx_filt + hor_filt + ver_filt + diag_filt)

    def _compute_spatio_temp_activity_map(self, filename: str, num_frames: int, frame_width: int, frame_height: int) -> None:
        '''
        Compute the aggregated spatio-temporal activity map.

        Args:
            filename: Name of video being processed. Used to infer and set temporary file paths.
            num_frames: Number of frames in the video.
            frame_width: Width of each frame (Default 1920).
            frame_height: Height of each frame (Default 1080).
        '''
        subband_files = [
            [open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}.subband'.format(filename, level+1, subband_name)), 'rb')
                if (subband_name != 'approx_lo' or level == self.levels-1) else None
                for subband_name in self._subband_names]
            for level in range(self.levels)
        ]

        for level in range(self.levels-1):
            # Remove unused approx_lo subband from all but the last level.
            del subband_files[level][0]

        subband_activity_filenames = [
            os.path.join(self.temp_dir, 'shan12_{}_{}.activity'.format(filename, subband_name)) for subband_name in self._subband_names
        ]

        activity_file_path = os.path.join(self.temp_dir, 'shan12_{}.activity'.format(filename))
        activity_tempfile_path = os.path.join(self.temp_dir, 'shan12_{}.activity.temp'.format(filename))

        # Initialize activity file with zeros
        zeros = np.zeros((frame_height, frame_width), dtype='float32')
        activity_file = open(activity_file_path, 'wb')
        for frame_ind in range(num_frames):
            zeros.tofile(activity_file)
        activity_file.close()

        frame_stride = frame_width * frame_height * 4  # Assuming 32-bit floats.

        for level, (subband_files_level, gaussian_kernel) in enumerate(zip(subband_files, self._gaussian_kernels)):
            ksize = len(gaussian_kernel)
            buf = CircularBuffer(ksize)
            subband_activity_files_level = [
                open(subband_activity_filename, 'wb') for subband_activity_filename in
                (subband_activity_filenames[1:] if level < self.levels-1 else subband_activity_filenames)
            ]

            for input_file, output_file in zip(subband_files_level, subband_activity_files_level):
                for frame_ind in range(num_frames):
                    input_file.seek(int(frame_ind * frame_stride))
                    img = np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
                    spat_activity = sp.ndimage.convolve1d(sp.ndimage.convolve1d(np.abs(img), gaussian_kernel, axis=0, mode='wrap'), gaussian_kernel, axis=1, mode='wrap')
                    buf.check_append(spat_activity)
                    self._edge_aware_temporal_filter(buf, gaussian_kernel).astype('float32').tofile(output_file)
                buf.clear()
                input_file.close()
                output_file.close()

            activity_file = open(activity_file_path, 'rb')
            subband_activity_files_level = [
                open(subband_activity_filename, 'rb') for subband_activity_filename in
                (subband_activity_filenames[1:] if level < self.levels-1 else subband_activity_filenames)
            ]
            activity_tempfile = open(activity_tempfile_path, 'wb')

            for frame_ind in range(num_frames):
                activity_file.seek(int(frame_ind * frame_stride))
                activity_aggregate = np.fromfile(activity_file, np.float32, (frame_width * frame_height)).reshape((frame_height, frame_width))
                for input_file in subband_activity_files_level:
                    input_file.seek(int(frame_ind * frame_stride))
                    activity = np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
                    activity_aggregate = activity_aggregate + activity
                activity_aggregate.astype('float32').tofile(activity_tempfile)

            activity_file.close()
            activity_tempfile.close()
            # Remove subband activity files.
            for input_file in subband_activity_files_level:
                input_file.close()
                os.remove(input_file.name)
            # Replace activity file with updated version.
            os.remove(activity_file_path)
            os.rename(activity_tempfile_path, activity_file_path)

    def _compute_spatial_activity_map(self, decomp: List[Tuple[np.ndarray, ...]]) -> np.ndarray:
        '''
        Compute the aggregated spatial activity map.

        Args:
            decomp: Wavelet decomposition.

        Returns:
            np.ndarray: Aggregated spatial acitvity map.
        '''
        spat_activity = np.zeros_like(decomp[0][0])
        for decomp_level, gaussian_kernel in zip(decomp, self._gaussian_kernels):
            for subband in decomp_level:
                spat_activity = spat_activity + sp.ndimage.convolve1d(sp.ndimage.convolve1d(np.abs(subband), gaussian_kernel, axis=0, mode='wrap'), gaussian_kernel, axis=1, mode='wrap')
        return spat_activity

    def _compute_spatial_activity_maps(self, decomp: List[Tuple[np.ndarray, ...]]) -> List[Tuple[np.ndarray, ...]]:
        '''
        Compute subband-wise spatial activity maps.

        Args:
            decomp: Wavelet decomposition.

        Returns:
            List[Tuple[np.ndarray, ...]]: Subband-wise spatial activity maps.
        '''
        spat_activities = []
        for decomp_level, gaussian_kernel in zip(decomp, self._gaussian_kernels):
            spat_activities_level = []
            for subband in decomp_level:
                spat_activities_level.append(sp.ndimage.convolve1d(sp.ndimage.convolve1d(np.abs(subband), gaussian_kernel, axis=0, mode='wrap'), gaussian_kernel, axis=1, mode='wrap'))
            spat_activities.append(tuple(spat_activities_level))
        return spat_activities

    def _scale_spat_temp_subbands(self, filename: str, num_frames: int, frame_width: int, frame_height: int) -> None:
        '''
        Scale spatio-temporal subbands using the gain function.

        Args:
            filename: Name of video being processed. Used to infer and set temporary file paths.
            num_frames: Number of frames in the video.
            frame_width: Width of each frame.
            frame_height: Height of each frame.
        '''
        frame_stride = frame_width * frame_height * 4  # Assuming 32-bit floats.
        activity_file = open(os.path.join(self.temp_dir, 'shan12_{}.activity'.format(filename)), 'rb')

        delta = 0
        for frame_ind in range(num_frames):
            activity_file.seek(int(frame_ind * frame_stride))
            activity = np.fromfile(activity_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
            delta += 0.1 * np.mean(activity)
        delta /= num_frames

        delta_num = 0
        for frame_ind in range(num_frames):
            activity_file.seek(int(frame_ind * frame_stride))
            activity = np.fromfile(activity_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
            delta_num += np.sum(activity * np.power((activity + self._eps) / delta, self._gamma-1))

        level_deltas = []
        for gamma in self._level_gammas:
            delta_den = 0
            for frame_ind in range(num_frames):
                activity_file.seek(int(frame_ind * frame_stride))
                activity = np.fromfile(activity_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
                delta_den += np.sum(activity * np.power((activity + self._eps), gamma-1))
            level_deltas.append(delta_num / delta_den)

        for level, (delta, gamma) in enumerate(zip(level_deltas, self._level_gammas)):
            subband_files_level = []
            for subband_name in self._subband_names[1:]:
                subband_files_level.append(open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}.subband'.format(filename, level+1, subband_name)), 'rb'))

            scaled_subband_files_level = []
            for subband_name in self._subband_names[1:]:
                scaled_subband_files_level.append(open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}_scaled.subband'.format(filename, level+1, subband_name)), 'wb'))

            for frame_ind in range(num_frames):
                activity_file.seek(frame_ind * frame_stride)
                activity_map = np.fromfile(activity_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
                gain_map = np.power(activity_map + self._eps, gamma-1) / delta  # Avoid inverting and re-applying gamma on delta.
                for input_file, output_file in zip(subband_files_level, scaled_subband_files_level):
                    input_file.seek(frame_ind * frame_stride)
                    subband = np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
                    subband = subband * gain_map
                    subband.tofile(output_file)

            for file in subband_files_level:
                file.close()
            for file in scaled_subband_files_level:
                file.close()

            for subband_name in self._subband_names[1:]:
                old_filename = os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}.subband'.format(filename, level+1, subband_name))
                new_filename = os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}_scaled.subband'.format(filename, level+1, subband_name))
                os.remove(old_filename)
                os.rename(new_filename, old_filename)

        # Handle the approx_lo subband separately for convenience. Processing is identical to all other subbands.
        delta = level_deltas[-1]
        gamma = self._level_gammas[-1]
        input_file = open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_approx_lo.subband'.format(filename, self.levels)), 'rb')
        output_file = open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_approx_lo_scaled.subband'.format(filename, level+1)), 'wb')

        for frame_ind in range(num_frames):
            activity_file.seek(frame_ind * frame_stride)
            activity_map = np.fromfile(activity_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
            gain_map = np.power((activity_map + self._eps), gamma-1) / delta
            input_file.seek(frame_ind * frame_stride)
            subband = np.fromfile(input_file, np.float32, (frame_width * frame_height)).reshape(frame_height, frame_width)
            subband = subband * gain_map
            subband.tofile(output_file)

        old_filename = os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_approx_lo.subband'.format(filename, self.levels))
        new_filename = os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_approx_lo_scaled.subband'.format(filename, self.levels))
        os.remove(old_filename)
        os.rename(new_filename, old_filename)

        activity_file.close()

    def _scale_spatial_subbands(self, decomp: List[Tuple[np.ndarray, ...]], activity_map: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
        '''
        Scale spatial subbands using the gain function.

        Args:
            decomp: Wavelet decomposition.
            activity_map: Aggregated spatial acitivity_map.

        Returns:
            List[Tuple[np.ndarray, ...]]: Subband-wise scaled subbands.
        '''
        delta = 0.1 * np.mean(activity_map)
        delta_num = np.sum(activity_map * np.power((activity_map + self._eps) / delta, self._gamma-1))

        level_deltas = []
        for gamma in self._level_gammas:
            delta_den = np.sum(activity_map * np.power((activity_map + self._eps), gamma-1))
            level_deltas.append(delta_num / delta_den)

        scaled_decomp = []
        for decomp_level, delta, gamma, in zip(decomp, level_deltas, self._level_gammas):
            gain_map = np.power(activity_map + self._eps, gamma-1) / delta  # Avoid inverting and re-applying gamma on delta.
            scaled_decomp_level = []
            for subband in decomp_level:
                output_subband = subband * gain_map
                scaled_decomp_level.append(output_subband)
            scaled_decomp.append(tuple(scaled_decomp_level))

        return scaled_decomp

    def _simple_scale_spatial_subbands(self, decomp: List[Tuple[np.ndarray, ...]], activity_map: np.ndarray) -> List[Tuple[np.ndarray, ...]]:
        '''
        Scale spatial subbands using the simpler gain function used by Li et al.

        Args:
            decomp: Wavelet decomposition.
            activity_map: Aggregated spatial acitivity_map.

        Returns:
            List[Tuple[np.ndarray, ...]]: Subband-wise scaled subbands.
        '''
        delta = 0.1 * np.mean(activity_map)
        gain_map = np.power((activity_map + self._eps) / delta, self._gamma-1)

        scaled_decomp = []
        for decomp_level, mult in zip(decomp, self._level_mults):
            scaled_decomp_level = []
            for subband in decomp_level:
                output_subband = subband * mult * gain_map
                scaled_decomp_level.append(output_subband)
            scaled_decomp.append(tuple(scaled_decomp_level))

        return scaled_decomp

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tone-map a frame using Shan's 2012 TMO.

        Args:
            frame: Frame object containing a frame from the input HDR video.

        Returns:
            Frame: Tone mapped frame
        '''
        hsv = cvt_color.linear_rgb2hsv(frame.linear_rgb, frame.standard)
        lum = hsv[..., 2]
        minval, maxval = lum.min(), lum.max()
        lum = (lum - minval) / (maxval - minval)

        decomp = self._spatial_wavelet_decomp(lum, self.levels)

        activity_map = self._compute_spatial_activity_map(decomp)

        scaled_decomp = self._scale_spatial_subbands(decomp, activity_map)
        # scaled_decomp = self._simple_scale_spatial_subbands(decomp, activity_map)

        value_recon = self._spatial_wavelet_recon(scaled_decomp, 0)

        hsv_out = np.stack([hsv[..., 0], hsv[..., 1], value_recon], axis=-1)

        out_frame = Frame(self.out_standard)
        out_rgb = cvt_color.hsv2linear_rgb(hsv_out, frame.standard)
        # Scale to use the full display range.
        out_rgb = (out_rgb - out_rgb.min()) / (out_rgb.max() - out_rgb.min())
        temp_out_frame = Frame(frame.standard)
        temp_out_frame.linear_rgb = out_rgb
        out_frame = self.gamut_map(temp_out_frame)

        return out_frame

    def tonemap_video_spatiotemporal(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a video using Shan's 2012 TMO and spatio-temporal wavelet transforms.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)

        name, _ = video.file_path.rsplit('/')[1].rsplit('.', 1)
        hue_file = open(os.path.join(self.temp_dir, 'shan12_{}.hue'.format(name)), 'wb')
        sat_file = open(os.path.join(self.temp_dir, 'shan12_{}.saturation'.format(name)), 'wb')
        value_file = open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_0_approx_lo.subband'.format(name)), 'wb')

        for frame in video:
            hsv = cvt_color.linear_rgb2hsv(frame.linear_rgb, frame.standard)
            hsv[..., 0].astype('float32').tofile(hue_file)
            hsv[..., 1].astype('float32').tofile(sat_file)
            hsv[..., 2].astype('float32').tofile(value_file)

            hue_file.close()
            sat_file.close()
            value_file.close()

            self._edge_aware_spatio_temp_wavelet_decomp(name, self.levels, video.num_frames, video.width, video.height)

            self._compute_spatio_temp_activity_map(name, video.num_frames, video.width, video.height)

            self._scale_spat_temp_subbands(name, video.num_frames, video.width, video.height)
            os.remove(os.path.join(self.temp_dir, 'shan12_{}.activity'.format(name)))

            self._edge_aware_spatio_temp_wavelet_recon(name, 0, video.num_frames, video.width, video.height)

        subband_filenames = [
            os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_{}_{}.subband'.format(name, level+1, subband_name))
            for level in range(self.levels) for subband_name in self._subband_names
        ]

        for subband_filename in subband_filenames:
            os.remove(subband_filename)

        hue_file = open(os.path.join(self.temp_dir, 'shan12_{}.hue'.format(name)), 'rb')
        sat_file = open(os.path.join(self.temp_dir, 'shan12_{}.saturation'.format(name)), 'rb')
        value_file = open(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_0_approx_lo.subband'.format(name)), 'rb')
        temp_out_frame = Frame(video.standard)

        frame_stride = video.width * video.height * 4  # Assuming 32-bit floats.
        for frame_ind in range(video.num_frames):
            hue_file.seek(int(frame_ind * frame_stride))
            sat_file.seek(int(frame_ind * frame_stride))
            value_file.seek(int(frame_ind * frame_stride))

            h = np.fromfile(hue_file, np.float32, (video.width * video.height)).reshape(video.height, video.width)
            s = np.fromfile(sat_file, np.float32, (video.width * video.height)).reshape(video.height, video.width)
            v = np.fromfile(value_file, np.float32, (video.width * video.height)).reshape(video.height, video.width)
            hsv = np.stack([h, s, v], axis=-1)

            out_rgb = cvt_color.hsv2linear_rgb(hsv, self.out_standard)
            # Scale to use the full display range.
            temp_out_frame.linear_rgb = (out_rgb - out_rgb.min()) / (out_rgb.max() - out_rgb.min())
            out_video.write_frame(self.gamut_map(temp_out_frame))

        os.remove(os.path.join(self.temp_dir, 'shan12_{}.hue'.format(name)))
        os.remove(os.path.join(self.temp_dir, 'shan12_{}.saturation'.format(name)))
        os.remove(os.path.join(self.temp_dir, 'shan12_{}_wavelet_level_0_approx_lo.subband'.format(name)))
        out_video.close()

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Shan's 2012 TMO.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        # First pass
        minval = np.inf
        maxval = -np.inf
        for frame in video:
            hsv = cvt_color.linear_rgb2hsv(frame.linear_rgb, frame.standard)
            lum = hsv[..., 2]
            minval = min(minval, lum.min())
            maxval = max(maxval, lum.max())
        video.reset()

        # Second pass
        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        for frame in video:
            hsv = cvt_color.linear_rgb2hsv(frame.linear_rgb, frame.standard)
            lum = hsv[..., 2]
            lum = (lum - minval) / (maxval - minval)

            decomp = self._spatial_wavelet_decomp(lum, self.levels)

            activity_map = self._compute_spatial_activity_map(decomp)

            scaled_decomp = self._scale_spatial_subbands(decomp, activity_map)
            # scaled_decomp = self._simple_scale_spatial_subbands(decomp, activity_map)

            value_recon = self._spatial_wavelet_recon(scaled_decomp, 0)

            hsv_out = np.stack([hsv[..., 0], hsv[..., 1], value_recon], axis=-1)

            out_frame = Frame(self.out_standard)
            out_rgb = cvt_color.hsv2linear_rgb(hsv_out, frame.standard)
            # Scale to use the full display range.
            out_rgb = (out_rgb - out_rgb.min()) / (out_rgb.max() - out_rgb.min())
            temp_out_frame = Frame(frame.standard)
            temp_out_frame.linear_rgb = out_rgb
            out_frame = self.gamut_map(temp_out_frame)
            out_video.append(out_frame)
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max'):
        '''
        Tone-map a video smoothly using Shan's 2012 TMO and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        Returns:
            Frame: Tone-mapped frame.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath)
