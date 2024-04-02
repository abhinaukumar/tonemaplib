from typing import List, Tuple, Union, Optional
import warnings
import os

import numpy as np
import scipy as sp

from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.buffer import CircularBuffer
from videolib.cvt_color import rgb2yuv
from .tmo import TMO


class Eilertsen15TMO(TMO):
    '''
    Implementation of Eilertsen's TMO from 2015.

    Args:
        out_standard (Standard): Standard to which the output frames must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        bin_width (float): Width of histogram bins.
        noise_a (float): Slope parameter of the Poisson-Gaussian noise model.
        noise_b (float): Intercept parameter of the Poisson-Gaussian noise model.
        desat (float): Desaturation parameter for color correction.

    Refs:
        Eilertsen, G., Mantiuk, R., & Unger, J. (November 2015)
        "Real-time noise-aware tone mapping."
        ACM Trans. Graph. 34, 6, Article 198
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        bin_width: Optional[float] = 0.05,
        noise_a: Optional[float] = 5e-8,
        noise_b: Optional[float] = 1e-9,
        desat: Optional[float] = 0.0
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            noise_a: Slope parameter of the Poisson-Gaussian noise model (Default: 5e-8).
            noise_b: Intercept parameter of the Poisson-Gaussian noise model (Default: 1e-9).
            desat: Desaturation parameter for color correction (Default: 0.0).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self._eps: float = 1e-20
        if noise_a < 0:
            warnings.warn('Clipping noise_a to 0.', RuntimeWarning)
            noise_a = 0
            if noise_b <= 0:
                warnings.warn('Clipping noise_b to {} since noise_a is 0.'.format(self._eps), RuntimeWarning)
        elif noise_b < 0:
            warnings.warn('Clipping noise_b to 0', RuntimeWarning)
            noise_b = 0

        # Noise function parameters
        self.noise_a: float = noise_a
        self.noise_b: float = noise_b

        # Desaturation parameter
        self.desat: float = desat

        self._d2h: float = 2.5  # Ratio of distance from screen to height of screen.
        self._patch_size: int = int(2160 * self._d2h * np.tan(np.pi/72))  # Assuming 1080p display.

        # Histogram parameters
        self.bin_width: float = bin_width
        self._bin_min: float = -8
        self._bin_max: float = 0
        self._bins: np.ndarray = np.arange(self._bin_min, self._bin_max + self.bin_width/2, self.bin_width)
        self._num_bins: int = len(self._bins)

        # IIR filter parameters
        self._iir_filter_a_coeffs: List[float] = [-2.895292177877897, 2.795994584283360, -0.900566088981622]
        self._iir_filter_b_coeffs: List[float] = [0.0000170396779801130, 0.0000511190339403389, 0.0000511190339403389, 0.0000170396779801130]
        self._iir_filter_taps: int = 4

        # Buffers to hold previous filtered and unfiltered frames.
        self._x_tone_buf: CircularBuffer = CircularBuffer(self._iir_filter_taps)
        self._y_tone_buf: CircularBuffer = CircularBuffer(self._iir_filter_taps-1)

        # Display parameters
        self._disp_gamma: float = 2.2
        self._disp_peak_lum: float = 100
        self._disp_black_lum: float = 0.8
        self._disp_ambient_lum: float = 400 / np.pi  # Converting 400 lux to nits.
        self._disp_reflectance: float = 0.01

        # Contrast threshold function parameters
        self._ct_p1: float = 30.182
        self._ct_p2: float = 4.3806 * 1e-4
        self._ct_p3: float = 1.5154
        self._ct_p4: float = 0.29412

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'bin_width', 'noise_a', 'noise_b', 'desat']

    def _noise_magnitude(self, intensity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute the noise magnitude in the log domain

        Args:
            intensity: Intensity of incident light.

        Returns:
            Union[float, np.ndarray]: Noise magnitude
        '''
        sigma_n = np.sqrt(self.noise_a * intensity + self.noise_b)
        return np.log10(1 + sigma_n / intensity)

    def _contrast_threshold(self, lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Contrast threshold versus luminance function (1 / CSF) from HDR-VDP2.

        Args:
            lum: Input luminance

        Returns:
            Union[float, np.ndarray]: Contrast threshold.
        '''
        return (1 / self._ct_p1) * np.power(1 + np.power(self._ct_p2/lum, self._ct_p3), self._ct_p4)

    def _visibility_threshold(self, lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute the visibility threshold corresponding to the given display luminance.

        Args:
            lum: Display luminance.

        Returns:
            Union[float, np.ndarray]: Visibility thresholds
        '''
        contrasts = self._contrast_threshold(lum)
        return 0.5 * np.log10((1 + contrasts) / (1 - contrasts))

    def _display_function(self, pix: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Apply the display function.

        Args:
            pix: Pixel values in the range [0, 1]

        Returns:
            Union[float, np.ndarray]: Display luminance.
        '''
        return pix**self._disp_gamma * (self._disp_peak_lum - self._disp_black_lum) + self._disp_black_lum + self._disp_reflectance*self._disp_ambient_lum

    def _inverse_display_function(self, lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Apply the inverse display function.

        Args:
            lum: Display luminance.

        Returns:
            Union[float, np.ndarray]: Pixel values.
        '''
        return np.power(lum - self._disp_black_lum - self._disp_reflectance * self._disp_ambient_lum / (self._disp_peak_lum - self._disp_black_lum), 1/self._disp_gamma)

    def _get_histogram(self, lum: np.ndarray, contrast: np.ndarray, noise: np.ndarray) -> np.ndarray:
        '''
        Compute noise-aware histogram of luminances, using contrast and noise information.

        Args:
            lum: Luminance image.
            contrast: Map for local contrasts.
            noise: Map of noise levels.

        Returns:
            np.ndarray: Histogram values.
        '''
        hist = np.zeros((self._num_bins,))

        contrast_mask = (contrast > noise)
        denom = np.sum(contrast_mask.astype('float32'))
        if denom != 0:
            for i, bin in enumerate(self._bins):
                bin_mask = (lum >= bin-self.bin_width/2) & (lum < bin+self.bin_width/2)
                hist[i] = np.sum((bin_mask & contrast_mask).astype('float32')) / denom

        return hist

    def _get_suboptimal_slopes(self, hist: np.ndarray, range_disp: float) -> np.ndarray:
        '''
        Find slopes of the tone curve. Slopes are suboptimal since they may be negative.

        Args:
            hist: Histogram
            range_disp: Dynamic range of LDR display.

        Returns:
            np.ndarray: Suboptimal slopes.
        '''
        hist_mask = (hist != 0)
        slopes = np.zeros(self._num_bins)
        inv_sum = np.sum(1 / hist[hist_mask])
        slopes[hist_mask] = 1 + (range_disp/self.bin_width - self._num_bins) / (hist[hist_mask] * inv_sum)
        return slopes

    def _get_optimal_tone_curve(self, hist: np.ndarray) -> np.ndarray:
        '''
        Find optimal piecewise linear tone curve.

        Args:
            hist: Normalized histogram values.

        Returns:
            np.ndarray: Array of values corresponding to the value at the center of each bin.
        '''
        range_disp: float = np.log10(self._display_function(1) / self._display_function(0))
        num_iter: int = 20
        thresh: float = 1e-4

        hist_mask = np.ones((self._num_bins, ), dtype=bool)
        for i in range(num_iter):
            hist_mask = (hist > thresh)
            high_bins = np.sum(hist_mask)
            if high_bins == 0:
                break
            thresh = max((high_bins - range_disp/self.bin_width) / np.sum(1 / hist[hist_mask]), 0)

        hist_mask = (hist > thresh)
        s = np.where(hist_mask, self._get_suboptimal_slopes(hist, range_disp), 0)
        if np.max(s) != 0:
            tone_curve = np.zeros((self._num_bins + 2,))
            tone_curve[1:-1] = -range_disp + np.cumsum(s) * self.bin_width
            tone_curve[0] = -range_disp
            tone_curve[-1] = tone_curve[-2]
            return tone_curve
        else:
            return -range_disp * np.ones((self._num_bins + 2))  # Set everything to lowest level if empty histogram

    def _get_local_tone_curves(self, lum: np.ndarray) -> np.ndarray:
        '''
        Compute local piecewise-linear tone curves.

        Args:
            lum: Luminance image.

        Returns:
            np.ndarray: Array of values of the tone curve at each patch.
        '''
        rows, cols = lum.shape

        lum = np.clip(lum, self._bin_min, self._bin_max)
        contrast = np.sqrt(np.clip(sp.ndimage.gaussian_filter(lum**2, 3) - sp.ndimage.gaussian_filter(lum, 3)**2, 0, None))
        noise = self._noise_magnitude(np.power(10, lum))

        hist_global = self._get_histogram(lum, contrast, noise)
        patch_rows = np.ceil(rows/self._patch_size).astype('int')
        patch_cols = np.ceil(cols/self._patch_size).astype('int')
        local_tone_curves = np.zeros((patch_rows + 2, patch_cols + 2, self._num_bins + 2))

        for i in range(1, patch_rows+1):
            for j in range(1, patch_cols+1):
                hist_patch = self._get_histogram(lum[(i-1)*self._patch_size: i*self._patch_size, (j-1)*self._patch_size: j*self._patch_size],
                                                 contrast[(i-1)*self._patch_size: i*self._patch_size, (j-1)*self._patch_size: j*self._patch_size],
                                                 noise[(i-1)*self._patch_size: i*self._patch_size, (j-1)*self._patch_size: j*self._patch_size]
                                                 )
                hist_blend = 0.9*hist_patch + 0.1*hist_global
                local_tone_curves[i, j, :] = self._get_optimal_tone_curve(hist_blend)

        local_tone_curves[0, :, :] = local_tone_curves[1, :, :]
        local_tone_curves[:, 0, :] = local_tone_curves[:, 1, :]
        local_tone_curves[-1, :, :] = local_tone_curves[-2, :, :]
        local_tone_curves[:, -1, :] = local_tone_curves[:, -2, :]

        return local_tone_curves

    def _apply_local_tone_curves(self, lum: np.ndarray, local_tone_curves: np.ndarray) -> np.ndarray:
        '''
        Apply the appropriate local tone curve to a luminance array

        Args:
            lum: Luminance array
            local_tone_curves: Coefficients defining piecewise tone curves for each patch.
        '''
        rows, cols = lum.shape
        row_inds, col_inds = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        patch_row_inds, half_row_inds = np.divmod(row_inds, self._patch_size)
        patch_col_inds, half_col_inds = np.divmod(col_inds, self._patch_size)
        patch_row_inds = patch_row_inds.astype('int') + 1
        patch_col_inds = patch_col_inds.astype('int') + 1
        neighbor_row_inds = patch_row_inds + np.where(half_row_inds <= self._patch_size//2, -1, 1)
        neighbor_col_inds = patch_col_inds + np.where(half_col_inds <= self._patch_size//2, -1, 1)
        lum_inds, half_lum_inds = np.divmod((lum - self._bin_min), self.bin_width)
        lum_inds = lum_inds.astype('int')
        neighbor_lum_inds = lum_inds + np.where(half_lum_inds <= self.bin_width/2, -1, 1)

        row_weights = 1 - np.abs(half_row_inds - self._patch_size//2) / self._patch_size
        col_weights = 1 - np.abs(half_col_inds - self._patch_size//2) / self._patch_size
        lum_weights = 1 - np.abs(half_lum_inds - self.bin_width/2) / self.bin_width

        # Collect 8 values and set the corresponding interpolation weights for 3D linear interpolation
        vals = []
        weights = []
        vals.append(local_tone_curves[patch_row_inds, patch_col_inds, lum_inds])
        weights.append(row_weights * col_weights * lum_weights)
        vals.append(local_tone_curves[patch_row_inds, neighbor_col_inds, lum_inds])
        weights.append(row_weights * (1 - col_weights) * lum_weights)
        vals.append(local_tone_curves[neighbor_row_inds, patch_col_inds, lum_inds])
        weights.append((1 - row_weights) * col_weights * lum_weights)
        vals.append(local_tone_curves[neighbor_row_inds, neighbor_col_inds, lum_inds])
        weights.append((1 - row_weights) * (1 - col_weights) * lum_weights)
        vals.append(local_tone_curves[patch_row_inds, patch_col_inds, neighbor_lum_inds])
        weights.append(row_weights * col_weights * (1 - lum_weights))
        vals.append(local_tone_curves[patch_row_inds, neighbor_col_inds, neighbor_lum_inds])
        weights.append(row_weights * (1 - col_weights) * (1 - lum_weights))
        vals.append(local_tone_curves[neighbor_row_inds, patch_col_inds, neighbor_lum_inds])
        weights.append((1 - row_weights) * col_weights * (1 - lum_weights))
        vals.append(local_tone_curves[neighbor_row_inds, neighbor_col_inds, neighbor_lum_inds])
        weights.append((1 - row_weights) * (1 - col_weights) * (1 - lum_weights))

        tone_vals = np.zeros((rows, cols))
        for weight, val in zip(weights, vals):
            tone_vals += weight * val

        return tone_vals

    @staticmethod
    def _edge_stop_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute edge stop function w_r(x)

        Args:
            x: Input pixel values.

        Returns:
            Union[float, np.ndarray]: Values of the edge stop function.
        '''
        edge_stop = 0.5
        return np.where(x <= edge_stop, (1 - (x / edge_stop)**2)**2, 0)

    @staticmethod
    def _gradient_norm(img: np.ndarray, half_size: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute the norm of gradients using a linear ramp.

        Args:
            img: Image for which gradient norm is to be computed.
            half_size: Half-size of the filter kernel.
        '''
        filt = np.arange(-half_size, half_size+1)
        grad1 = sp.ndimage.convolve1d(img, filt, axis=0)
        grad2 = sp.ndimage.convolve1d(img, filt, axis=1)
        return np.sqrt(grad1**2 + grad2**2)

    @staticmethod
    def _base_detail_decomposition(lum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Use fast detail extraction to decompose into base and detail layers.

        Args:
            lum: Luminance array to be decomposed

        Returns:
            Tuple[np.ndarray, np.ndarray]: Base and detail layers.
        '''
        num_iter = 12
        sigma = 0.1
        base = lum.copy()
        for i in range(num_iter):
            sigma_iter = np.sqrt(2*i + 1) * sigma
            base_filt = sp.ndimage.gaussian_filter(base, sigma_iter)
            grad_norm = np.maximum(Eilertsen15TMO._gradient_norm(base, np.ceil(3*sigma_iter)), (i+1)*np.abs(base_filt - lum))
            edge_stop_weights = Eilertsen15TMO._edge_stop_function(grad_norm)
            base = base * (1 - edge_stop_weights) + base_filt * edge_stop_weights

        detail = lum - base
        return base, detail

    def _filter_tone_curves(self, tone_curves: np.ndarray) -> np.ndarray:
        '''
        Apply IIR filter to the tone curves.

        Args:
            tone_curves: Current value of the tone curves.

        Returns:
            np.ndarray: Filtered tone curves
        '''
        if self._y_tone_buf.isempty():
            self._y_tone_buf.fill(tone_curves)
        if self._x_tone_buf.isempty():
            self._x_tone_buf.fill(tone_curves)

        self._x_tone_buf.append(tone_curves)
        filtered_tone_curves = np.zeros_like(tone_curves)
        for weight_x, prev_x in zip(self._iir_filter_b_coeffs, self._x_tone_buf):
            filtered_tone_curves = filtered_tone_curves + weight_x * prev_x
        for weight_y, prev_y in zip(self._iir_filter_a_coeffs, self._y_tone_buf):
            filtered_tone_curves = filtered_tone_curves - weight_y * prev_y

        self._y_tone_buf.append(filtered_tone_curves)
        return filtered_tone_curves

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tone-map a frame using Eilertsen's 2015 TMO.

        Args:
            frame : Frame object containing a frame from the input HDR video.

        Returns:
            Frame: Tone mapped frame
        '''
        linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
        lum = linear_yuv_const[..., 0]
        log_lum = np.log10(lum)

        base, detail = Eilertsen15TMO._base_detail_decomposition(log_lum)

        local_tone_curves = self._get_local_tone_curves(base)

        base_tonemapped = self._apply_local_tone_curves(base, local_tone_curves)
        detail_scale_factor = np.clip(self._visibility_threshold(np.power(10, base_tonemapped)) / self._noise_magnitude(np.power(10, base)), 0, 1)
        detail_mod = detail_scale_factor * detail

        log_lum_out = base_tonemapped + detail_mod
        lum_out = Frame._lift_to_multichannel(np.power(10, log_lum_out))
        lum = Frame._lift_to_multichannel(lum)

        temp_out_frame = Frame(frame.standard)
        temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
        out_frame = self.gamut_map(temp_out_frame)

        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Eilertsen's 2015 TMO.

        Args:
            video: Video object containing the input HDR video.
            out_filepath: Path to which output file must be written.
        '''
        if not os.path.isdir(os.path.dirname(out_filepath)):
            raise ValueError('{} is not a valid file path'.format(out_filepath))

        out_video = Video(out_filepath, self.out_standard, mode='w', format=self.out_format)

        mean_local_tone_curves = None

        # First pass
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            lum = linear_yuv_const[..., 0]
            log_lum = np.log10(lum)

            base, detail = Eilertsen15TMO._base_detail_decomposition(log_lum)

            local_tone_curves = self._get_local_tone_curves(base)
            if mean_local_tone_curves is None:
                mean_local_tone_curves = local_tone_curves
            else:
                mean_local_tone_curves += local_tone_curves
        video.reset()

        mean_local_tone_curves = mean_local_tone_curves / video.num_frames 

        # Second pass
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            lum = linear_yuv_const[..., 0]
            log_lum = np.log10(lum)

            base, detail = Eilertsen15TMO._base_detail_decomposition(log_lum)
            base_tonemapped = self._apply_local_tone_curves(base, mean_local_tone_curves)
            detail_scale_factor = np.clip(self._visibility_threshold(np.power(10, base_tonemapped)) / self._noise_magnitude(np.power(10, base)), 0, 1)
            detail_mod = detail_scale_factor * detail

            log_lum_out = base_tonemapped + detail_mod
            lum_out = Frame._lift_to_multichannel(np.power(10, log_lum_out))
            lum = Frame._lift_to_multichannel(lum)

            temp_out_frame = Frame(frame.standard)
            temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
            out_frame = self.gamut_map(temp_out_frame)
            out_video.write_frame(out_frame)

        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a video smoothly using Eilertsen's 2015 TMO and filtering.

        Args:
            video: Video object containing the input HDR video.
            out_filepath: Path to which output file must be written.
        '''
        if not os.path.isdir(os.path.dirname(out_filepath)):
            raise ValueError('{} is not a valid file path'.format(out_filepath))

        out_video = Video(out_filepath, self.out_standard, mode='w', format=self.out_format)

        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            lum = linear_yuv_const[..., 0]
            log_lum = np.log10(lum)

            base, detail = Eilertsen15TMO._base_detail_decomposition(log_lum)

            local_tone_curves = self._get_local_tone_curves(base)
            filtered_local_tone_curves = self._filter_tone_curves(local_tone_curves)

            base_tonemapped = self._apply_local_tone_curves(base, filtered_local_tone_curves)
            detail_scale_factor = np.clip(self._visibility_threshold(np.power(10, base_tonemapped)) / self._noise_magnitude(np.power(10, base)), 0, 1)
            detail_mod = detail_scale_factor * detail

            log_lum_out = base_tonemapped + detail_mod
            lum_out = Frame._lift_to_multichannel(np.power(10, log_lum_out))
            lum = Frame._lift_to_multichannel(lum)

            temp_out_frame = Frame(frame.standard)
            temp_out_frame.linear_rgb = ((frame.linear_rgb / (lum + 1e-10) - 1) * (1 - self.desat) + 1) * lum_out
            out_frame = self.gamut_map(temp_out_frame)
            out_video.write_frame(out_frame)

        out_video.close()
