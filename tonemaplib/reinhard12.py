from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import scipy as sp

from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.utils import apply_transfer_mat
from .tmo import TMO


class Reinhard12TMO(TMO):
    '''
    Implementation of Reinhard's 2012 TMO.

    Args:
        out_standard (Standard): Standard to which the output frames must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        viewing_cond (str): Viewing condition to assume. Must be one of 'blue', 'neutral', 'red' or 'green' (Default: 'neutral').
    Refs:
        Reinhard, E., Pouli, T., Kunkel, T., Long, B., Ballestad, A. & Damberg, G. (November 2012)
        "Calibrated image appearance reproduction."
        ACM Trans. Graph. 31, 6, Article 201
    '''
    params = ['video_mode', 'viewing_cond']
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        viewing_cond: Optional[str] = 'neutral'
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            viewing_cond: Viewing condition to assume. Must be one of 'blue', 'neutral', 'red' or 'green' (Default: 'neutral').
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        viewing_conds = ['blue', 'neutral', 'red', 'green']
        if viewing_cond not in viewing_conds:
            raise ValueError('Invalid choice of viewing condition. Must be one of {}'.format(viewing_conds))
        else:
            self.viewing_cond: str = viewing_cond

        disp_lum = 100
        disp_lums = dict(zip(viewing_conds, [disp_lum]*len(viewing_conds)))
        disp_max_lum = 191
        disp_max_lums = dict(zip(viewing_conds, [disp_max_lum]*len(viewing_conds)))
        disp_white_point = np.array([92, 100, 108])
        disp_white_points = dict(zip(viewing_conds, [disp_white_point]*len(viewing_conds)))
        disp_max_white_point = np.array([172, 191, 190])
        disp_max_white_points = dict(zip(viewing_conds, [disp_max_white_point]*len(viewing_conds)))

        view_lum = 800
        view_lums = dict(zip(viewing_conds, [view_lum]*len(viewing_conds)))
        view_max_lum = 7010
        view_max_lums = dict(zip(viewing_conds, [view_max_lum]*len(viewing_conds)))
        view_white_points = dict(zip(viewing_conds, [np.array([0.21, 0.25, 1-0.21-0.25])*view_lum/0.25, np.array([0.3118, 0.3236, 1-0.3118-0.3236])*view_lum/0.3236, np.array([0.4, 0.38, 1-0.4-0.38])*view_lum/0.38, np.array([0.29, 0.33, 1-0.29-0.32])*800/0.33]))
        view_max_white_points = {view_cond: view_white_points[view_cond]*view_max_lum/view_white_points[view_cond][1] for view_cond in view_white_points}
        d2h = 3.2
        d2hs = dict(zip(viewing_conds, [d2h]*len(viewing_conds)))

        # Display parameters
        self._disp_luminance = disp_lums[self.viewing_cond]
        self._disp_max_luminance = disp_max_lums[self.viewing_cond]
        self._disp_white_point = disp_white_points[self.viewing_cond]
        self._disp_max_white_point = disp_max_white_points[self.viewing_cond]
        self._disp_adaptation = 1

        # Viewing parameters
        self._view_luminance = view_lums[self.viewing_cond]
        self._view_max_luminance = view_max_lums[self.viewing_cond]
        self._view_white_point = view_white_points[self.viewing_cond]
        self._view_max_white_point = view_max_white_points[self.viewing_cond]
        self._view_adaptation = 1
        self._d2h = d2hs[self.viewing_cond]
        self._diag_size = np.sqrt((1920**2 + 1080**2))

        self._relative_visual_angle = 3/np.pi * np.arctan2((1/self._d2h)*np.sqrt(9**2 + 16**2), 18)

        # Combine view and display parameters
        self._view_disp_luminance = self._relative_visual_angle*self._disp_luminance + (1 - self._relative_visual_angle)*self._view_luminance
        self._view_disp_max_luminance = self._relative_visual_angle*self._disp_max_luminance + (1 - self._relative_visual_angle)*self._view_max_luminance
        self._view_disp_white_point = self._relative_visual_angle*self._disp_white_point + (1 - self._relative_visual_angle)*self._view_white_point
        self._view_disp_max_white_point = self._relative_visual_angle*self._disp_max_white_point + (1 - self._relative_visual_angle)*self._view_max_white_point
        self._view_disp_adaptation = self._relative_visual_angle*self._disp_adaptation + (1 - self._relative_visual_angle)*self._view_adaptation

        # Convert parameters to LMS
        self._disp_white_point_lms = Reinhard12TMO._xyz2lms(self._disp_white_point)
        self._disp_max_white_point_lms = Reinhard12TMO._xyz2lms(self._disp_max_white_point)
        self._view_disp_white_point_lms = Reinhard12TMO._xyz2lms(self._view_disp_white_point)
        self._view_disp_max_white_point_lms = Reinhard12TMO._xyz2lms(self._view_disp_max_white_point)

        # Median cut parameters
        self._min_split_size = 10
        self._median_cut_levels = 7

        # Leaky integration parameters
        self._interp_factor = 0.08
        self._max_mapped_inv = None
        self._saturation_terms = None

    def _leaky_integrate_params(self, max_mapped_inv: np.ndarray, saturation_terms: np.ndarray) -> None:
        '''
        Leaky integrate parameters.

        Args:
            max_mapped_inv: Inverse of max of mapped values.
            saturation_terms: Saturation terms in the output mapping function.
        '''
        self._max_mapped_inv = \
            self._interp_factor*max_mapped_inv + (1 - self._interp_factor)*self._max_mapped_inv if self._max_mapped_inv is not None \
            else max_mapped_inv
        self._saturation_terms = \
            self._interp_factor*saturation_terms + (1 - self._interp_factor)*self._saturation_terms if self._saturation_terms is not None \
            else saturation_terms

    @staticmethod
    def _xyz2lms(xyz: np.ndarray) -> np.ndarray:
        '''
        Convert XYZ to Hunter-Point-Estevez LMS.

        Args:
            xyz: Input in XYZ space.

        Returns:
            np.ndarray: Output in LMS space.
        '''
        hpe_transfer_mat = np.array([[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.00000, 0.00000, 1.00000]])
        if xyz.ndim == 1:
            if len(xyz) != 3:
                raise ValueError
            return hpe_transfer_mat @ xyz
        else:
            Frame._assert_or_make_3channel(xyz)
            return apply_transfer_mat(xyz, hpe_transfer_mat)

    @staticmethod
    def _lms2xyz(lms: np.ndarray) -> np.ndarray:
        '''
        Convert Hunter-Point-Estevez LMS to XYZ.

        Args:
            lms: Input in LMS space.

        Returns:
            np.ndarray: Output in XYZ space.
        '''
        hpe_transfer_mat = np.array([[0.38971, 0.68898, -0.07868], [-0.22981, 1.18340, 0.04641], [0.00000, 0.00000, 1.00000]])
        if lms.ndim == 1:
            if len(lms) != 3:
                raise ValueError
            return np.linalg.inv(hpe_transfer_mat) @ lms
        else:
            Frame._assert_or_make_3channel(lms)
            return apply_transfer_mat(lms, np.linalg.inv(hpe_transfer_mat))

    @staticmethod
    def _estimate_scene_white_point(xyz: np.ndarray) -> np.ndarray:
        '''
        Estimate the white point of the scene.

        Args:
            xyz: Scene image in XYZ space.

        Returns:
            np.ndarray: Estimated white point in XYZ space.
        '''
        geomeans = np.array([sp.stats.mstats.gmean(xyz[..., dim].flatten()) for dim in range(3)])
        achrom_white = geomeans[1] * np.ones((3,))
        return (geomeans + achrom_white)/2

    @staticmethod
    def _get_adaptation(lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute degree of adaptation (CIECAM02).

        Args:
            lum: Adaptation luminances.

        Returns:
            Union[float, np.ndarray]: Degrees of adaptation.
        '''
        return 1 - np.exp(-(lum + 42)/92) / 3.6

    @staticmethod
    def _pupil_size(lum: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute pupil area.

        Args:
            lum: Maximum adapting luminance.

        Returns:
            Union[float, np.ndarray]: Pupil area.
        '''
        return np.pi * (2.45 - 1.5 * np.tanh(0.4*np.log(lum + 1)))

    @staticmethod
    def _bleaching_function(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Compute the probability that a photoreceptor is able to function.

        Args:
            x: Photoreceptor channel input (LMS).

        Returns:
            Union[float, np.ndarray]: Probability that photoreceptor is able to function.
        '''
        return 4.3 / (4.3 + np.log(x))

    @staticmethod
    def _integral_image(img: np.ndarray) -> np.ndarray:
        '''
        Compute integral image aka summed area table.

        Args:
            img: Input image.

        Returns:
            np.ndarray: Integral image.
        '''
        img = Frame._assert_or_make_1channel(img)
        rows, cols = img.shape
        int_img = np.zeros((rows+1, cols+1))
        int_img[1:, 1:] = np.cumsum(np.cumsum(img, 0), 1)
        return int_img

    @staticmethod
    def _get_sum_from_integral_image(int_img: np.ndarray, inds: Tuple[int, int], size: Tuple[int, int]) -> float:
        '''
        Obtain sum over a window from an integral image.

        Args:
            int_img: Integral image.
            inds: Indices of the top-left corner of the window (0-based)
            size: Size along each dimenson of the window.

        Returns:
            float: Sum over the given window.
        '''
        i, j = inds
        rows, cols = size
        return int_img[i, j] + int_img[i+rows, j+cols] - int_img[i+rows, j] - int_img[i, j+cols]

    def _median_cut_split(self, int_img: np.ndarray, inds: Tuple[int, int], size: Tuple[int, int], dim: int) -> int:
        '''
        Use binary search to find the index along a dimension that splits the luminance into half.

        Args:
            int_img: Integral image of luminances.
            inds: Indices defining the top left of the sub-image.
            size: Size along each dimension of the sub-image.
            dim: Dimension along which to split the sub-image.

        Returns:
            Index (with respect to the sub-image) that splits total luminance in roughly half.
        '''
        total_lum = Reinhard12TMO._get_sum_from_integral_image(int_img, inds, size)
        rows, cols = size
        lo = self._min_split_size
        hi = (rows if dim == 0 else cols) - self._min_split_size
        mid = -1
        while lo <= hi:
            mid = (lo + hi)//2
            split_size = (mid, cols) if dim == 0 else (rows, mid)
            sum = Reinhard12TMO._get_sum_from_integral_image(int_img, inds, split_size)
            if sum == total_lum/2:
                break
            elif sum < total_lum/2:
                lo = mid + 1
            else:
                hi = mid - 1
        return mid

    def _median_cut_help(self, xyz: np.ndarray, int_lum: np.ndarray, cur_level: int, inds: Tuple[int, int]) -> List[Dict[float, Union[Tuple, np.ndarray]]]:
        '''
        Recurse in the median cut algorithm.

        Args:
            lum: Input luminance sub-image.
            int_lum: Integral image of the Y channel of the entire source image.
            cur_level: Current recursion level.
            inds: Indices defining the top left of the sub-image.

        Returns:
            List[Dict[float, Union[Tuple, np.ndarray]]]: Dictionary containing the centers and white points of the cuts.
        '''
        rows, cols, _ = xyz.shape
        end_recursion = (cur_level == self._median_cut_levels)  # Maximum recursion depth has been reached.

        if not end_recursion:
            dim = int(rows < cols)
            split_ind = self._median_cut_split(int_lum, inds, (rows, cols), dim)
            end_recursion = (split_ind == -1)  # Minimum cut size has been reached.

        if end_recursion:
            return [{'center': (inds[0]+rows//2, inds[1]+cols//2),
                     'white_point': Reinhard12TMO._estimate_scene_white_point(xyz)}]

        if dim == 0:
            return self._median_cut_help(xyz[:split_ind+1, :, :], int_lum, cur_level+1, inds) + \
                   self._median_cut_help(xyz[split_ind+1:, :, :], int_lum, cur_level+1, (inds[0]+split_ind+1, inds[1]))
        else:
            return self._median_cut_help(xyz[:, :split_ind+1, :], int_lum, cur_level+1, inds) + \
                   self._median_cut_help(xyz[:, split_ind+1:, :], int_lum, cur_level+1, (inds[0], inds[1]+split_ind+1))

    def _median_cut(self, xyz: np.ndarray) -> List[Dict[float, Union[Tuple, np.ndarray]]]:
        '''
        Perform the median cut algorithm.

        Args:
            lum: Input luminance image.

        Returns:
            Dict[float, Union[Tuple, np.ndarray]]: Dictionary containing the centers and white points of the cuts.
        '''
        Frame._assert_or_make_3channel(xyz)
        lum = xyz[..., 1]
        int_lum = Reinhard12TMO._integral_image(lum)
        return self._median_cut_help(xyz, int_lum, 0, (0, 0))

    @staticmethod
    def _max_neural_response(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Calculate maximum neural responses.

        Args:
            x: Effective maximum photoreceptor LMS responses.

        Returns:
            Union[float, np.ndarray]: Maximum neural reponse.
        '''
        return 34 / np.sqrt(1 + x/67)

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tone-map a frame using Reinhard's 2012 TMO.

        Args:
            frame: Frame to be tone-mapped.

        Returns:
            Frame: Tone-mapped frame.
        '''
        xyz = np.clip(frame.xyz * frame.standard.linear_range, 1e-4, None)
        cuts = self._median_cut(xyz)
        lum = xyz[..., 1]
        lms = self._xyz2lms(xyz)
        global_scene_max_lum = np.percentile(lum, 90)
        pupil_size = Reinhard12TMO._pupil_size(global_scene_max_lum)

        global_scene_white_point = Reinhard12TMO._estimate_scene_white_point(xyz)
        global_scene_lum = global_scene_white_point[1]
        global_scene_max_white_point = global_scene_white_point * global_scene_max_lum / global_scene_lum
        global_scene_max_white_point_lms = self._xyz2lms(global_scene_max_white_point)
        global_scene_scale_factor = pupil_size * Reinhard12TMO._bleaching_function(global_scene_white_point)

        scene_max_response = Reinhard12TMO._max_neural_response(pupil_size * Reinhard12TMO._bleaching_function(global_scene_max_white_point_lms) * global_scene_max_white_point_lms)
        view_max_response = Reinhard12TMO._max_neural_response(pupil_size * Reinhard12TMO._bleaching_function(self._view_disp_max_white_point_lms) * self._view_disp_max_white_point_lms)
        view_scale_factor = pupil_size * Reinhard12TMO._bleaching_function(self._view_disp_white_point_lms)
        view_semi_saturation = self._view_disp_adaptation * view_scale_factor * self._view_disp_white_point + (1 - self._view_disp_adaptation) * pupil_size * self._view_disp_luminance

        for cut in cuts:
            cut['lum_adapt'] = cut['white_point'][1]
            cut['degree_adapt'] = self._get_adaptation(cut['lum_adapt'])
            cut['effective_white_point'] = pupil_size * Reinhard12TMO._bleaching_function(cut['white_point']) * cut['white_point']
            cut['effective_white_point_lms'] = self._xyz2lms(cut['effective_white_point'])

        rows, cols = lum.shape
        row_inds, col_inds = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

        scene_lums = np.zeros((rows, cols))
        scene_adaptations = np.zeros((rows, cols))
        scene_eff_white_points_lms = np.zeros((rows, cols, 3))
        norm_factors = np.zeros((rows, cols))

        for cut in cuts:
            dists = np.sqrt((row_inds - cut['center'][0])**2 + (col_inds - cut['center'][1])**2)
            weights = (1 - np.cos(np.pi/2 - self._relative_visual_angle * dists/self._diag_size))**4
            scene_lums = scene_lums + weights*cut['lum_adapt']
            scene_adaptations = scene_adaptations + weights*cut['degree_adapt']
            scene_eff_white_points_lms = scene_eff_white_points_lms + Frame._lift_to_multichannel(weights)*np.tile(np.expand_dims(cut['effective_white_point_lms'], [0, 1]), [rows, cols, 1])
            norm_factors += weights

        scene_lums = scene_lums / norm_factors
        scene_adaptations = scene_adaptations / norm_factors
        scene_eff_white_points_lms = scene_eff_white_points_lms / Frame._lift_to_multichannel(norm_factors)

        scene_semi_saturations = \
            Frame._lift_to_multichannel(scene_adaptations) * scene_eff_white_points_lms + \
            (1 - Frame._lift_to_multichannel(scene_adaptations)) * pupil_size * Frame._lift_to_multichannel(scene_lums)

        # Tone mapping.
        max_mapped_inv = np.abs(1/scene_max_response - 1/view_max_response) * (view_scale_factor / view_semi_saturation)
        saturation_terms = \
            (scene_semi_saturations * np.tile(np.expand_dims(view_scale_factor, [0, 1]), [rows, cols, 1])) / \
            np.tile(np.expand_dims(view_semi_saturation * global_scene_scale_factor, [0, 1]), [rows, cols, 1])

        self._max_mapped_inv = max_mapped_inv
        self._saturation_terms = saturation_terms

        output_lms = lms / (lms * np.tile(np.expand_dims(self._max_mapped_inv, [0, 1]), [rows, cols, 1]) + self._saturation_terms)
        output_xyz = self._lms2xyz(output_lms)

        out_frame = Frame(self.out_standard)
        out_frame.xyz = output_xyz / output_xyz[..., 1].max()

        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Reinhard's 2012 TMO.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)

        # First pass
        mean_scene_max_response = 0
        mean_view_max_response = 0
        mean_view_scale_factor = 0
        mean_view_semi_saturation = 0
        for frame in video:
            xyz = np.clip(frame.xyz * frame.standard.linear_range, 1e-4, None)
            lum = xyz[..., 1]
            global_scene_max_lum = np.percentile(lum, 90)
            pupil_size = Reinhard12TMO._pupil_size(global_scene_max_lum)

            global_scene_white_point = Reinhard12TMO._estimate_scene_white_point(xyz)
            global_scene_lum = global_scene_white_point[1]
            global_scene_max_white_point = global_scene_white_point * global_scene_max_lum / global_scene_lum
            global_scene_max_white_point_lms = self._xyz2lms(global_scene_max_white_point)
            global_scene_scale_factor = pupil_size * Reinhard12TMO._bleaching_function(global_scene_white_point)

            scene_max_response = Reinhard12TMO._max_neural_response(pupil_size * Reinhard12TMO._bleaching_function(global_scene_max_white_point_lms) * global_scene_max_white_point_lms)
            view_max_response = Reinhard12TMO._max_neural_response(pupil_size * Reinhard12TMO._bleaching_function(self._view_disp_max_white_point_lms) * self._view_disp_max_white_point_lms)
            view_scale_factor = pupil_size * Reinhard12TMO._bleaching_function(self._view_disp_white_point_lms)
            view_semi_saturation = self._view_disp_adaptation * view_scale_factor * self._view_disp_white_point + (1 - self._view_disp_adaptation) * pupil_size * self._view_disp_luminance

            mean_scene_max_response += scene_max_response
            mean_view_max_response += view_max_response
            mean_view_scale_factor += view_scale_factor
            mean_view_semi_saturation += view_semi_saturation
        video.reset()

        mean_scene_max_response /= video.num_frames
        mean_view_max_response /= video.num_frames
        mean_view_scale_factor /= video.num_frames
        mean_view_semi_saturation /= video.num_frames

        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        for frame in video:
            xyz = np.clip(frame.xyz * frame.standard.linear_range, 1e-4, None)
            cuts = self._median_cut(xyz)
            lms = self._xyz2lms(xyz)
            for cut in cuts:
                cut['lum_adapt'] = cut['white_point'][1]
                cut['degree_adapt'] = self._get_adaptation(cut['lum_adapt'])
                cut['effective_white_point'] = pupil_size * Reinhard12TMO._bleaching_function(cut['white_point']) * cut['white_point']
                cut['effective_white_point_lms'] = self._xyz2lms(cut['effective_white_point'])

            rows, cols, _ = lms.shape
            row_inds, col_inds = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

            scene_lums = np.zeros((rows, cols))
            scene_adaptations = np.zeros((rows, cols))
            scene_eff_white_points_lms = np.zeros((rows, cols, 3))
            norm_factors = np.zeros((rows, cols))

            for cut in cuts:
                dists = np.sqrt((row_inds - cut['center'][0])**2 + (col_inds - cut['center'][1])**2)
                weights = (1 - np.cos(np.pi/2 - self._relative_visual_angle * dists/self._diag_size))**4
                scene_lums = scene_lums + weights*cut['lum_adapt']
                scene_adaptations = scene_adaptations + weights*cut['degree_adapt']
                scene_eff_white_points_lms = scene_eff_white_points_lms + Frame._lift_to_multichannel(weights)*np.tile(np.expand_dims(cut['effective_white_point_lms'], [0, 1]), [rows, cols, 1])
                norm_factors += weights

            scene_lums = scene_lums / norm_factors
            scene_adaptations = scene_adaptations / norm_factors
            scene_eff_white_points_lms = scene_eff_white_points_lms / Frame._lift_to_multichannel(norm_factors)

            scene_semi_saturations = \
                Frame._lift_to_multichannel(scene_adaptations) * scene_eff_white_points_lms + \
                (1 - Frame._lift_to_multichannel(scene_adaptations)) * pupil_size * Frame._lift_to_multichannel(scene_lums)

            # Tone mapping.
            max_mapped_inv = np.abs(1/mean_scene_max_response - 1/mean_view_max_response) * (mean_view_scale_factor / mean_view_semi_saturation)
            saturation_terms = \
                (scene_semi_saturations * np.tile(np.expand_dims(mean_view_scale_factor, [0, 1]), [rows, cols, 1])) / \
                np.tile(np.expand_dims(mean_view_semi_saturation * global_scene_scale_factor, [0, 1]), [rows, cols, 1])

            self._leaky_integrate_params(max_mapped_inv, saturation_terms)

            output_lms = lms / (lms * np.tile(np.expand_dims(self._max_mapped_inv, [0, 1]), [rows, cols, 1]) + self._saturation_terms)
            output_xyz = self._lms2xyz(output_lms)

            out_frame = Frame(self.out_standard)
            out_frame.xyz = output_xyz / output_xyz[..., 1].max()
            out_video.append(out_frame)

        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a video smoothly using Reinhard's 2012 TMO and leaky integration.

        Args:
            video: Video to be tone-mapped
            out_filepath: Path to which tone-mapped video must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        for frame in video:
            xyz = np.clip(frame.xyz * frame.standard.linear_range, 1e-4, None)
            cuts = self._median_cut(xyz)
            lum = xyz[..., 1]
            lms = self._xyz2lms(xyz)
            global_scene_max_lum = np.percentile(lum, 90)
            pupil_size = Reinhard12TMO._pupil_size(global_scene_max_lum)

            global_scene_white_point = Reinhard12TMO._estimate_scene_white_point(xyz)
            global_scene_lum = global_scene_white_point[1]
            global_scene_max_white_point = global_scene_white_point * global_scene_max_lum / global_scene_lum
            global_scene_max_white_point_lms = self._xyz2lms(global_scene_max_white_point)
            global_scene_scale_factor = pupil_size * Reinhard12TMO._bleaching_function(global_scene_white_point)

            scene_max_response = Reinhard12TMO._max_neural_response(pupil_size * Reinhard12TMO._bleaching_function(global_scene_max_white_point_lms) * global_scene_max_white_point_lms)
            view_max_response = Reinhard12TMO._max_neural_response(pupil_size * Reinhard12TMO._bleaching_function(self._view_disp_max_white_point_lms) * self._view_disp_max_white_point_lms)
            view_scale_factor = pupil_size * Reinhard12TMO._bleaching_function(self._view_disp_white_point_lms)
            view_semi_saturation = self._view_disp_adaptation * view_scale_factor * self._view_disp_white_point + (1 - self._view_disp_adaptation) * pupil_size * self._view_disp_luminance

            for cut in cuts:
                cut['lum_adapt'] = cut['white_point'][1]
                cut['degree_adapt'] = self._get_adaptation(cut['lum_adapt'])
                cut['effective_white_point'] = pupil_size * Reinhard12TMO._bleaching_function(cut['white_point']) * cut['white_point']
                cut['effective_white_point_lms'] = self._xyz2lms(cut['effective_white_point'])

            rows, cols = lum.shape
            row_inds, col_inds = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')

            scene_lums = np.zeros((rows, cols))
            scene_adaptations = np.zeros((rows, cols))
            scene_eff_white_points_lms = np.zeros((rows, cols, 3))
            norm_factors = np.zeros((rows, cols))

            for cut in cuts:
                dists = np.sqrt((row_inds - cut['center'][0])**2 + (col_inds - cut['center'][1])**2)
                weights = (1 - np.cos(np.pi/2 - self._relative_visual_angle * dists/self._diag_size))**4
                scene_lums = scene_lums + weights*cut['lum_adapt']
                scene_adaptations = scene_adaptations + weights*cut['degree_adapt']
                scene_eff_white_points_lms = scene_eff_white_points_lms + Frame._lift_to_multichannel(weights)*np.tile(np.expand_dims(cut['effective_white_point_lms'], [0, 1]), [rows, cols, 1])
                norm_factors += weights

            scene_lums = scene_lums / norm_factors
            scene_adaptations = scene_adaptations / norm_factors
            scene_eff_white_points_lms = scene_eff_white_points_lms / Frame._lift_to_multichannel(norm_factors)

            scene_semi_saturations = \
                Frame._lift_to_multichannel(scene_adaptations) * scene_eff_white_points_lms + \
                (1 - Frame._lift_to_multichannel(scene_adaptations)) * pupil_size * Frame._lift_to_multichannel(scene_lums)

            # Tone mapping.
            max_mapped_inv = np.abs(1/scene_max_response - 1/view_max_response) * (view_scale_factor / view_semi_saturation)
            saturation_terms = \
                (scene_semi_saturations * np.tile(np.expand_dims(view_scale_factor, [0, 1]), [rows, cols, 1])) / \
                np.tile(np.expand_dims(view_semi_saturation * global_scene_scale_factor, [0, 1]), [rows, cols, 1])

            self._leaky_integrate_params(max_mapped_inv, saturation_terms)

            output_lms = lms / (lms * np.tile(np.expand_dims(self._max_mapped_inv, [0, 1]), [rows, cols, 1]) + self._saturation_terms)
            output_xyz = self._lms2xyz(output_lms)

            out_frame = Frame(self.out_standard)
            out_frame.xyz = output_xyz / output_xyz[..., 1].max()
            out_video.append(out_frame)

        out_video.close()
