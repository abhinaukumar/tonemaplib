from typing import Optional, List

import numpy as np

from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.buffer import CircularBuffer
from .tmo import TMO


class Oskarsson17TMO(TMO):
    '''
    Implementation of Oskarsson's (Democratic) TMO from 2017.
    Adapted from MATLAB implementation linked in refs.

    Args:
        out_standard (Standard): Standard to which the output frames must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        num_clusters (int): Number of clusters to use in K-means clustering.
        hist_context (int): Number of frames to use for histograms when tone-mapping videos.
        channelwise_model_weight (float): Weight assigned to channel-wise tone-mapped output.

    Refs:
        Oskarsson, M. (2017)
        "Temporally consistent tone mapping of images and video using optimal k-means clustering."
        Journal of mathematical imaging and vision 57.2: 225-238.

        https://github.com/hamburgerlady/democratic-tonemap
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        num_clusters: Optional[int] = 256,
        hist_context: Optional[int] = 7,
        channelwise_model_weight: Optional[float] = 0.7
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which the output frames must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            num_clusters: Number of clusters to use in K-means clustering (Default: 256).
            hist_context: Number of frames to use for histograms when tone-mapping videos (Default: 7).
            channelwise_model_weight: Weight assigned to channel-wise tone-mapped output (Default: 0.7).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        self.num_clusters: int = num_clusters
        self.hist_context: int = hist_context
        self.channelwise_model_weight: float = channelwise_model_weight
        self._num_bins: int = 5000
        self._bins: np.ndarray = np.linspace(0, np.log(2), self._num_bins+1)  # All inputs are in the range log(1) = 0 to log(2).
        self._bin_centers: np.ndarray = (self._bins[:-1] + self._bins[1:])/2
        self._interp_length: int = 20
        self._prev_hist: np.ndarray = None

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'num_clusters', 'hist_context', 'channelwise_model_weight']

    def _get_buf_hist(self, buf: CircularBuffer, normalize: Optional[bool] = True) -> np.ndarray:
        '''
        Get combined histogram of all images in the buffer.

        Args:
            buf: Buffer containing images.
            normalize: Flag to normalize the histogram to sum to 1 (Default: True).

        Returns:
            np.ndarray: Histogram, normalized to sum to 1 if normalize is True.
        '''
        hist = np.zeros((self._num_bins,))
        for img in buf:
            hist = hist + np.histogram(img.flatten(), bins=self._bins)[0]
        if normalize:
            hist = hist / hist.sum()
        return hist

    def _get_hist(self, img: np.ndarray, normalize: Optional[bool] = True) -> np.ndarray:
        '''
        Get histogram of an image.

        Args:
            img: Input image.
            normalize: Flag to normalize the histogram to sum to 1 (Default: True).

        Returns:
            np.ndarray: Histogram, normalized to sum to 1 if normalize is True.
        '''
        hist = np.histogram(img.flatten(), bins=self._bins)[0]
        if normalize:
            hist = hist / hist.sum()
        return hist

    def _cluster_hist(self, hist: np.ndarray) -> np.ndarray:
        '''
        Apply K-means clustering on a histogram.

        Args:
            hist: Histogram representing data to be clustered.

        Returns:
            np.ndarray: Cluster centers.
        '''
        cum_hist = np.zeros((len(hist)+1,))
        cum_hist[1:] = np.cumsum(hist)

        # single_cluster_cost[i, j]: Cost of clustering bins[i:j] into a cluster
        single_cluster_cost = np.full((self._num_bins, self._num_bins+1), np.inf)
        single_cluster_centers = np.zeros((self._num_bins, self._num_bins+1))

        # Compute cluster costs and corresponding centers.
        diag_inds = np.arange(self._num_bins).astype('int')
        # Base case - Single element clusters.
        single_cluster_cost[diag_inds, diag_inds+1] = 0
        single_cluster_centers[diag_inds, diag_inds+1] = self._bin_centers
        # Recursively update cluster centers.
        for hi in range(2, self._num_bins+1):
            mask = (diag_inds+2 <= hi)
            single_cluster_centers[mask, hi] = (hist[hi-1]*self._bin_centers[hi-1] + cum_hist[hi-1]*single_cluster_centers[mask, hi-1])/cum_hist[hi] if cum_hist[hi] != 0 else 0
            single_cluster_cost[mask, hi] = single_cluster_cost[mask, hi-1] + \
                ((hist[hi-1]*cum_hist[hi-1]/cum_hist[hi])*(self._bin_centers[hi-1] - single_cluster_centers[mask, hi-1])**2 if cum_hist[hi] != 0 else 0)

        # Start with single cluster.
        old_dp_costs = single_cluster_cost[0, :]
        # Use corresponding cluster centers
        old_dp_cluster_centers = [[center] for center in single_cluster_centers[0, :]]
        # Iteratively increment the number of clusters and optimally update costs and centers.
        for num_clusters in range(2, self.num_clusters+1):
            # Costs and centers for the new number of clusters.
            new_dp_costs = np.empty((self._num_bins+1))
            new_dp_costs.fill(np.inf)
            new_dp_cluster_centers = [[] for _ in range(self._num_bins+1)]
            new_dp_costs[num_clusters] = 0
            new_dp_cluster_centers[num_clusters] = list(self._bin_centers[:num_clusters])

            for num_bins in range(num_clusters+1, self._num_bins+1):
                # Code uses the vectorized form of the equation
                # costs[i] = dp[i, num_clusters-1] (Optimally cluster i bins) + single_cluster_cost[i, num_bins] (Group bins[i:num_bins] into one cluster),
                costs = old_dp_costs[:num_bins] + single_cluster_cost[:num_bins, num_bins]

                # Choose the one with the lowest cost.
                opt_ind = np.argmin(costs)
                new_dp_costs[num_bins] = costs[opt_ind]
                new_dp_cluster_centers[num_bins] = old_dp_cluster_centers[opt_ind] + [single_cluster_centers[opt_ind, num_bins]]

            # Update old costs and clusters to use in the next iteration.
            old_dp_costs = new_dp_costs
            old_dp_cluster_centers = new_dp_cluster_centers

        return np.array(new_dp_cluster_centers[self._num_bins])

    def _tone_curve(self, img: np.ndarray, centers: np.ndarray) -> np.ndarray:
        '''
        Apply tone curve to an input image based on the cluster centers.

        Args:
            img: Image to be tone mapped.
            centers: Cluster centers.

        Returns:
            np.ndarray: Tone-mapped image where values have been replaced by their closest cluster centers.
        '''
        costs = np.zeros(img.shape + (self.num_clusters,))
        for i, center in enumerate(centers):
            costs[..., i] = np.abs(img - center)
        center_inds = np.argmin(costs, axis=-1).astype('int32')
        return centers[center_inds]

    def tonemap_frame(self, frame: Frame, maxval: Optional[float] = None, centers: Optional[np.ndarray] = None) -> Frame:
        '''
        Tone-map a frame using Oskarsson's 2017 TMO.

        Args:
            frame: Frame to be tone-mapped.
            maxval: Maximum value to use when tone-mapping (Default: None).
            centers: Cluster centers to use when tone-mapping (Default: None).

        Returns:
            Frame: Tone-mapped frame.
        '''
        gray = frame.linear_rgb.max(-1)
        if maxval is None:
            maxval = gray.max()

        log_rgb = np.log(1 + frame.linear_rgb/maxval)
        log_gray = np.log(1 + gray/maxval)

        if centers is None:
            hist = self._get_hist(log_gray)
            centers = self._cluster_hist(hist)

        channelwise_tm = self._tone_curve(log_rgb, centers)
        gray_tm = Frame._lift_to_multichannel(self._tone_curve(log_gray, centers) / gray) * frame.linear_rgb

        temp_out_frame = Frame(frame.standard)
        temp_out_frame.linear_rgb = self.channelwise_model_weight * channelwise_tm + (1 - self.channelwise_model_weight) * gray_tm
        out_frame = self.gamut_map(temp_out_frame)
        return out_frame

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Oskarsson's 2017 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which output file must be written.
        '''
        maxval = -np.inf
        for frame in video:
            maxval = max(maxval, frame.linear_rgb.max())
        video.reset()

        hist = np.zeros((self._num_bins,))
        for frame in video:
            hist += self._get_hist(np.log(1 + frame.linear_rgb.max(-1)/maxval), normalize=True)
        hist /= hist.sum()
        video.reset()
        centers = self._cluster_hist(hist)

        out_video = Video(out_filepath, self.out_standard, 'w', video.width, video.height, self.out_format)
        for frame in video:
            out_frame = self.tonemap_frame(frame, maxval, centers)
            out_video.write_frame(out_frame)
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a video smoothly using Oskarsson's 2017 TMO and histogram interpolation.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which output file must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        temp_out_frame = Frame(video.standard)

        buf: CircularBuffer = None
        left_centers: np.ndarray = None
        right_centers: np.ndarray = None
        left_frame_ind: int = None
        right_frame_ind: int = None

        for frame_ind, frame in enumerate(video):
            if frame_ind % self._interp_length == 0:
                # Set up "left" tone-curve
                if frame_ind == 0:
                    buf = CircularBuffer(self.hist_context)
                    for i in range(min(self.hist_context//2 + (self.hist_context & 1), video.num_frames)):
                        buf.check_append(np.log(1 + video[i].linear_rgb.max(-1)/video[i].linear_rgb.max()))
                    left_frame_ind = 0

                    # Compute cluster centers
                    hist = self._get_buf_hist(buf)
                    buf.clear()
                    left_centers = self._cluster_hist(hist)
                else:
                    left_centers = right_centers
                    left_frame_ind = right_frame_ind

                # Set up "right" tone-curve
                right_frame_ind = min(left_frame_ind + self._interp_length, video.num_frames)
                buf = CircularBuffer(self.hist_context)
                for i in range(max(0, right_frame_ind-self.hist_context//2), min(right_frame_ind + self.hist_context//2 + (self.hist_context & 1), video.num_frames)):
                    buf.check_append(np.log(1 + video[i].linear_rgb.max(-1)/video[i].linear_rgb.max()))

                # Compute cluster centers
                hist = self._get_buf_hist(buf)
                buf.clear()
                right_centers = self._cluster_hist(hist)

            interp_weight = (right_frame_ind - frame_ind) / (right_frame_ind - left_frame_ind)
            centers = interp_weight * left_centers + (1 - interp_weight) * right_centers

            gray = frame.linear_rgb.max(-1)
            maxval = gray.max()
            log_rgb = np.log(1 + frame.linear_rgb/maxval)
            log_gray = np.log(1 + gray/maxval)

            channelwise_tm = self._tone_curve(log_rgb, centers)
            gray_tm = Frame._lift_to_multichannel(self._tone_curve(log_gray, centers) / gray) * frame.linear_rgb

            temp_out_frame.linear_rgb = self.channelwise_model_weight * channelwise_tm + (1 - self.channelwise_model_weight) * gray_tm
            out_video.append(self.gamut_map(temp_out_frame))

        out_video.close()
        return out_video
