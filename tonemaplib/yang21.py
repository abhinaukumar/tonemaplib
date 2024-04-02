import os
from typing import List, Optional, Tuple

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from .laplacianet.utils.utilities import cut_dark_end_percent, norm_0_to_1
from .laplacianet.utils.utils_lap_pyramid import lpyr_gen, lpyr_enlarge_to_top_but_bottom, cond, body
from .laplacianet.net import net_new_structure as ns

from .tmo import TMO
from .boitard12 import Boitard12TMO
from videolib import Frame, Video, standards
from videolib.standards import Standard
from videolib.cvt_color import rgb2yuv

tf.disable_v2_behavior()


class Yang21TMO(TMO):
    '''
    Implementation of Yang's 2021 Deep Reformulated Laplacial Tone Mapping

    Args:
        out_standard (Standard): Standard to which outputs must conform.
        video_mode (str): Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
        out_format (str): Format to which output video must be written.
        desat (float): Desaturation parameter for color correction.

    Refs:
        Yang, J., Liu, Z., Lin, M., Yanushkevich, S. & Yadid-Pecht, O. (2021).
        "Deep reformulated Laplacian tone mapping".
        arXiv preprint arXiv:2102.00348.

        https://github.com/linmc86/Deep-Reformulated-Laplacian-Tone-Mapping
    '''
    def __init__(
        self,
        out_standard: Optional[Standard] = standards.sRGB,
        video_mode: str = 'framewise',
        out_format: Optional[str] = 'encoded',
        desat: float = 0.4,
    ) -> None:
        '''
        Initializer.

        Args:
            out_standard: Standard to which outputs must conform (Default: sRGB).
            video_mode: Method used to tone map videos. Must be one of 'framewise', 'shot' or 'smooth' (Default: 'framewise').
            out_format: Format to which output video must be written (Default: 'encoded').
            desat: Desaturation parameter for color correction (Default: 0.4).
        '''
        super().__init__(out_standard=out_standard, video_mode=video_mode, out_format=out_format)
        tf.logging.set_verbosity(tf.logging.ERROR)
        self.desat = desat
        self.pad_width = 10
        self.levels = 4  # Number of Laplacian Pyramid levels
        self.checkpoint_path = os.path.join(os.environ.get('YANG21_FILES_DIR', ''), 'checkpoint', 'demo')  # models are saved here

    @property
    def params(self) -> List[str]:
        '''
        Return a list of parameter names that define the TMO.
        '''
        return ['video_mode', 'desat']

    def _init_network(self) -> None:
        '''
        Construct network graph and load weights.
        '''
        self.sess = tf.Session(config=tf.ConfigProto())
        self.high_layer_input = tf.placeholder(tf.float32, name='high_in')
        self.bot_layer_input = tf.placeholder(tf.float32, name='bot_in')
        self.height_input = tf.placeholder(tf.int32, name='h')
        self.width_input = tf.placeholder(tf.int32, name='w')
        self.output = self._construct_graph()

        variables_to_restore = []
        for v in tf.global_variables():
            if not (v.name.startswith('vgg_16')):
                variables_to_restore.append(v)

        # print(self.checkpoint_path)
        saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(self.checkpoint_path)
            saver.restore(self.sess, full_path)
        else:
            raise RuntimeError('Did not load checkpoint.')

    def _dualize(self, py_layers: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Convert Laplacian pyramid into bottom and frequency layers.

        Args:
            py_layers: Layers of the Laplacian Pyramid

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing frqeuency and bottom layers
        '''
        freq_layer = 0
        bottom_layer = py_layers[-1]
        freq_layers = py_layers[:-1]
        for item in range(0, len(freq_layers)):
            freq_layer += freq_layers[item]

        dual_layers = (freq_layer, bottom_layer)
        return dual_layers

    def _preprocessing_transform(self, hdr_gray: np.ndarray, limits: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Preprocess HDR luminance prior to applying network.

        Args:
            hdr_gray: HDR luminance image.
            limits: Limits of data to use when normalizing (Default: None).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing high and bottom layers
        '''
        # Data preprocessing
        hdr_gray_clipped = cut_dark_end_percent(hdr_gray, 0.001)
        hdr_logged = np.log(hdr_gray_clipped + np.finfo(float).eps)
        # log(np.finfo(float).eps) is -52.  so we clipped -50 off
        if limits is None:
            minval = maxval = None
        else:
            minval, maxval = limits
        hdr_preprocessed = np.clip(hdr_logged, a_min=-50, a_max=maxval)

        # Bring to [0,1]
        if limits is None:
            hdr_ready = norm_0_to_1(hdr_preprocessed)
        else:
            hdr_ready = (hdr_preprocessed - minval) / (maxval - minval)

        # Create Laplacian Pyramid
        hdr_py = lpyr_gen(hdr_ready, self.levels)

        hdr_py_aligned, _ = lpyr_enlarge_to_top_but_bottom(hdr_py)
        [high_layer, bot_layer] = self._dualize(hdr_py_aligned)
        high_layer = np.expand_dims(np.expand_dims(high_layer, axis=0), 3)
        bot_layer = np.expand_dims(np.expand_dims(bot_layer, axis=0), axis=3)

        # Padding the bottom layer to avoid the ripple-border effect
        paddings = np.array([[0, 0], [self.pad_width, self.pad_width], [self.pad_width, self.pad_width], [0, 0]])
        bot_layer = np.pad(bot_layer, paddings)
        return high_layer, bot_layer

    def _postprocessing_transform(self, lum_out: np.ndarray, lum_in: np.ndarray, rgb_in: np.ndarray) -> np.ndarray:
        '''
        Convert tone-mapped luminance to tone-mapped RGB.

        Args:
            lum_out: Output luminance from network.
            lum_in: Input HDR luminance.
            rgb_in: Input HDR RGB values.

        Returns:
            np.ndarray: Array containing toenmapped RGB image.
        '''
        rgb_out = np.zeros_like(rgb_in)
        for channel_ind in range(3):
            rgb_out[:, :, channel_ind] = ((rgb_in[:, :, channel_ind]/(lum_in + 1e-10)) ** (1 - self.desat))*lum_out
        return rgb_out

    def _calc_bot_shape(self, h: tf.placeholder, w: tf.placeholder) -> Tuple[tf.Tensor, tf.Tensor]:
        '''
        Calculate the dimensions of the bottom layer.

        Args:
            h: Placeholder containing the height of the original image.
            w: Placeholder containing the width of the original image.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Tuple containing tensors containing the height and width of the bottom layer.
        '''
        new_h, new_w = h, w
        for i in range(self.levels):
            new_h = tf.cast(tf.ceil(tf.divide(new_h, 2)), tf.int32)
            new_w = tf.cast(tf.ceil(tf.divide(new_w, 2)), tf.int32)
        return new_h, new_w

    def _construct_graph(self) -> tf.Tensor:
        '''
        Construct the network's computational graph.

        Returns:
            tf.Tenosr: Tensor containing the network output.
        '''
        high_out = ns.nethighlayer(self.high_layer_input)
        bot_out = ns.netbotlayer(self.bot_layer_input)

        bot_shape = tf.shape(bot_out)
        bot_out = tf.slice(bot_out, [0, self.pad_width, self.pad_width, 0], [-1, bot_shape[1]-self.pad_width*2, bot_shape[2]-self.pad_width*2, -1])

        bot_h, bot_w = self._calc_bot_shape(self.height_input, self.width_input)
        bot_out = tf.squeeze(bot_out)
        tfbot_upsampling = tf.reshape(bot_out, [bot_h, bot_w])

        i = tf.constant(0)
        n = tf.constant(self.levels)
        fullsize_bottom, i, n = tf.while_loop(cond, body, [tfbot_upsampling, i, n],
                                              shape_invariants=[tf.TensorShape([None, None]), i.get_shape(),
                                                                n.get_shape()])

        fullsize_bottom = tf.slice(fullsize_bottom, [0, 0], [self.height_input, self.width_input])
        fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=0)
        fullsize_bottom = tf.expand_dims(fullsize_bottom, axis=3)

        imgpatch = high_out + fullsize_bottom
        return ns.netftlayer(imgpatch)

    def tonemap_frame(self, frame: Frame) -> Frame:
        '''
        Tone-map a frame using Yang's 2021 TMO.

        Args:
            frame: Frame to be tone-mapped.

        Returns:
            Frame: Tone-mapped frame.
        '''
        self._init_network()

        linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
        high_layer, bot_layer = self._preprocessing_transform(linear_yuv_const[..., 0] * frame.standard.linear_range)

        lum_out = self.sess.run(
            [self.output],
            feed_dict={
                self.high_layer_input: high_layer,
                self.bot_layer_input: bot_layer,
                self.height_input: frame.height,
                self.width_input: frame.width
            }
        )
        lum_out = norm_0_to_1(np.squeeze(lum_out))

        temp_out_frame = Frame(self.out_standard)
        temp_out_frame.rgb = self.out_standard.range * self._postprocessing_transform(lum_out, linear_yuv_const[..., 0], frame.linear_rgb)
        out_frame = self.gamut_map(temp_out_frame)

        self.sess.close()
        tf.reset_default_graph()
        return out_frame

    def tonemap_video_framewise(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a video using Yang's 2021 TMO.

        Args:
            video: Video object containing the input HDR video.
            out_filepath: Path to which output file must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        self._init_network()

        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            high_layer, bot_layer = self._preprocessing_transform(linear_yuv_const[..., 0] * frame.standard.linear_range)

            lum_out = self.sess.run(
                [self.output],
                feed_dict={
                    self.high_layer_input: high_layer,
                    self.bot_layer_input: bot_layer,
                    self.height_input: frame.height,
                    self.width_input: frame.width
                }
            )
            lum_out = norm_0_to_1(np.squeeze(lum_out))

            temp_out_frame = Frame(self.out_standard)
            temp_out_frame.rgb = self.out_standard.range * self._postprocessing_transform(lum_out, linear_yuv_const[..., 0], frame.linear_rgb)
            out_frame = self.gamut_map(temp_out_frame)

            out_video.write_frame(out_frame)

        tf.reset_default_graph()
        self.sess.close()
        out_video.close()

    def tonemap_shot(self, video: Video, out_filepath: str) -> None:
        '''
        Tone-map a shot using Yang's 2021 TMO.

        Args:
            video: Video object containing the input HDR video.
            out_filepath: Path to which output file must be written.
        '''
        TMO._assert_out_filepath_is_valid(out_filepath)
        minval = np.inf
        maxval = -np.inf
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            hdr_gray_clipped = cut_dark_end_percent(linear_yuv_const[..., 0], 0.001)
            hdr_logged = np.log(hdr_gray_clipped + np.finfo(float).eps)
            # log(np.finfo(float).eps) is -52.  so we clipped -50 off
            hdr_preprocessed = np.clip(hdr_logged, a_min=-50, a_max=maxval)
            minval = min(minval, hdr_preprocessed.min())
            maxval = max(maxval, hdr_preprocessed.max())
        video.reset()

        self._init_network()

        out_video = Video(out_filepath, self.out_standard, 'w', format=self.out_format)
        for frame in video:
            linear_yuv_const = rgb2yuv(frame.linear_rgb, frame.standard, range=1)
            high_layer, bot_layer = self._preprocessing_transform(linear_yuv_const[..., 0])

            lum_out = self.sess.run(
                [self.output],
                feed_dict={
                    self.high_layer_input: high_layer,
                    self.bot_layer_input: bot_layer,
                    self.height_input: frame.height,
                    self.width_input: frame.width
                }
            )
            lum_out = norm_0_to_1(np.squeeze(lum_out))

            temp_out_frame = Frame(self.out_standard)
            temp_out_frame.rgb = self.out_standard.range * self._postprocessing_transform(lum_out, linear_yuv_const[..., 0], frame.linear_rgb)
            out_frame = self.gamut_map(temp_out_frame)

            out_video.write_frame(out_frame)

        tf.reset_default_graph()
        self.sess.close()
        out_video.close()

    def tonemap_video_smooth(self, video: Video, out_filepath: str, boitard_scale_method: Optional[str] = 'max'):
        '''
        Tone-map a video smoothly using Yang's 2021 TMO and Boitard's 2012 TMO.

        Args:
            video: Video to be tone-mapped.
            out_filepath: Path to which tone-mapped video must be written.
        Returns:
            Frame: Tone-mapped frame.
        '''
        boitard_tmo = Boitard12TMO(self.out_standard, self.out_format, base_tmo=self, scale_method=boitard_scale_method)
        boitard_tmo(video, out_filepath)
