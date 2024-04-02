import argparse
import os
from joblib import Parallel, delayed
from videolib import Video, standards

from tonemaplib import get_tmoclass
from tonemaplib.utils import import_python_file, load_args


def run_tmo_on_video(args, video_path, standard, out_video_path, device=None):
    TMOClass = get_tmoclass(args.tmo)
    tmo_args, tmo_kwargs = load_args(args.tmo_args_file)
    if args.tmo == 'Rana19':
        tmo_kwargs['device'] = device
    tmo = TMOClass(*tmo_args, **tmo_kwargs)
    video = Video(video_path, standard, 'r', args.width, args.height, quantization=args.quantization, dither=args.dither)
    tmo.tonemap_shot(video, out_video_path)
    print('Processed {}'.format(video.file_path))
    video.close()
    if args.tmo == 'Rana19':
        import torch
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Program to run a TMO on a dataset.')
    parser.add_argument('--tmo', help='Name of the TMO', required=True, type=str)
    parser.add_argument('--tmo_args_file', help='Path to Python file containing TMO\'s args.', type=str, default=None)
    parser.add_argument('--videos_file', help='Path to Python file containing a list of paths to YUV files of HDR videos', required=True, type=str)
    parser.add_argument('--out_dir', help='Path to directory where output videos must be written.', required=True, type=str)
    parser.add_argument('--width', help='Width of video (Default: 1920).', type=int, required=False, default=1920)
    parser.add_argument('--height', help='Height of video (Default: 1080).', type=int, required=False, default=1080)
    parser.add_argument('--jobs', help='Number of jobs to launch (Default: 1).', type=int, required=False, default=1)
    parser.add_argument('--quantization', help='Number of quantization levels to use for HDR video (Default: None - uses full bitdepth).', type=int, required=False, default=None)
    parser.add_argument('--dither', help='Flag to dither after quantization (Default: False)', action='store_true', required=False, default=False)
    args = parser.parse_args()

    video_list_module = import_python_file(args.videos_file)
    if hasattr(video_list_module, 'videos'):
        if isinstance(video_list_module.videos, list):
            video_list = video_list_module.videos
        else:
            raise TypeError('\'videos\' must be of type List.')
    else:
        raise AttributeError('Attribute \'videos\' not found.')

    if hasattr(video_list_module, 'standard'):
        if isinstance(video_list_module.standard, standards.Standard):
            standard = video_list_module.standard
        else:
            raise TypeError('\'standard\' must be of type standards.Standard.')
    else:
        raise AttributeError('Attribute \'standard\' not found.')

    for video_path in video_list:
        if video_path[-4:] != '.yuv':
            raise OSError('All input files must be YUV files.')

    TMOClass = get_tmoclass(args.tmo)
    tmo_args, tmo_kwargs = load_args(args.tmo_args_file)
    tmo = TMOClass(*tmo_args, **tmo_kwargs)

    out_dir = os.path.join(args.out_dir, str(tmo))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    del tmo
    if args.tmo == 'Rana19':
        import torch
        torch.cuda.empty_cache()

    out_video_list = [os.path.join(out_dir, os.path.basename(video_path)) for video_path in video_list]

    if args.tmo == 'Rana19':
        device_count = torch.cuda.device_count()
        Parallel(n_jobs=args.jobs)([delayed(run_tmo_on_video)(args, video_path, out_video_path, (device_id+1) % device_count) for device_id, (video_path, out_video_path) in enumerate(zip(video_list, out_video_list))])
    else:
        Parallel(n_jobs=args.jobs)([delayed(run_tmo_on_video)(args, video_path, standard, out_video_path) for video_path, out_video_path in zip(video_list, out_video_list)])


if __name__ == '__main__':
    main()
