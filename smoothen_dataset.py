import argparse
import os
from joblib import Parallel, delayed
from videolib import Video, standards

import tonemaplib
from tonemaplib.utils import import_python_file


def smoothen_video(args, video_path, src_video_path, out_video_path, device=None):
    tmo = tonemaplib.Boitard12TMO(scale_method=args.scale_method)
    src_video = Video(src_video_path, standards.rec_2100_pq, 'r', args.width, args.height)
    video = Video(video_path, standards.sRGB, 'r', args.width, args.height)
    tmo.postprocess_video(video, src_video, out_video_path)
    print('Processed {}'.format(video.file_path))
    video.close()


def main():
    parser = argparse.ArgumentParser(description='Program to smoothen a dataset using Boitard12.')
    parser.add_argument('--scale_method', help='Scale method to be used by Boitard12TMO.', required=True, type=str)
    parser.add_argument('--videos_file', help='Path to Python file containing a list of paths of SDR videos', required=True, type=str)
    parser.add_argument('--src_videos_file', help='Path to Python file containing a list of paths of HDR videos', required=True, type=str)
    parser.add_argument('--out_dir', help='Path to directory where output videos must be written.', required=True, type=str)
    parser.add_argument('--width', help='Width of video (Default: 1920).', type=int, required=False, default=1920)
    parser.add_argument('--height', help='Height of video (Default: 1080).', type=int, required=False, default=1080)
    parser.add_argument('--jobs', help='Number of jobs to launch (Default: 1).', type=int, required=False, default=1)
    args = parser.parse_args()

    video_list_module = import_python_file(args.videos_file)
    src_video_list_module = import_python_file(args.src_videos_file)
    if hasattr(video_list_module, 'videos'):
        if isinstance(video_list_module.videos, list):
            video_list = video_list_module.videos
        else:
            raise TypeError('\'videos\' must be of type List.')
    else:
        raise AttributeError('Attribute \'videos\' not found.')

    if hasattr(src_video_list_module, 'videos'):
        if isinstance(src_video_list_module.videos, list):
            src_video_list = src_video_list_module.videos
        else:
            raise TypeError('\'videos\' must be of type List.')
    else:
        raise AttributeError('Attribute \'videos\' not found.')

    if args.scale_method not in ['mean', 'max']:
        raise ValueError('scale_method must be either \'mean\' or \'max\'')
    out_dir = os.path.join(args.out_dir, os.path.dirname(video_list[0]) + '_Boitard12TMO_scale_method_' + args.scale_method)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    out_video_list = [os.path.join(out_dir, os.path.basename(video_path)) for video_path in video_list]

    Parallel(n_jobs=args.jobs)([delayed(smoothen_video)(args, video_path, src_video_path, out_video_path) for video_path, src_video_path, out_video_path in zip(video_list, src_video_list, out_video_list)])


if __name__ == '__main__':
    main()
