import argparse
import os
from joblib import Parallel, delayed
from scipy.io import loadmat
from videolib import Video, standards

import tonemaplib
from tonemaplib.utils import import_python_file


def smoothen_video_using_keys(args, video_path, src_keys_path, out_video_path, device=None):
    # if os.path.isfile(out_video_path):
    #     return
    tmo = tonemaplib.Boitard12TMO(scale_method=args.scale_method)
    f = loadmat(src_keys_path)
    src_keys = f['keys'].squeeze()
    video = Video(video_path, standards.sRGB, 'r', args.width, args.height)
    tmo.postprocess_video_using_keys(video, src_keys, out_video_path)
    print('Processed {}'.format(video.file_path))
    video.close()


def main():
    parser = argparse.ArgumentParser(description='Program to smoothen a dataset using Boitard12.')
    parser.add_argument('--scale_method', help='Scale method to be used by Boitard12TMO.', required=False, type=str, default='max')
    parser.add_argument('--videos_file', help='Path to Python file containing a list of paths of SDR videos', required=True, type=str)
    parser.add_argument('--src_keys_file', help='Path to Python file containing a list of files containing keys of HDR videos', required=True, type=str)
    parser.add_argument('--out_dir', help='Path to directory where output videos must be written.', required=True, type=str)
    parser.add_argument('--width', help='Width of video (Default: 1920).', type=int, required=False, default=1920)
    parser.add_argument('--height', help='Height of video (Default: 1080).', type=int, required=False, default=1080)
    parser.add_argument('--jobs', help='Number of jobs to launch (Default: 1).', type=int, required=False, default=1)
    args = parser.parse_args()

    video_list_module = import_python_file(args.videos_file)
    src_key_list_module = import_python_file(args.src_keys_file)
    if hasattr(video_list_module, 'videos'):
        if isinstance(video_list_module.videos, list):
            video_list = video_list_module.videos
        else:
            raise TypeError('\'videos\' must be of type List.')
    else:
        raise AttributeError('Attribute \'videos\' not found.')

    if hasattr(src_key_list_module, 'files'):
        if isinstance(src_key_list_module.files, list):
            src_key_list = src_key_list_module.files
        else:
            raise TypeError('\'files\' must be of type List.')
    else:
        raise AttributeError('Attribute \'files\' not found.')

    if args.scale_method not in ['mean', 'max']:
        raise ValueError('scale_method must be either \'mean\' or \'max\'')
    # out_dir = os.path.join(args.out_dir, os.path.basename(os.path.dirname(video_list[0])) + '_Boitard12TMO_scale_method_' + args.scale_method)
    out_dir = os.path.join(args.out_dir, os.path.basename(os.path.dirname(video_list[0])).replace('framewise', 'smooth'))
    print('Saving results to', out_dir)
    if not os.path.isdir(out_dir):
        print('Creating', out_dir)
        os.mkdir(out_dir)

    out_video_list = [os.path.join(out_dir, os.path.basename(video_path)) for video_path in video_list]

    Parallel(n_jobs=args.jobs)([delayed(smoothen_video_using_keys)(args, video_path, src_key_path, out_video_path) for video_path, src_key_path, out_video_path in zip(video_list, src_key_list, out_video_list)])


if __name__ == '__main__':
    main()
