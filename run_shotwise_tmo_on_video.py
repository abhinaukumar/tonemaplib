import argparse

import matplotlib.pyplot as plt
from videolib import standards, Video

from tonemaplib import get_tmoclass
from tonemaplib.utils import load_args

parser = argparse.ArgumentParser(description='Program to run a TMO on a video.')
parser.add_argument('--tmo', help='Name of the TMO', required=True, type=str)
parser.add_argument('--tmo_args_file', help='Path to Python file containing TMO\'s args.', type=str, default=None)
parser.add_argument('--video', help='Path to HDR video\'s YUV file', required=True, type=str)
parser.add_argument('--plot', help='Flag to show frames as plots', action='store_true', default=False)
parser.add_argument('--interactive', help='Flag to run matplotlib in interactive mode if showing frames', action='store_true', default=False)
parser.add_argument('--out_video', help='Path to output video.', required=True, type=str)
parser.add_argument('--width', help='Width of video (Default: 1920).', type=int, required=False, default=1920)
parser.add_argument('--height', help='Height of video (Default: 1080).', type=int, required=False, default=1080)
args = parser.parse_args()

TMOClass = get_tmoclass(args.tmo)
tmo_args, tmo_kwargs = load_args(args.tmo_args_file)

tmo = TMOClass(*tmo_args, **tmo_kwargs)
video = Video(args.video, standards.rec_2100_pq, 'r', args.width, args.height)
tmo.tonemap_shot(video, args.out_video)
video.close()

if args.plot:
    out_video = Video(args.out_video, tmo.out_standard, 'w', args.width, args.height)
    if args.interactive:
        plt.ion()
        plt.figure()

    for frame_tm in enumerate(out_video):
        if not args.interactive:
            plt.figure()
        plt.imshow(frame_tm.rgb.astype('uint8'))
        if args.interactive:
            plt.pause(0.02)
        else:
            plt.show()

    out_video.close()
