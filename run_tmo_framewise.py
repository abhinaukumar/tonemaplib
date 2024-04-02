import argparse

import matplotlib.pyplot as plt
from videolib import Video, standards

from tonemaplib import get_tmoclass
from tonemaplib.utils import load_args

parser = argparse.ArgumentParser(description='Program to run a TMO framewise on a video.')
parser.add_argument('--tmo', help='Name of TMO class', required=True, type=str)
parser.add_argument('--tmo_args_file', help='Path to Python file containing TMO\'s args.', type=str, default=None)
parser.add_argument('--video', help='Path to HDR video\'s YUV file', required=True, type=str)
parser.add_argument('--interactive', help='Flag to run matplotlib in interactive mode', action='store_true', default=False)
parser.add_argument('--out_video', help='Path to output video. Output not saved if not provided.', required=False, type=str, default=None)
parser.add_argument('--width', help='Width of video (Default: 1920).', type=int, required=False, default=1920)
parser.add_argument('--height', help='Height of video (Default: 1080).', type=int, required=False, default=1080)
parser.add_argument('--max_frames', help='Maximum number of frames to process (Default: None - all frames are processed).', type=int, required=False, default=None)
args = parser.parse_args()

TMOClass = get_tmoclass(args.tmo)
tmo_args, tmo_kwargs = load_args(args.tmo_args_file)

tmo = TMOClass(*tmo_args, **tmo_kwargs)
video = Video(args.video, standards.rec_2100_pq, 'r', args.width, args.height)
if args.out_video is not None:
    out_video = Video(args.out_video, tmo.out_standard, 'w', args.width, args.height)
else:
    out_video = None

if args.interactive:
    plt.ion()
    plt.figure()

for frame_ind, frame in enumerate(video):
    if args.max_frames is not None and frame_ind >= args.max_frames:
        break

    frame_tm = tmo(frame)

    if not args.interactive:
        plt.figure()
    plt.imshow(frame_tm.rgb.astype('uint8'))
    if args.interactive:
        plt.pause(0.02)
    else:
        plt.show()

    if out_video is not None:
        out_video.append(frame_tm)

video.close()
if out_video is not None:
    out_video.close()
