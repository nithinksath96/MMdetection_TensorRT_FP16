# Merge and convert real-time results
# Optionally, visualize the output
# This script does not need to run in real-time

import argparse
from os import scandir
from os.path import join, isfile

from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2
from vis.make_videos_numbered import worker_func as make_video_func

data_dir = '/data2/mengtial'
out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse1.1/vid/compute'))

text = [
    ['Single GPU Final Solution', 'Optimized and predictive (AP 17.8)'],
    ['Infinite GPU Final Solution', 'More responsive (AP 20.3)'],
]

dirs = [
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/pps_mrcnn50_nm_ds_s0.75_kf_fba_iou_lin_pkt/val'),
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/pps_mrcnn50_nm_inf_s0.75_kf_fba_iou_lin_pkt/val'),
]

seq = 'cd5bb988-092e-396c-8f33-e30969c98535'
fps = 30
vis_scale = 1
overwrite = True
make_video = True

clip_start = 210
# make 16:9 instead of 16:10
crop_top = 60
crop_bottom = 60


t_duration = 330
t_transition = 30
t_text_appear = 30

line_width = 15
line_color = [241, 159, 93]

# font_path = r'C:\Windows\Fonts\Rock.ttf'
font_path = '/home/mengtial/fonts/ROCK.TTF'

font = ImageFont.truetype(font_path, size=70)
text_region = [100, 90, 1490, 290]
text_region_alpha = 0.46 # assuming color black
text_xy = [150, 100]
text_line_sep = 85
text_color = (217, 217, 217)

# Smoothing functions
# map time from 0-1 to progress from 0-1
def ease_in_out(t):
    return -np.cos(np.pi*t)/2 + 0.5

# animations
def split_anime_accelerate(t, l, line_width):
    small_end = -line_width//2 - 1
    big_end = l + line_width//2

    start_pos = big_end
    end_pos = small_end
    p = ease_in_out(t)
    return start_pos + p*(end_pos - start_pos)


def main():
    line_color_np = np.array(line_color, dtype=np.uint8).reshape((1, 1, 3))


    seq_dir_out = mkdir2(join(out_dir, seq))
    frame_list = [item.name for item in scandir(join(dirs[0], seq)) if item.is_file() and item.name.endswith('.jpg')]
    frame_list = sorted(frame_list)
    
    n_method = len(text)
    fidx = 0
    j_start = 0
    for i in tqdm(range(n_method)):
        for j in range(j_start, t_duration):
            fidx += 1
            out_path = join(seq_dir_out, '%06d.jpg' % fidx)
            if not overwrite and isfile(out_path):
                continue

            img_A = Image.open(join(dirs[i], seq, frame_list[j + clip_start]))
            # cropping
            img_A = np.array(img_A)
            img_A = img_A[crop_top:-crop_bottom]

            progress_text = ease_in_out((j - j_start) / t_text_appear) if j - j_start < t_text_appear else 1
            # render text region
            img_A[text_region[1]:text_region[3] + 1, text_region[0]:text_region[2] + 1] = \
                np.round((1 + progress_text*(text_region_alpha - 1))*img_A[text_region[1]:text_region[3] + 1, text_region[0]:text_region[2] + 1]).astype(np.uint8)
            img_A_with_text = Image.fromarray(img_A)

            # using TrueType supported in PIL
            draw = ImageDraw.Draw(img_A_with_text)
            draw.text(
                (text_xy[0], text_xy[1]),
                text[i][0], (*text_color, 255), # RGBA
                font=font,
            )
            draw.text(
                (text_xy[0], text_xy[1] + text_line_sep),
                text[i][1], (*text_color, 255), # RGBA
                font=font,
            )
            img_A_with_text = np.array(img_A_with_text)
            img_A = np.round(progress_text*img_A_with_text + (1 - progress_text)*img_A).astype(np.uint8)
            img_A = Image.fromarray(img_A)

            if i < n_method - 1 and j >= t_duration - t_transition:
                # transition period
                j_start = t_transition
                j2 = j + t_transition - t_duration
                img_B = Image.open(join(dirs[i + 1], seq, frame_list[j2 + clip_start]))
                img_B = np.array(img_B)
                img_B = img_B[crop_top:-crop_bottom]

                h, w, _ = img_B.shape
                split_pos = split_anime_accelerate(j2/t_transition, w, line_width)
                split_pos = int(round(split_pos))
                line_start = split_pos - (line_width - 1)//2
                line_end = split_pos + line_width//2            # inclusive

                if split_pos <= 0:
                    img = img_B
                else:
                    img = np.array(img_A)
                    img_B = np.asarray(img_B)
                    img[:, split_pos:] = img_B[:, split_pos:]
                
                if line_start < w and line_end >= 0:
                    # line is visible
                    line_start = max(0, line_start)
                    line_end = min(w, line_end)
                    img[:, line_start:line_end] = line_color_np

                out_img = Image.fromarray(img)
            else:
                out_img = img_A
            out_img.save(out_path)

    if make_video:
        out_path = seq_dir_out + '.mp4'
        if overwrite or not isfile(out_path):
            print('Making the video')
            class Dummy():
                fps
            opts = Dummy()
            opts.fps = fps
            make_video_func((seq_dir_out, opts))
    else:
        print(f'python vis/make_videos_numbered.py "{opts.out_dir}" --fps {opts.fps}')

if __name__ == '__main__':
    main()