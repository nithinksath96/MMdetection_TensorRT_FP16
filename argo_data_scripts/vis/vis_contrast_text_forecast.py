'''
ECCV sup: Viz Compare (15s).mp4 
'''

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
out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse1.1/vid/forecasting'))

text = [
    'Without Association & Forecasting (AP 13.0)',
    'With Association & Forecasting (AP 16.7)',
]

dirs = [
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/srt_mrcnn50_nm_ds_s0.5_pkt/val'),
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/pps_mrcnn50_nm_ds_s0.5_kf_fba_iou_lin_pkt/val'),
]

seq = 'b1ca08f1-24b0-3c39-ba4e-d5a92868462c'
fps = 30
vis_scale = 1
overwrite = True
make_video = True

clip_start = 0
# crop_top = 0
# crop_bottom = 0

# # make 16:9 instead of 16:10
crop_top = 120
crop_bottom = 0

t_duration = 900
t_transition = 0
t_text_appear = 30

line_width = 15
line_color = [241, 159, 93]

# font_path = r'C:\Windows\Fonts\ROCK.TTF'
font_path = '/home/mengtial/fonts/ROCK.TTF'

font = ImageFont.truetype(font_path, size=60)
text_region = [324, 55, 1595, 174]
text_region_alpha = 0.46 # assuming color black
text_xy = [960, 115]
text_line_sep = 65
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
    assert t_duration <= len(frame_list)

    n_method = len(text)
    for i in tqdm(range(t_duration)):
        out_path = join(seq_dir_out, '%06d.jpg' % (i + 1))
        if not overwrite and isfile(out_path):
            continue
        for j in range(n_method):
            img = Image.open(join(dirs[j], seq, frame_list[i + clip_start]))
            
            # cropping
            img = np.array(img)
            if crop_top > 0:
                img = img[crop_top:-crop_bottom] if crop_bottom else img[crop_top:]
            h, w, _ = img.shape

            text_offset = h - text_region[3] - text_region[1]  if j else 0

            progress_text = ease_in_out(i / t_text_appear) if i < t_text_appear else 1

            # render text region
            img[text_region[1] + text_offset: text_region[3] + text_offset + 1 , text_region[0]:text_region[2] + 1] = \
                np.round((1 + progress_text*(text_region_alpha - 1))
                *img[text_region[1] + text_offset: text_region[3] + text_offset + 1, text_region[0]:text_region[2] + 1]).astype(np.uint8)
            img_with_text = Image.fromarray(img)

            # using TrueType supported in PIL
            draw = ImageDraw.Draw(img_with_text)
            tw, th = draw.textsize(text[j], font=font)
            draw.text(
                (text_xy[0] - tw/2, text_xy[1] - th/2 + text_offset),
                text[j], (*text_color, 255), # RGBA
                font=font,
            )
            img_with_text = np.array(img_with_text)
            img = np.round(progress_text*img_with_text + (1 - progress_text)*img).astype(np.uint8)
            # img = Image.fromarray(img)

            if j > 0:
                img_B = img
                split_pos = int(round(0.55*1200 - crop_top))
                line_start = split_pos - (line_width - 1)//2
                line_end = split_pos + line_width//2            # inclusive

                if split_pos <= 0:
                    img = img_B
                else:
                    img = img_A
                    # img = np.array(img_A)
                    # img_B = np.asarray(img_B)
                    img[split_pos:] = img_B[split_pos:]
                
                if line_start < h and line_end >= 0:
                    # line is visible
                    line_start = max(0, line_start)
                    line_end = min(h, line_end)
                    img[line_start:line_end] = line_color_np

                # img = Image.fromarray(img)
            img_A = img

        img = Image.fromarray(img)
        img.save(out_path)

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