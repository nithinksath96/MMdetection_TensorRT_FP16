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
out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse1.1/vid/pss'))

text = [
    ['Offline Accurate', 'AP 38.6'],
    ['Real-time Accurate', 'AP 6.3'],
    ['Real-time Ours', 'AP 17.6'],
]

dirs = [
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/htc_dconv2_ms_s1.0/val'),
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/srt_htc_dconv2_ms_vm_s1.0/val'),
    join(data_dir, 'Exp/ArgoVerse1.1/visf-th0.5/pps_mrcnn50_vm_ds_s0.75_fba_iou_lin_pkt/val'),
]

seq = '5ab2697b-6e3e-3454-a36a-aba2c6f27818'
fps = 30
vis_scale = 1
overwrite = True
make_video = True

clip_start = 0
crop_top = 0
crop_bottom = 0

# # make 16:9 instead of 16:10
# crop_top = 60
# crop_bottom = 60

t_duration = 150
t_transition = 10
t_text_appear = 20

line_width = 8
line_color = [241, 159, 93]

# font_path = r'C:\Windows\Fonts\ROCK.TTF'
font_path = '/home/mengtial/fonts/ROCK.TTF'

font = ImageFont.truetype(font_path, size=60)
text_region = [50, 55, 615, 225]
text_region_alpha = 0.46 # assuming color black
text_xy = [70, 70]
text_line_sep = 70
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
    for i in tqdm(range(n_method)):
        for j in range(t_duration):
            fidx += 1
            out_path = join(seq_dir_out, '%06d.jpg' % fidx)
            if not overwrite and isfile(out_path):
                continue

            # method 1
            j_start = 0
            text_offset = 0

            img_A = Image.open(join(dirs[0], seq, frame_list[fidx + clip_start]))
            # cropping
            img_A = np.array(img_A)
            if crop_bottom > 0:
                img_A = img_A[crop_top:-crop_bottom] if crop_bottom else img_A[crop_top:]
            h, w, _ = img_A.shape

            progress_text = ease_in_out((j - j_start) / t_text_appear) if i ==0 and j - j_start < t_text_appear else 1

            # render text region
            img_A[text_region[1]:text_region[3] + 1, text_region[0] + text_offset:text_region[2] + text_offset + 1] = \
                np.round((1 + progress_text*(text_region_alpha - 1))
                *img_A[text_region[1]:text_region[3] + 1, text_region[0] + text_offset:text_region[2] + text_offset + 1]).astype(np.uint8)
            img_A_with_text = Image.fromarray(img_A)

            # using TrueType supported in PIL
            draw = ImageDraw.Draw(img_A_with_text)
            draw.text(
                (text_xy[0] + text_offset, text_xy[1]),
                text[0][0], (*text_color, 255), # RGBA
                font=font,
            )
            draw.text(
                (text_xy[0] + text_offset, text_xy[1] + text_line_sep),
                text[0][1], (*text_color, 255), # RGBA
                font=font,
            )
            img_A_with_text = np.array(img_A_with_text)
            img_A = np.round(progress_text*img_A_with_text + (1 - progress_text)*img_A).astype(np.uint8)
            img_A = Image.fromarray(img_A)

            if i >= 1:
                # method 2
                j_start = 0

                img_B = Image.open(join(dirs[1], seq, frame_list[fidx + clip_start]))
                # cropping
                img_B = np.array(img_B)
                if crop_bottom > 0:
                    img_B = img_B[crop_top:-crop_bottom] if crop_bottom else img_B[crop_top:]

                progress_text = ease_in_out((j - j_start) / t_text_appear) if i == 1 and j - j_start < t_text_appear else 1

                # render text region
                text_offset = int(round(img_B.shape[1]/3))

                img_B[text_region[1]:text_region[3] + 1, text_region[0] + text_offset:text_region[2] + text_offset + 1] = \
                    np.round((1 + progress_text*(text_region_alpha - 1))
                    *img_B[text_region[1]:text_region[3] + 1, text_region[0] + text_offset:text_region[2] + text_offset + 1]).astype(np.uint8)
                img_B_with_text = Image.fromarray(img_B)

                # using TrueType supported in PIL
                draw = ImageDraw.Draw(img_B_with_text)
                draw.text(
                    (text_xy[0] + text_offset, text_xy[1]),
                    text[1][0], (*text_color, 255), # RGBA
                    font=font,
                )
                draw.text(
                    (text_xy[0] + text_offset, text_xy[1] + text_line_sep),
                    text[1][1], (*text_color, 255), # RGBA
                    font=font,
                )
                img_B_with_text = np.array(img_B_with_text)
                img_B = np.round(progress_text*img_B_with_text + (1 - progress_text)*img_B).astype(np.uint8)
                img_B = Image.fromarray(img_B)

                # move transition to the beginning
                t = j/t_transition if i == 1 and j < t_transition else 1
                # 
                split_pos = split_anime_accelerate(t, 2/3*w, line_width)
                split_pos += 1/3*w
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

                img_A = Image.fromarray(img)

                if i >= 2:
                    # method 3
                    j_start = 0

                    img_B = Image.open(join(dirs[2], seq, frame_list[fidx + clip_start]))
                    # cropping
                    img_B = np.array(img_B)
                    if crop_bottom > 0:
                        img_B = img_B[crop_top:-crop_bottom] if crop_bottom else img_B[crop_top:]

                    progress_text = ease_in_out((j - j_start) / t_text_appear) if i == 2 and j - j_start < t_text_appear else 1

                    # render text region
                    text_offset = int(round(img_B.shape[1]*2/3))

                    img_B[text_region[1]:text_region[3] + 1, text_region[0] + text_offset:text_region[2] + text_offset + 1] = \
                        np.round((1 + progress_text*(text_region_alpha - 1))
                        *img_B[text_region[1]:text_region[3] + 1, text_region[0] + text_offset:text_region[2] + text_offset + 1]).astype(np.uint8)
                    img_B_with_text = Image.fromarray(img_B)

                    # using TrueType supported in PIL
                    draw = ImageDraw.Draw(img_B_with_text)
                    draw.text(
                        (text_xy[0] + text_offset, text_xy[1]),
                        text[2][0], (*text_color, 255), # RGBA
                        font=font,
                    )
                    draw.text(
                        (text_xy[0] + text_offset, text_xy[1] + text_line_sep),
                        text[2][1], (*text_color, 255), # RGBA
                        font=font,
                    )
                    img_B_with_text = np.array(img_B_with_text)
                    img_B = np.round(progress_text*img_B_with_text + (1 - progress_text)*img_B).astype(np.uint8)
                    img_B = Image.fromarray(img_B)

                    # transition period
                    t = j/t_transition if i == 2 and j < t_transition else 1
                    split_pos = split_anime_accelerate(t, 1/3*w, line_width)
                    split_pos += 2/3*w
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

                    img_A = Image.fromarray(img)

            img_A.save(out_path)

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