import os
import sys
import argparse
from datetime import datetime
import random
from typing import Any, List
from functools import partial

from accelerate.utils import set_seed
import omegaconf
from torchvision.utils import save_image
from datetime import datetime

import sysconfig

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.engine import merge_and_update_config
from libs.utils.argparse import accelerate_parser, base_data_parser

import sys
sys.path.append("/content/CLIPasso/diffvg/build/lib.linux-x86_64-3.7")
import PIL.Image as pilimg
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.utils.data

class ChexpertTrainDataset(Dataset):

    def __init__(self,transform = None, indices = None):
        
        csv_path = "C:/Users/hb/Desktop/Data/CheXpert-v1.0-small/labels(former)/selected_train.csv"
        self.dir = "C:/Users/hb/Desktop/Data/"
        self.transform = transform

        self.all_data = pd.read_csv(csv_path)
        self.selecte_data = self.all_data
        self.class_num = 10
        self.all_classes = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Fracture']
        
        self.total_ds_cnt = self.get_total_cnt()
        self.total_ds_cnt = np.array(self.total_ds_cnt)
        self.imbalance = 0
        difference_cnt = self.total_ds_cnt - self.total_ds_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] * difference_cnt[i]        
        for i in range(len(difference_cnt)):
            difference_cnt[i] = difference_cnt[i] / difference_cnt.sum()
        difference_cnt -= difference_cnt.mean()
        for i in range(len(difference_cnt)):
            difference_cnt[i] = (difference_cnt[i] * difference_cnt[i])
    
        self.imbalance = 1 / difference_cnt.sum()

    def __getitem__(self, index):

        row = self.selecte_data.iloc[index, :]
        img = pilimg.open(self.dir + row['Path'])
        label = torch.FloatTensor(row[2:])
        gray_img = self.transform(img)
        return torch.cat([gray_img,gray_img,gray_img], dim = 0), label

    def __len__(self):
        return len(self.selecte_data)

    def get_total_cnt(self):
        total_ds_cnt = [0] * self.class_num
        for i in range(len(self.selecte_data)):
            row = self.selecte_data.iloc[i, 5:]
            for j in range(len(row)):
                total_ds_cnt[j] += int(row[j])
        return total_ds_cnt

    def get_ds_cnt(self):

        raw_pos_freq = self.total_ds_cnt
        raw_neg_freq = self.total_ds_cnt.sum() - self.total_ds_cnt

        return raw_pos_freq, raw_neg_freq

    def get_name(self):
        return 'CheXpert'

    def get_class_cnt(self):
        return 10


def render_batch_wrap(args: omegaconf.DictConfig,
                      seed_range: List,
                      pipeline: Any,
                      **pipe_args):
    start_time = datetime.now()
    for idx, seed in enumerate(seed_range):
        args.seed = seed
        print(f"\n-> [{idx}/{len(seed_range)}], "
              f"current seed: {seed}, "
              f"current time: {datetime.now() - start_time}\n")
        pipe = pipeline(args)
        pipe.painterly_rendering(10, **pipe_args)


def main(args, seed_range):
    args.batch_size = 1

    render_batch_fn = partial(render_batch_wrap, args=args, seed_range=seed_range)
    now = datetime.now()
    result_dir = 'output_sketches/{}_{}_{}'.format(now.date(),str(now.hour), str(now.minute))
    os.makedirs(result_dir, exist_ok=True)

    if args.task == "diffsketcher":
        from pipelines.painter.diffsketcher_pipeline import DiffSketcherPipeline
        
        if not args.render_batch:
            pipe = DiffSketcherPipeline(args)
            final_raster_sketches = pipe.painterly_rendering(args.num_outputs, args.prompt)
            for i in range(len(final_raster_sketches)):
                save_image(final_raster_sketches[i], result_dir + '/output{}.png'.format(i))

        else:
            render_batch_fn(pipeline=DiffSketcherPipeline, prompt=args.prompt)

    elif args.task == "style-diffsketcher":
        from pipelines.painter.diffsketcher_stylized_pipeline import StylizedDiffSketcherPipeline

        if not args.render_batch:
            pipe = StylizedDiffSketcherPipeline(args)
            pipe.painterly_rendering(args.prompt, args.style_file)
        else:
            render_batch_fn(pipeline=StylizedDiffSketcherPipeline, prompt=args.prompt, style_fpath=args.style_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="vary style and content painterly rendering",
        parents=[accelerate_parser(), base_data_parser()]
    )
    parser.add_argument("-tk", "--task",
                        default="diffsketcher", type=str,
                        choices=['diffsketcher', 'style-diffsketcher'],
                        help="choose a method.")
    parser.add_argument("-n", "--num_outputs", default=4)
    parser.add_argument("-m", "--model_id", default="medsd", type=str)
    parser.add_argument("-c", "--config",
                        required=False, type=str,
                        default="diffsketcher.yaml",
                        help="YAML/YML file for configuration.")
    parser.add_argument("-style", "--style_file",
                        default="", type=str,
                        help="the path of style img place.")
    parser.add_argument("-pt", "--prompt", default="A horse is drinking water by the lake", type=str)
    parser.add_argument("-npt", "--negative_prompt", default="", type=str)
    parser.add_argument("--print_timing", "-timing", action="store_true",
                        help="set print svg rendering timing.")
    parser.add_argument("--download", action="store_true",
                        help="download models from huggingface automatically.")
    parser.add_argument("--force_download", "-download", action="store_true",
                        help="force the models to be downloaded from huggingface.")
    parser.add_argument("--resume_download", "-dpm_resume", action="store_true",
                        help="download the models again from the breakpoint.")
    parser.add_argument("--render_batch", "-rdbz", default=False , action="store_true")
    parser.add_argument("-srange", "--seed_range",
                        required=False, nargs='+',
                        help="Sampling quantity.")
    parser.add_argument("-mv", "--make_video", action="store_true",
                        help="make a video of the rendering process.")
    parser.add_argument("-frame_freq", "--video_frame_freq",
                        default=1, type=int,
                        help="video frame control.")
    parser.add_argument("-framerate", "--video_frame_rate",
                        default=36, type=int,
                        help="by adjusting the frame rate, you can control the playback speed of the output video.")

    args = parser.parse_args()

    seed_range = None
    if args.render_batch:
        start_, end_ = 1, 1000000
        if args.seed_range is not None:
            seed_range_ = list(args.seed_range)
            assert len(seed_range_) == 2 and int(seed_range_[1]) > int(seed_range_[0])
            start_, end_ = int(seed_range_[0]), int(seed_range_[1])
            seed_range = [i for i in range(start_, end_)]
        else:
            numbers = list(range(start_, end_))
            seed_range = random.sample(numbers, k=1000)

    args = merge_and_update_config(args)

    set_seed(args.seed)

    main(args, seed_range)
