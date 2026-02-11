import pathlib
from PIL import Image
from functools import partial

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets.folder import is_image_file
from tqdm.auto import tqdm
import numpy as np
from skimage.color import rgb2gray
import diffusers
from diffusers import StableDiffusionPipeline
from dreamsim import dreamsim
from itertools import combinations

from libs.engine import ModelState
from libs.metric.lpips_origin import LPIPS
from libs.metric.piq.perceptual import DISTS as DISTS_PIQ
from libs.metric.clip_score import CLIPScoreWrapper
from methods.painter.diffsketcher import (
    Painter, SketchPainterOptimizer, Token2AttnMixinASDSPipeline, Token2AttnMixinASDSSDXLPipeline)
from methods.painter.diffsketcher.sketch_utils import (
    log_tensor_img, plt_batch, plt_attn, save_tensor_img, fix_image_scale)
from methods.painter.diffsketcher.mask_utils import get_mask_u2net
from methods.token2attn.attn_control import AttentionStore, EmptyControl
from methods.token2attn.ptp_utils import view_images
from methods.diffusers_warp import init_diffusion_pipeline, model2res
from methods.diffvg_warp import init_diffvg
from methods.painter.diffsketcher.process_svg import remove_low_opacity_paths

from itertools import combinations
from torch.utils.data import DataLoader, TensorDataset
from piqa import SSIM
import glob, os, sys, pdb, time
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from imageio import imread
from PIL import Image, ImageOps
import torchvision.models as models 
import torch.nn as nn
from matplotlib import pyplot as plt
import csv
import pandas as pd
import PIL.Image as pilimg

import numpy as np
from numpy.core.fromnumeric import mean
import torch.utils.data as data
import torchvision.transforms as transforms
import sys

class ChexpertTrainDataset(Dataset):

    def __init__(self,transform = None, indices = None):
        
        csv_path = "/home/hbc/DS/DiffSketcher/train.csv"
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

def mse(imageA, imageB):
    err = torch.mean((imageA - imageB) ** 2)
    return err

def psnr(original, compressed):
    max_pixel = 1.0
    mean_squared_error = mse(original, compressed)
    if mean_squared_error == 0:
        return float('inf')
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mean_squared_error))
    return 1 / psnr_value

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    window = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return window / window.sum()

def ssim(img1, img2, window_size=11, window_sigma=1.5, size_average=True):
    channels = 3
    window_1d = gaussian_window(window_size, window_sigma)
    window = window_1d[:, None] * window_1d[None, :]
    window = window.to(img1.device).expand(channels, 1, window_size, window_size)
    
    if img1.is_cuda:
        window = window.cuda(img1.device)
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channels) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return (ssim_map.mean() + 1) / 2
    else:
        return (ssim_map.mean(1).mean(1).mean(1) + 1) / 2

class DiffSketcherPipeline(ModelState):

    def __init__(self, args):

        attn_log_ = ""
        if args.attention_init:
            attn_log_ = f"-tk{args.token_ind}" \
                        f"{'-XDoG' if args.xdog_intersec else ''}" \
                        f"-atc{args.attn_coeff}-tau{args.softmax_temp}"

        logdir_ = f"sd{args.seed}-im{args.image_size}" \
                  f"-P{args.num_paths}W{args.width}{'OP' if args.optim_opacity else 'BL'}" \
                  f"{attn_log_}"
        super().__init__(args, log_path_suffix=logdir_)

        self.png_logs_dir = self.results_path / "png_logs"
        self.svg_logs_dir = self.results_path / "svg_logs"
        self.attn_logs_dir = self.results_path / "attn_logs"
        if self.accelerator.is_main_process:
            self.png_logs_dir.mkdir(parents=True, exist_ok=True)
            self.svg_logs_dir.mkdir(parents=True, exist_ok=True)
            self.attn_logs_dir.mkdir(parents=True, exist_ok=True)

        self.make_video = self.args.make_video
        if self.make_video:
            self.frame_idx = 0
            self.frame_log_dir = self.results_path / "frame_logs"
            self.frame_log_dir.mkdir(parents=True, exist_ok=True)

        init_diffvg(self.device, True, args.print_timing)

        if args.model_id == "sdxl":
            custom_pipeline = Token2AttnMixinASDSSDXLPipeline
            custom_scheduler = diffusers.DPMSolverMultistepScheduler
            self.args.cross_attn_res = self.args.cross_attn_res * 2
        elif args.model_id == 'sd21' or args.model_id == 'medsd':
            custom_pipeline = Token2AttnMixinASDSPipeline
            custom_scheduler = diffusers.DDIMScheduler
        else:
            custom_pipeline = Token2AttnMixinASDSPipeline
            custom_scheduler = diffusers.DDIMScheduler

        

        self.diffusion = init_diffusion_pipeline(
            self.args.model_id,
            custom_pipeline=custom_pipeline,
            custom_scheduler=custom_scheduler,
            device=self.device,
            local_files_only=not args.download,
            force_download=args.force_download,
            resume_download=args.resume_download,
            ldm_speed_up=args.ldm_speed_up,
            enable_xformers=args.enable_xformers,
            gradient_checkpoint=args.gradient_checkpoint,
        )

        

        self.g_device = torch.Generator(device=self.device).manual_seed(args.seed)

        self.cargs = self.args.clip
        self.clip_score_fn = CLIPScoreWrapper(self.cargs.model_name,
                                              device=self.device,
                                              visual_score=True,
                                              feats_loss_type=self.cargs.feats_loss_type,
                                              feats_loss_weights=self.cargs.feats_loss_weights,
                                              fc_loss_weight=self.cargs.fc_loss_weight)

    def load_render(self, target_img, attention_map, mask=None):
        renderer = Painter(self.args,
                           num_strokes=self.args.num_paths,
                           num_segments=self.args.num_segments,
                           imsize=self.args.image_size,
                           device=self.device,
                           target_im=target_img,
                           attention_map=attention_map,
                           mask=mask)
        return renderer

    def extract_ldm_attn(self, GT, prompts):
        controller = AttentionStore() if self.args.attention_init else EmptyControl()

        height = width = model2res(self.args.model_id)
        outputs = GT

        target_file = self.results_path / "ldm_generated_image.png"
        view_images([np.array(img) for img in outputs.images], save_image=True, fp=target_file)

        if self.args.attention_init:
            cross_attention_maps, tokens = \
                self.diffusion.get_cross_attention([prompts],
                                                   controller,
                                                   res=self.args.cross_attn_res,
                                                   from_where=("up", "down"),
                                                   save_path=self.results_path / "cross_attn.png")

            self.print(f"the length of tokens is {len(tokens)}, select {self.args.token_ind}-th token")
            self.print(f"origin cross_attn_map shape: {cross_attention_maps.shape}")
            cross_attn_map = cross_attention_maps[:, :, self.args.token_ind]
            self.print(f"select cross_attn_map shape: {cross_attn_map.shape}\n")
            cross_attn_map = 255 * cross_attn_map / cross_attn_map.max()
            cross_attn_map = cross_attn_map.unsqueeze(-1).expand(*cross_attn_map.shape, 3)
            cross_attn_map = cross_attn_map.permute(2, 0, 1).unsqueeze(0)
            cross_attn_map = F.interpolate(cross_attn_map, size=self.args.image_size, mode='bicubic')
            cross_attn_map = torch.clamp(cross_attn_map, min=0, max=255)
            cross_attn_map = rgb2gray(cross_attn_map.squeeze(0).permute(1, 2, 0)).astype(np.float32)
            if cross_attn_map.shape[-1] != self.args.image_size and cross_attn_map.shape[-2] != self.args.image_size:
                cross_attn_map = cross_attn_map.reshape(self.args.image_size, self.args.image_size)
            cross_attn_map = (cross_attn_map - cross_attn_map.min()) / (cross_attn_map.max() - cross_attn_map.min())

            self_attention_maps, svd, vh_ = \
                self.diffusion.get_self_attention_comp([prompts],
                                                       controller,
                                                       res=self.args.self_attn_res,
                                                       from_where=("up", "down"),
                                                       img_size=self.args.image_size,
                                                       max_com=self.args.max_com,
                                                       save_path=self.results_path)

            if self.args.mean_comp:
                self_attn = np.mean(vh_, axis=0)
                self.print(f"use the mean of {self.args.max_com} comps.")
            else:
                self_attn = vh_[self.args.comp_idx]
                self.print(f"select {self.args.comp_idx}-th comp.")
            self_attn = (self_attn - self_attn.min()) / (self_attn.max() - self_attn.min())
            self_attn_vis = np.copy(self_attn)
            self_attn_vis = self_attn_vis * 255
            self_attn_vis = np.repeat(np.expand_dims(self_attn_vis, axis=2), 3, axis=2).astype(np.uint8)
            view_images(self_attn_vis, save_image=True, fp=self.results_path / "self-attn-final.png")

            attn_map = self.args.attn_coeff * cross_attn_map + (1 - self.args.attn_coeff) * self_attn
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

            self.print(f"-> fusion attn_map: {attn_map.shape}")
        else:
            attn_map = None

        return target_file.as_posix(), attn_map

    @property
    def clip_norm_(self):
        return transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def clip_pair_augment(self,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          im_res: int,
                          augments: str = "affine_norm",
                          num_aug: int = 4):
        augment_list = []
        if "affine" in augments:
            augment_list.append(
                transforms.RandomPerspective(fill=0, p=1.0, distortion_scale=0.5)
            )
            augment_list.append(
                transforms.RandomResizedCrop(im_res, scale=(0.8, 0.8), ratio=(1.0, 1.0))
            )
        augment_list.append(self.clip_norm_)

        augment_compose = transforms.Compose(augment_list)
        x_augs, y_augs = [self.clip_score_fn.normalize(x)], [self.clip_score_fn.normalize(y)]
        for n in range(num_aug):
            augmented_pair = augment_compose(torch.cat([x, y]))
            x_augs.append(augmented_pair[0].unsqueeze(0))
            y_augs.append(augmented_pair[1].unsqueeze(0))
        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        return xs, ys

    def painterly_rendering(self, num_outputs ,prompt: str, GT, path):
        self.print(f"prompt: {prompt}")
        self.print(f"negative_prompt: {self.args.negative_prompt}\n")
        if self.args.negative_prompt is None:
            self.args.negative_prompt = ""

        self.num_outputs = num_outputs
        self.alpha = 0.05
        self.step = 0

        target_files = []
        attention_maps = []
        for i in range(self.num_outputs):
            self.g_device = torch.Generator(device=self.device).manual_seed(i)
            target_file = GT
            attention_map = GT
            target_files.append(GT)
            attention_maps.append(np.array(GT))
            
        timesteps_ = self.diffusion.scheduler.timesteps.cpu().numpy().tolist()
        self.print(f"{len(timesteps_)} denoising steps, {timesteps_}")

        perceptual_loss_fn = None
        if self.args.perceptual.coeff > 0:
            if self.args.perceptual.name == "lpips":
                lpips_loss_fn = LPIPS(net=self.args.perceptual.lpips_net).to(self.device)
                perceptual_loss_fn = partial(lpips_loss_fn.forward, return_per_layer=False, normalize=False)
            elif self.args.perceptual.name == "dists":
                perceptual_loss_fn = DISTS_PIQ()

        inputs = []
        masks = []
        for i in range(num_outputs):
            input, mask = self.get_target(path,
                                       self.args.image_size,
                                       self.results_path,
                                       self.args.u2net_path,
                                       self.args.mask_object,
                                       self.args.fix_scale,
                                       self.device)
            input = input.detach()
            inputs.append(input)
            masks.append(mask)
        
        self.print("inputs shape: ", inputs[0].shape)

        DreamSim_network, DreamSim_preprocess = dreamsim(pretrained=True)

        renderers = []
        
        
        for i in range(self.num_outputs):
            renderer = Painter(self.args,
                           num_strokes=self.args.num_paths,
                           num_segments=self.args.num_segments,
                           imsize=self.args.image_size,
                           device=self.device,
                           target_im=inputs[i],
                           attention_map=attention_maps[i], 
                           mask=masks[i]) 
            renderers.append(renderer) 


        for i in range(len(renderers)):
            img = renderers[i].init_image(stage=0)

        self.print("init_image shape: ", img.shape)
        log_tensor_img(img, self.results_path, output_prefix="init_sketch")
        optimizer = SketchPainterOptimizer(renderer,
                                           self.args.lr,
                                           self.args.optim_opacity,
                                           self.args.optim_rgba,
                                           self.args.color_lr,
                                           self.args.optim_width,
                                           self.args.width_lr)
        optimizers = []
        for i in range(self.num_outputs):
            optimizers.append(SketchPainterOptimizer(renderers[i],
                                           self.args.lr,
                                           self.args.optim_opacity,
                                           self.args.optim_rgba,
                                           self.args.color_lr,
                                           self.args.optim_width,
                                           self.args.width_lr))
            optimizers[i].init_optimizers()


        self.print(f"-> Painter points Params: {len(renderer.get_points_params())}")
        self.print(f"-> Painter width Params: {len(renderer.get_width_parameters())}")
        self.print(f"-> Painter opacity Params: {len(renderer.get_color_parameters())}")

        best_visual_loss, best_semantic_loss = 100, 100
        best_iter_v, best_iter_s = 0, 0
        min_delta = 1e-6
        similarity_loss = torch.tensor(0.)
        final_raster_sketches = []
        intermmediate_sketches = []        

        self.print(f"\ntotal optimization steps: {self.args.num_iter}")
        with tqdm(initial=self.step, total=self.args.num_iter, disable=not self.accelerator.is_main_process) as pbar:
            while self.step < self.args.num_iter:
                prev_intermmediate_sketches = intermmediate_sketches
                intermmediate_sketches = []
                for s in range(self.num_outputs):

                    
                    raster_sketch = renderers[s].get_image().to(self.device)


                    sds_loss, grad = torch.tensor(0), torch.tensor(0)
                    if self.step >= self.args.sds.warmup:
                        grad_scale = self.args.sds.grad_scale if self.step > self.args.sds.warmup else 0
                        sds_loss, grad = self.diffusion.score_distillation_sampling(
                            raster_sketch,
                            crop_size=self.args.sds.crop_size,
                            augments=self.args.sds.augmentations,
                            prompt=[self.args.prompt],
                            negative_prompt=[self.args.negative_prompt],
                            guidance_scale=self.args.sds.guidance_scale,
                            grad_scale=grad_scale,
                            t_range=list(self.args.sds.t_range),
                        )

                    raster_sketch_aug, inputs_aug = self.clip_pair_augment(
                        raster_sketch, inputs[s],
                        im_res=224,
                        augments=self.cargs.augmentations,
                        num_aug=self.cargs.num_aug
                    )
                    total_visual_loss = torch.tensor(0)
                    l_clip_fc, l_clip_conv, clip_conv_loss_sum = torch.tensor(0), [], torch.tensor(0)
                    if self.args.clip.vis_loss > 0:
                        l_clip_fc, l_clip_conv = self.clip_score_fn.compute_visual_distance(
                            raster_sketch_aug, inputs_aug, clip_norm=False
                        )
                        clip_conv_loss_sum = sum(l_clip_conv)
                        total_visual_loss = self.args.clip.vis_loss * (clip_conv_loss_sum + l_clip_fc)

                    l_percep = torch.tensor(0.)
                    if perceptual_loss_fn is not None:
                        l_perceptual = perceptual_loss_fn(raster_sketch, inputs[s]).mean()
                        l_percep = l_perceptual * self.args.perceptual.coeff

                    l_tvd = torch.tensor(0.)
                    if self.cargs.text_visual_coeff > 0:
                        l_tvd = self.clip_score_fn.compute_text_visual_distance(
                            raster_sketch_aug, self.args.prompt
                        ) * self.cargs.text_visual_coeff

                    optimizers[s].zero_grad_()

                    if self.step > 600:
                        
                        total_dreamsim_loss = torch.tensor(0.).cuda()
                        total_lowlevel_loss = torch.tensor(0.).cuda()
                        count = torch.tensor(0.).cuda()

                        for i in range(len(prev_intermmediate_sketches)):
                            if i != s:

                                sketch1 = prev_intermmediate_sketches[s]
                                sketch2 = prev_intermmediate_sketches[i]

                                
                                
                                dreamsim_distance = DreamSim_network(sketch1, sketch2)
                                
                                total_dreamsim_loss += (1 - dreamsim_distance.mean())
                                total_lowlevel_loss += (1 - torch.mean((sketch1 - sketch2) ** 2))


                                count += 1

                        total_lowlevel_loss /= count
                        similarity_loss = self.alpha * total_dreamsim_loss + (1 - self.alpha) * total_lowlevel_loss
                        (similarity_loss * 0.1).backward(retain_graph=True)

                        

                    loss = sds_loss + l_percep + l_tvd  + total_visual_loss                    
                    loss.backward()
                    optimizers[s].step_()

                    intermmediate_sketches.append(renderers[s].get_image().to(self.device))


                    if self.args.lr_scheduler:
                        optimizers[s].update_lr(self.step, self.args.decay_steps)

                    pbar.set_description(
                        f"lr: {optimizers[s].get_lr():.2f}, "
                        f"l_total: {loss.item():.4f}, "
                        f"l_clip_fc: {l_clip_fc.item():.4f}, "
                        f"l_clip_conv({len(l_clip_conv)}): {clip_conv_loss_sum.item():.4f}, "
                        f"l_tvd: {l_tvd.item():.4f}, "
                        f"l_percep: {l_percep.item():.4f}, "
                        f"l_sm: {similarity_loss.item():.4f}, "
                        f"sds: {grad.item():.4e}"
                    )


                self.step += 1
                pbar.update(1)


                


                        
                        


        renderer.save_svg(self.svg_logs_dir.as_posix(), "final_svg_tmp")
        if self.args.opacity_delta != 0:
            remove_low_opacity_paths(self.svg_logs_dir / "final_svg_tmp.svg",
                                     self.results_path / "final_svg.svg",
                                     self.args.opacity_delta)

        for i in range(len(renderers)):
            final_raster_sketches.append(renderers[i].get_image().to(self.device))

        if self.args.make_video:
            from subprocess import call
            call([
                "ffmpeg",
                "-framerate", f"{self.args.video_frame_rate}",
                "-i", (self.frame_log_dir / "iter%d.png").as_posix(),
                "-vb", "20M",
                (self.results_path / "out.mp4").as_posix()
            ])

        self.close(msg="painterly rendering complete.")

        return final_raster_sketches

    def get_target(self,
                   target_file,
                   image_size,
                   output_dir,
                   u2net_path,
                   mask_object,
                   fix_scale,
                   device):
        if not is_image_file(target_file):
            raise TypeError(f"{target_file} is not image file.")

        target = Image.open(target_file)

        if target.mode == "RGBA":
            new_image = Image.new("RGBA", target.size, "WHITE")
            new_image.paste(target, (0, 0), target)
            target = new_image
        target = target.convert("RGB")

        mask = target
        if mask_object:
            if pathlib.Path(u2net_path).exists():
                masked_im, mask = get_mask_u2net(target, output_dir, u2net_path, device)
                target = masked_im
            else:
                self.print(f"'{u2net_path}' is not exist, disable mask target")

        if fix_scale:
            target = fix_image_scale(target)

        transforms_ = []
        if target.size[0] != target.size[1]:
            transforms_.append(transforms.Resize((image_size, image_size)))
        else:
            transforms_.append(transforms.Resize(image_size))
            transforms_.append(transforms.CenterCrop(image_size))
        transforms_.append(transforms.ToTensor())

        data_transforms = transforms.Compose(transforms_)
        target_ = data_transforms(target).unsqueeze(0).to(self.device)

        return target_, mask
