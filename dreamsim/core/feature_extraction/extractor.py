import types
from typing import List, Tuple
import math
import torch
import torch.nn.modules.utils as nn_utils
from torch import nn
import os
from .load_clip_as_dino import load_clip_as_dino
from .load_open_clip_as_dino import load_open_clip_as_dino
from .vision_transformer import DINOHead
from .load_mae_as_vit import load_mae_as_vit


class ViTExtractor:

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, load_dir: str = "./models",
                 device: str = 'cuda'):
        self.model_type = model_type
        self.device = device
        self.model = ViTExtractor.create_model(model_type, load_dir)
        if type(self.model) is tuple:
            self.proj = self.model[1]
            self.model = self.model[0]
        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride).eval().to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p) is tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride
        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str, load_dir: str = "./models") -> nn.Module:
        if 'dino' in model_type:
            torch.hub.set_dir(load_dir)
            model = torch.hub.load('facebookresearch/dino:main', model_type)
            if model_type == 'dino_vitb16':
                sd = torch.load(os.path.join(load_dir, 'dino_vitb16_pretrain.pth'), map_location='cpu')
                proj = DINOHead(768, 2048)
                proj.mlp[0].weight.data = sd['student']['module.head.mlp.0.weight']
                proj.mlp[0].bias.data = sd['student']['module.head.mlp.0.bias']
                proj.mlp[2].weight.data = sd['student']['module.head.mlp.2.weight']
                proj.mlp[2].bias.data = sd['student']['module.head.mlp.2.bias']
                proj.mlp[4].weight.data = sd['student']['module.head.mlp.4.weight']
                proj.mlp[4].bias.data = sd['student']['module.head.mlp.4.bias']
                proj.last_layer.weight.data = sd['student']['module.head.last_layer.weight']
                model = (model, proj)
        elif 'open_clip' in model_type:
            if model_type == 'open_clip_vitb16':
                model = load_open_clip_as_dino(16, load_dir)
            elif model_type == 'open_clip_vitb32':
                model = load_open_clip_as_dino(32, load_dir)
            elif model_type == 'open_clip_vitl14':
                model = load_open_clip_as_dino(14, load_dir, l14=True)
            else:
                raise ValueError(f"Model {model_type} not supported")
        elif 'clip' in model_type:
            if model_type == 'clip_vitb16':
                model = load_clip_as_dino(16, load_dir)
            elif model_type == 'clip_vitb32':
                model = load_clip_as_dino(32, load_dir)
            elif model_type == 'clip_vitl14':
                model = load_clip_as_dino(14, load_dir, l14=True)
            else:
                raise ValueError(f"Model {model_type} not supported")
        elif 'mae' in model_type:
            model = load_mae_as_vit(model_type, load_dir)
        else:
            raise ValueError(f"Model {model_type} not supported")
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        patch_size = model.patch_embed.patch_size
        if stride == patch_size or (stride,stride) == patch_size:
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        model.patch_embed.proj.stride = stride
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def forward(self, x, is_proj=False):
        if is_proj:
            if 'clip' in self.model_type:
                return self.model(x) @ self.proj.to(self.device)
            elif 'dino' in self.model_type:
                if self.model_type == 'dino_vitb16':
                    self.proj = self.proj.to(self.device)
                    return self.proj(self.model(x))
                raise NotImplementedError
            elif 'mae' in self.model_type:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            return self.model(x)

    def _get_drop_hook(self, drop_rate):

        def dt_pre_hook(module, tkns):
            bp_ = torch.ones_like(tkns[0][0, :, 0])
            bp_[1:] = drop_rate
            tkns = tkns[0][:, torch.bernoulli(bp_) > 0.5, :]
            return tkns

        return dt_pre_hook

    def fix_random_seeds(suffix):
        seed = hash(suffix) % (2 ** 31 - 1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    def _get_hook(self):
        def _hook(model, input, output):
            self._feats.append(output)
        return _hook

    def _register_hooks(self, layers: List[int],drop_rate=0) -> None:
        if drop_rate > 0:
            self.hook_handlers.append(self.model.blocks[0].register_forward_pre_hook(self._get_drop_hook(drop_rate)))
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                self.hook_handlers.append(block.register_forward_hook(self._get_hook()))

    def _unregister_hooks(self) -> None:
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, drop_rate=0) -> List[torch.Tensor]:
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, drop_rate)
        try:
            _ = self.model(batch)
            self._unregister_hooks()
        except Exception as e:
            self._unregister_hooks()
            raise e
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, drop_rate=0) -> torch.Tensor:
        if type(layer) is not list:
            layer = [ layer ]

        self._extract_features(batch, layer, drop_rate)
        x = torch.stack(self._feats, dim=1)
        x = x.unsqueeze(dim=2)
        desc = x.permute(0, 1, 3, 4, 2).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=2)
        return desc
