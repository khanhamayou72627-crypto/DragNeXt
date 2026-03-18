import math
import torch
import numpy as np
import copy

import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from utils.drag_utils import point_tracking, check_handle_reach_target, interpolate_feature_patch
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from diffusers import DDIMScheduler, AutoencoderKL
from pytorch_lightning import seed_everything
from accelerate import Accelerator
import pdb
from matplotlib import pyplot as plt
import torchvision
pdist = torch.nn.PairwiseDistance(p=2)
import time as time
import cv2

def override_forward(self):
    def forward(
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            return_intermediates: bool = False,
            last_up_block_idx: int = None,
    ):
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None:
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(down_block_res_samples, down_block_additional_residuals):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        all_intermediate_features = [sample]
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size)
            all_intermediate_features.append(sample)
            if last_up_block_idx is not None and i == last_up_block_idx:
                return all_intermediate_features

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if return_intermediates:
            return sample, all_intermediate_features
        else:
            return sample
    return forward


class DragNextDragger:
    def __init__(self, 
                 device, 
                 model_path: str, 
                 prompt: str,
                 full_height: int, 
                 full_width: int,
                 inversion_strength: float,
                 r1: int = 4, 
                 r2: int = 12, 
                 beta: int = 4,
                 drag_end_step: int = 10, 
                 drag_end_step_1: int = 10, 
                 track_per_denoise: int = 10,
                 lam: float = 0.2, 
                 latent_lr: float = 0.01,
                 n_inference_step: int = 50, 
                 guidance_scale: float = 1.0, 
                 feature_idx: int = 3,
                 compare_mode: bool = False,
                 vae_path: str = "default", 
                 lora_path: str = '', 
                 seed: int = 42,
                 max_drag_per_track: int = 10, 
                 drag_loss_threshold: float = 4.0, 
                 once_drag: bool = False,
                 max_track_no_change: int = 10, 
                 enable_center_points: bool = False, 
                 enable_center_points_set: list=None):
        
        print("*** enable center points ***: ", enable_center_points)
        self.enable_center_points = enable_center_points_set
        self.device = device
        self.vae_path = vae_path
        self.lora_path = lora_path
        scheduler = DDIMScheduler(beta_start=0.00085, 
                                  beta_end=0.012,
                                  beta_schedule="scaled_linear", 
                                  clip_sample=False,
                                  set_alpha_to_one=False, 
                                  steps_offset=1)

        is_sdxl = 'xl' in model_path
        self.is_sdxl = is_sdxl
        if is_sdxl:
            self.model = StableDiffusionXLPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)
            self.model.unet.config.addition_embed_type = None
        else:
            self.model = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler).to(self.device)
        self.modify_unet_forward()
        if vae_path != "default":
            self.model.vae = AutoencoderKL.from_pretrained(
                vae_path
            ).to(self.device, self.model.vae.dtype)

        self.set_lora()
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        seed_everything(seed)

        self.prompt = prompt
        self.full_height = full_height
        self.full_width = full_width
        self.sup_res_h = int(0.5 * full_height)
        self.sup_res_w = int(0.5 * full_width)

        self.n_inference_step = n_inference_step
        self.n_actual_inference_step = round(inversion_strength * self.n_inference_step)
        self.guidance_scale = guidance_scale
        self.unet_feature_idx = [feature_idx]

        self.r_1 = r1
        self.r_2 = r2
        self.lam = lam
        self.beta = beta

        self.lr = latent_lr
        self.compare_mode = compare_mode

        self.t2 = drag_end_step
        self.t1 = drag_end_step_1 
        self.track_per_denoise = track_per_denoise
        self.total_drag = int(track_per_denoise * self.t2)
        self.model.scheduler.set_timesteps(self.n_inference_step)

        self.do_drag = True
        self.drag_count = 0
        self.max_drag_per_track = max_drag_per_track

        self.drag_loss_threshold = drag_loss_threshold * ((2 * self.r_1) ** 2)
        self.once_drag = once_drag
        self.no_change_track_num = 0
        self.max_no_change_track_num = max_track_no_change

    def set_lora(self):
        if self.lora_path == "":
            print("applying default parameters")
            self.model.unet.set_default_attn_processor()
        else:
            print("applying lora: " + self.lora_path)
            self.model.unet.load_attn_procs(self.lora_path)

    def modify_unet_forward(self):
        self.model.unet.forward = override_forward(self.model.unet)

    def get_handle_target_points(self, points, enable_center_points_set):
        # points: [(w1, h1), (w2, h2), ...] should be reshaped
        points_rotate = []
        points_move = []
        
        handle_points_move = []
        target_points_move = []

        center_points_rotate = []
        handle_points_rotate = []
        target_points_rotate = []

        for idx, point in enumerate(points):
            if enable_center_points_set[idx]:
                points_rotate.append(point)
            else:
                points_move.append(point)
        
        if len(points_move)>0:
            center_points = None
            for idx, point in enumerate(points_move):
                cur_point = torch.tensor(
                    [point[1] / self.full_height * self.sup_res_h, point[0] / self.full_width * self.sup_res_w])
                cur_point = torch.round(cur_point)
                if idx % 2 == 0:
                    handle_points_move.append(cur_point)
                else:
                    target_points_move.append(cur_point)
        
        if len(points_rotate)>0:

            for idx, point in enumerate(points_rotate):
                cur_point = torch.tensor([point[1] / self.full_height * self.sup_res_h, point[0] / self.full_width * self.sup_res_w])
                cur_point = torch.round(cur_point)
                
                if idx % 3 == 0:
                    center_points_rotate.append(cur_point)
                if idx % 3 == 1:
                    handle_points_rotate.append(cur_point)
                if idx % 3 == 2:
                    target_points_rotate.append(cur_point)          
        

        return [handle_points_move, target_points_move], [handle_points_rotate, target_points_rotate, center_points_rotate]

    def inv_step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            x: torch.FloatTensor,
            verbose=False):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(
            timestep - self.model.scheduler.config.num_train_timesteps // self.model.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.model.scheduler.alphas_cumprod[
            timestep] if timestep >= 0 else self.model.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.model.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_dir = (1 - alpha_prod_t_next) ** 0.5 * model_output
        x_next = alpha_prod_t_next ** 0.5 * pred_x0 + pred_dir
        return x_next, pred_x0

    @torch.no_grad()
    def image2latent(self, image):
        if type(image) is Image:
            image = np.array(image)
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)

        latents = self.model.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.no_grad()
    def get_text_embeddings(self, prompt):
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt")
        
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_embeddings

    def forward_unet_features(self, z, t, encoder_hidden_states):
        unet_output, all_intermediate_features = self.model.unet(
            z,
            t,
            encoder_hidden_states=encoder_hidden_states,
            return_intermediates=True)

        all_return_features = []
        for idx in self.unet_feature_idx:
            feat = all_intermediate_features[idx]
            feat = F.interpolate(feat, (self.sup_res_h, self.sup_res_w), mode='bilinear')
            all_return_features.append(feat)
        return_features = torch.cat(all_return_features, dim=1)

        del all_intermediate_features
        torch.cuda.empty_cache()

        return unet_output, return_features

    @torch.no_grad()
    def invert(
            self,
            image: torch.Tensor,
            prompt,
            return_intermediates=False,):
        
        """
        invert a real image into noise map with determinisc DDIM inversion
        """
        batch_size = image.shape[0]
        if isinstance(prompt, list):
            if batch_size == 1:
                image = image.expand(len(prompt), -1, -1, -1)
        elif isinstance(prompt, str):
            if batch_size > 1:
                prompt = [prompt] * batch_size

        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(prompt)
        else:
            text_embeddings = self.get_text_embeddings(prompt)

        latents = self.image2latent(image)

        if self.guidance_scale > 1.:
            unconditional_embeddings = self.get_text_embeddings([''] * batch_size)
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings], dim=0)

        print("Valid timesteps: ", self.model.scheduler.timesteps)
        latents_list = [latents]
        pred_x0_list = [latents]

        for i, t in enumerate(tqdm(reversed(self.model.scheduler.timesteps), desc="DDIM Inversion")):
            if self.n_actual_inference_step is not None and i >= self.n_actual_inference_step:
                continue

            if self.guidance_scale > 1.:
                model_inputs = torch.cat([latents] * 2)
            else:
                model_inputs = latents

            t_ = self.model.scheduler.timesteps[-(i + 2)]
            noise_pred = self.model.unet(model_inputs, t, encoder_hidden_states=text_embeddings)
            if self.guidance_scale > 1.:
                noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncon + self.guidance_scale * (noise_pred_con - noise_pred_uncon)

            latents, pred_x0 = self.inv_step(noise_pred, t, latents)
            latents_list.append(latents)
            pred_x0_list.append(pred_x0)

        if return_intermediates:
            return latents, latents_list
        return latents

    def get_original_features(self, init_code, text_embeddings):
        timesteps = self.model.scheduler.timesteps
        strat_time_step_idx = self.n_inference_step - self.n_actual_inference_step
        original_step_output = {}
        features = {}
        cur_latents = init_code.detach().clone()
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps[strat_time_step_idx:], desc="Denosing for mask features")):
                if i <= self.t1:
                    model_inputs = cur_latents
                    noise_pred, F0 = self.forward_unet_features(model_inputs, t, encoder_hidden_states=text_embeddings)
                    cur_latents = self.model.scheduler.step(noise_pred, t, model_inputs, return_dict=False)[0]
                    original_step_output[t.item()] = cur_latents.cpu()
                    features[t.item()] = F0.cpu()

        del noise_pred, cur_latents, F0
        torch.cuda.empty_cache()

        return original_step_output, features
    
    def get_noise_features(self, input_latents, t, text_embeddings):
        unet_output, F1 = self.forward_unet_features(input_latents, t, encoder_hidden_states=text_embeddings)
        return unet_output, F1

    def move_mask(self, mask, shift_offset):
        shift_offset = shift_offset.long()

        _, _, height, width = mask.shape
        # mask_shift = torch.zeros_like(mask)
        y_indices, x_indices = torch.where(mask[0, 0] == 1)

        y_indices = y_indices.long()
        x_indices = x_indices.long()

        y_indices_shifted = y_indices + shift_offset[0]
        x_indices_shifted = x_indices + shift_offset[1]

        valid_indices_shifted = (y_indices_shifted >= 0) & (y_indices_shifted < height) & (x_indices_shifted >= 0) & (x_indices_shifted < width)
        
        y_indices_shifted = y_indices_shifted[valid_indices_shifted]
        x_indices_shifted = x_indices_shifted[valid_indices_shifted]


        y_indices = y_indices[valid_indices_shifted]
        x_indices = x_indices[valid_indices_shifted]

        return [y_indices, x_indices], [y_indices_shifted, x_indices_shifted] #, mask_shift, mask_ada, mask_non_overlap, mask_supp

    def rotate_mask(self, target_mask, angle, center=None):
        """
        Rotate the region in target_mask where values are 1 by a given angle.

        Args:
            target_mask (torch.Tensor): Input mask of shape (1, 1, height, width), where values are 0 or 1.
            angle (float): Rotation angle in degrees (counterclockwise).
            center (tuple, optional): Center of rotation (y, x). If None, the center of the mask is used.

        Returns:
            torch.Tensor: Rotated mask of the same shape as target_mask.
        """
        _, _, height, width = target_mask.shape

        y_indices, x_indices = torch.where(target_mask[0, 0] == 1)

        if center is None:
            center = (height // 2, width // 2)

        angle_rad = torch.deg2rad(torch.tensor(angle))

        cos_theta = torch.cos(angle_rad)
        sin_theta = torch.sin(angle_rad)
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]]).cuda()

        coords = torch.stack([y_indices - center[0], x_indices - center[1]], dim=0)
        rotated_coords = torch.matmul(rotation_matrix, coords.float())

        rotated_y_indices = (rotated_coords[0] + center[0]).round().long()
        rotated_x_indices = (rotated_coords[1] + center[1]).round().long()

        valid_indices = (rotated_y_indices >= 0) & (rotated_y_indices < height) & \
                        (rotated_x_indices >= 0) & (rotated_x_indices < width)
        rotated_y_indices = rotated_y_indices[valid_indices]
        rotated_x_indices = rotated_x_indices[valid_indices]
        
        #debug_mask = torch.zeros_like(target_mask)
        #debug_mask[0, 0, rotated_y_indices,rotated_x_indices] = 1
        return  [y_indices, x_indices], [rotated_y_indices, rotated_x_indices]


    def cal_motion_supervision_loss(self, handle_points, target_points, F1, x_prev_updated, original_prev,
                                    interp_mask, interp_target_mask, original_features, original_points, alpha=None):
        drag_loss = 0.0
        for i_ in range(len(handle_points)):
            pi, ti, oi = handle_points[i_], target_points[i_], original_points[i_]
            norm_dis = (ti - pi).norm()
            if norm_dis < 2.:
                continue
  
            di = (ti - pi) / (ti - pi).norm() * min(self.beta, norm_dis) 
            original_features.requires_grad_(True) #?why need grad? 
            f0_patch = original_features[:, :, int(oi[0]) - self.r_1:int(oi[0]) + self.r_1 + 1,
                       int(oi[1]) - self.r_1:int(oi[1]) + self.r_1 + 1].detach()
            f1_patch = interpolate_feature_patch(F1, pi[0] + di[0], pi[1] + di[1], self.r_1)

            try:
                drag_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch)
            except:
                target_size = f1_patch.shape[-2:]
                f0_patch = F.interpolate(f0_patch, size=target_size, mode='bilinear', align_corners=False)
                drag_loss += ((2 * self.r_1) ** 2) * F.l1_loss(f0_patch, f1_patch)



        print(f'Loss from drag: {drag_loss}')
        loss = drag_loss + self.lam * ((x_prev_updated - original_prev)
                                       * (1.0 - interp_mask)).abs().sum()
        print('Loss total=%f' % loss)
        return loss, drag_loss

    def track_step(self, original_feature, original_feature_, F1, F1_, handle_points, handle_points_init):
        if self.compare_mode:
            handle_points = point_tracking(original_feature,
                                           F1, handle_points, handle_points_init, self.r_2)
        else:
            handle_points = point_tracking(original_feature_,
                                           F1_, handle_points, handle_points_init, self.r_2)
        return handle_points

    def compare_tensor_lists(self, lst1, lst2):
        if len(lst1) != len(lst2):
            return False
        return all(torch.equal(t1, t2) for t1, t2 in zip(lst1, lst2))

    def dragnext_step(self, init_code, t, t_, text_embeddings, handle_points, target_points,
                      features, handle_points_init, original_step_output, interp_mask, interp_target_mask):
        
        # features: the concated and intermediate features of the UNet;
        # original_step_output: the estimated noise at some timestamps;
        drag_latents = init_code.clone().detach()
        drag_latents.requires_grad_(True)

        first_drag = True
        need_track = False
        track_num = 0
        cur_drag_per_track = 0
        self.compare_mode = True
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision='fp16'
        )

        optimizer = torch.optim.Adam([drag_latents], lr=self.lr)
        drag_latents, self.model.unet, optimizer = accelerator.prepare(drag_latents, self.model.unet, optimizer)

        #?what is the track_num? 
        while track_num < self.track_per_denoise:
            optimizer.zero_grad()
            # F1 is the intermediate features of the UNet, and upsampled to the original resolution;
            unet_output, F1 = self.forward_unet_features(drag_latents, t, text_embeddings)
            #?what is the x_prev_updated?
            #!unet_output is just the estimated noise! from t to (t-1)!
            x_prev_updated = self.model.scheduler.step(unet_output, t, drag_latents, return_dict=False)[0]
            # (1) estimate the noise at the timestamp t; (2) estimate the latent at t;

            if (need_track or first_drag) and (not self.compare_mode):
                with torch.no_grad():
                    _, F1_ = self.forward_unet_features(x_prev_updated, t_, text_embeddings)

            if first_drag: #?The first iter of track_num!
                first_drag = False
                if self.compare_mode:
                    #!Point tracking
                    handle_points = point_tracking(features[t.item()].cuda(),
                                                   F1, handle_points, handle_points_init, self.r_2)
                else:                 
                    handle_points = point_tracking(features[t_.item()].cuda(),
                                                   F1_, handle_points, handle_points_init, self.r_2)

                print(f'After denoise new handle points: {handle_points}, drag count: {self.drag_count}')

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                self.do_drag = False
                print('Reached the target points')
                break

            if self.no_change_track_num == self.max_no_change_track_num:
                self.do_drag = False
                print('Early stop.')
                break

            del unet_output
            if need_track and (not self.compare_mode):
                del _
            torch.cuda.empty_cache()

            loss, drag_loss = self.cal_motion_supervision_loss(handle_points, target_points, F1, x_prev_updated,
                                                               original_step_output[t.item()].cuda(), 
                                                               interp_mask, interp_target_mask,
                                                               original_features=features[t.item()].cuda(),
                                                               original_points=handle_points_init)

            accelerator.backward(loss)
            optimizer.step()

            cur_drag_per_track += 1
            need_track = (cur_drag_per_track == self.max_drag_per_track) or (drag_loss <= self.drag_loss_threshold) or self.once_drag
            if need_track:            
                track_num += 1
                handle_points_cur = copy.deepcopy(handle_points)
                if self.compare_mode:
                    handle_points = point_tracking(features[t.item()].cuda(),
                                                   F1, handle_points, handle_points_init, self.r_2)
                else:
                    handle_points = point_tracking(features[t_.item()].cuda(),
                                                   F1_, handle_points, handle_points_init, self.r_2)

                if self.compare_tensor_lists(handle_points, handle_points_cur):
                    self.no_change_track_num += 1
                    print(f'{self.no_change_track_num} times handle points no changes.')
                else:
                    self.no_change_track_num = 0

                self.drag_count += 1
                cur_drag_per_track = 0
                print(f'New handle points: {handle_points}, drag count: {self.drag_count}')

        init_code = drag_latents.clone().detach()
        init_code.requires_grad_(False)
        del optimizer, drag_latents
        torch.cuda.empty_cache()
        return init_code, handle_points
    

    def calculate_angle(self, center_points, handle_points, target_points):
        """
        Calculate the angles between vectors:
        - v1 (handle_points - center_points) and v0 (y-axis vector).
        - v2 (target_points - center_points) and v0 (y-axis vector).

        Args:
            center_points (torch.Tensor): Center point (cx, cy).
            handle_points (torch.Tensor): Handle point (hx, hy).
            target_points (torch.Tensor): Target point (tx, ty).

        Returns:
            tuple: Angles in degrees (angle_v1_v0, angle_v2_v0), range [0°, 360°].
        """
        
        
        center_points = torch.tensor([center_points[1], center_points[0]], dtype=torch.float32)
        handle_points = torch.tensor([handle_points[1], handle_points[0]], dtype=torch.float32)
        target_points = torch.tensor([target_points[1], target_points[0]], dtype=torch.float32)

        v1 = handle_points - center_points
        v2 = target_points - center_points
        v0 = torch.tensor([1, 0], dtype=torch.float32).to(v1.device)  # y-axis vector

        dot_product_v1_v0 = torch.dot(v1, v0)
        dot_product_v2_v0 = torch.dot(v2, v0)
        magnitude_v1 = torch.norm(v1)
        magnitude_v2 = torch.norm(v2)
        magnitude_v0 = torch.norm(v0)

        if magnitude_v1 == 0 or magnitude_v2 == 0:
            raise ValueError("One of the vectors has zero magnitude, cannot calculate angle.")

        cos_theta_v1_v0 = dot_product_v1_v0 / (magnitude_v1 * magnitude_v0)
        cos_theta_v2_v0 = dot_product_v2_v0 / (magnitude_v2 * magnitude_v0)

        cos_theta_v1_v0 = torch.clamp(cos_theta_v1_v0, -1.0, 1.0)
        cos_theta_v2_v0 = torch.clamp(cos_theta_v2_v0, -1.0, 1.0)

        cross_product_v1_v0 = v1[0] * v0[1] - v1[1] * v0[0]
        cross_product_v2_v0 = v2[0] * v0[1] - v2[1] * v0[0]

        angle_v1_v0_rad = torch.atan2(cross_product_v1_v0, dot_product_v1_v0)
        angle_v2_v0_rad = torch.atan2(cross_product_v2_v0, dot_product_v2_v0)

        angle_v1_v0_deg = math.degrees(angle_v1_v0_rad)
        angle_v2_v0_deg = math.degrees(angle_v2_v0_rad)

        print("angle_handle: ", angle_v1_v0_deg)
        print("angle_handle: ", angle_v1_v0_deg)
        return angle_v1_v0_deg, angle_v2_v0_deg

    def RecInp(self, init_code, iter, t, t_, text_embeddings,
                      features, original_step_output, interp_mask, target_mask, points_move, points_rotate, contours_sort_move, contours_sort_rotate):
        drag_latents = init_code.clone().detach()
        drag_latents.requires_grad_(True)

        first_drag = True
        need_track = False
        track_num = 0
        cur_drag_per_track = 0
        self.compare_mode = True
        accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='fp16')

        prompt = self.prompt.split(',')
        optimizer = torch.optim.Adam([drag_latents], lr=self.lr)
        drag_latents, self.model.unet, optimizer = accelerator.prepare(drag_latents, self.model.unet, optimizer)

        points_move_handle, points_move_target = points_move[0], points_move[1]
        points_rotate_handle, points_rotate_target, points_rotate_center = points_rotate[0], points_rotate[1], points_rotate[2]

        for num in range(self.track_per_denoise):
            optimizer.zero_grad()
            F0 = features[t.item()].cuda()
            original_prev = original_step_output[t.item()].cuda()
            unet_output, F1 = self.forward_unet_features(drag_latents, t,text_embeddings)
            x_prev_updated = self.model.scheduler.step(unet_output, t, drag_latents, return_dict=False)[0]  
            loss = 0.0
            
            if len(contours_sort_move)>0:
                for i, (contour, han, tar) in enumerate(zip(contours_sort_move, points_move_handle, points_move_target)):
                    target_mask_ = torch.Tensor(contour).unsqueeze(0).unsqueeze(0)
                    offset = torch.Tensor((tar[0] - han[0], tar[1] - han[1]))
                    if iter<self.t2:
                        print(num+1+self.track_per_denoise*iter)
                        offset_ = ((num+1+self.track_per_denoise*iter)/(self.track_per_denoise*self.t2))*offset
                    else:
                        offset_ = offset
                
                    if offset_.long().abs().max()<1:
                        pass
                    else:
                        target_mask_ada, target_mask_shift = self.move_mask(target_mask_, offset_)
                        f0_patch = F0[:, :, target_mask_ada[0], target_mask_ada[1]]
                        f1_patch = F1[:, :, target_mask_shift[0], target_mask_shift[1]]
                        consistency_loss = F.l1_loss(f0_patch, f1_patch, reduction="mean") * 10                              
                        latent_loss = self.lam * ((x_prev_updated - original_prev) * (1.0 - interp_mask)).abs().sum()
                        loss = loss + consistency_loss + latent_loss


            if len(contours_sort_rotate)>0:
                for i, (contour, han, tar, cen) in enumerate(zip(contours_sort_rotate, points_rotate_handle, points_rotate_target, points_rotate_center)):
                    target_mask_ = torch.Tensor(contour).unsqueeze(0).unsqueeze(0).cuda()           
                    center = cen
                    angle_han, angle_tar = self.calculate_angle(cen, han, tar)
                    angle = angle_tar - angle_han

                    if angle < -180:
                        angle = 360 + angle
                    elif angle > 180:
                        angle = angle - 360

                    interval = 2
                    _, _, height, width = target_mask.shape
                    
                    if iter<self.t2:
                        print(num+1+self.track_per_denoise*iter)
                        angle_ = ((num+1+self.track_per_denoise*iter)/(self.track_per_denoise*self.t2))*angle
                        angle_ = round(angle_/interval) * interval
                    else:
                        angle_ = angle
                    
                    print(angle_, "angle is ok!")
                    #pytorch中角度
                    target_mask_shift = torchvision.transforms.functional.rotate(target_mask_, angle_, center=list(torch.flip(center,dims=[0]))) #正角度表示逆时针，负角度表示顺时针
                    F0_mask_shift = torchvision.transforms.functional.rotate(F0*target_mask_, angle_, center=list(torch.flip(center,dims=[0])))
                    y, x = torch.where(target_mask_shift[0, 0] == 1)
                    f0_patch = F0_mask_shift[:, :, y, x]
                    f1_patch = F1[:, :, y, x]
                    consistency_loss = F.l1_loss(f0_patch, f1_patch, reduction="mean") * 10           
                    latent_loss = self.lam * ((x_prev_updated - original_prev) * (1.0 - interp_mask)).abs().sum()
                    loss = loss + consistency_loss + latent_loss 
                
            #pdb.set_trace()
            #print("consistency_loss: ", consistency_loss)
            if torch.is_tensor(loss):
                accelerator.backward(loss)
                optimizer.step()
             
        init_code = drag_latents.clone().detach()
        init_code.requires_grad_(False)
        del optimizer, drag_latents
        torch.cuda.empty_cache()
        return init_code, points_move[0]

    def prepare_mask(self, mask):
        mask = torch.from_numpy(mask).float() / 255.
        mask[mask > 0.0] = 1.0
        mask = rearrange(mask, "h w -> 1 1 h w").cuda()
        mask = F.interpolate(mask, (self.sup_res_h, self.sup_res_w), mode="nearest")
        return mask

    def set_latent_masactrl(self):
        editor = MutualSelfAttentionControl(start_step=0,
                                            start_layer=10,
                                            total_steps=self.n_inference_step,
                                            guidance_scale=self.guidance_scale)
        if self.lora_path == "":
            register_attention_editor_diffusers(self.model, editor, attn_processor='attn_proc')
        else:
            register_attention_editor_diffusers(self.model, editor, attn_processor='lora_attn_proc')

    def get_intermediate_images(self, intermediate_images, intermediate_images_original, intermediate_images_t_idx,
                                valid_timestep, text_embeddings):
        for i in range(len(intermediate_images)-1):
            current_original_code = intermediate_images_original[i].to(self.device)
            current_init_code = intermediate_images[i].to(self.device)

            self.set_latent_masactrl()

            for inter_i, inter_t in enumerate(valid_timestep[intermediate_images_t_idx[i] + 1:]):
                with torch.no_grad():
                    noise_pred_all = self.model.unet(torch.cat([current_original_code, current_init_code]), inter_t,
                                                     encoder_hidden_states=torch.cat(
                                                         [text_embeddings, text_embeddings]))
                    noise_pred = noise_pred_all[1]
                    noise_pred_original = noise_pred_all[0]
                    current_init_code = \
                        self.model.scheduler.step(noise_pred, inter_t, current_init_code, return_dict=False)[0]
                    current_original_code = \
                        self.model.scheduler.step(noise_pred_original, inter_t, current_original_code,
                                                  return_dict=False)[0]
            intermediate_images[i] = self.latent2image(current_init_code, return_type="pt").cpu()
        intermediate_images.pop()
        return intermediate_images

    def drag_next(self,
                  source_image,
                  points,
                  mask, target_mask,
                  return_intermediate_images=False,
                  return_intermediate_features=False
                  ):
        init_code = self.invert(source_image, self.prompt)
        original_init = init_code.detach().clone()
        
        if self.is_sdxl:
            text_embeddings, _, _, _ = self.model.encode_prompt(self.prompt)
            text_embeddings = text_embeddings.detach()
        else:
            text_embeddings = self.get_text_embeddings(self.prompt).detach()

        self.model.text_encoder.to('cpu')
        self.model.vae.encoder.to('cpu')

        timesteps = self.model.scheduler.timesteps
        start_time_step_idx = self.n_inference_step - self.n_actual_inference_step

        handle_points = []
        target_points = []
        points_move, points_rotate = self.get_handle_target_points(points, self.enable_center_points)
        handle_points.extend(points_move[0])
        handle_points.extend(points_rotate[0])
        target_points.extend(points_move[1])
        target_points.extend(points_rotate[1])
        original_step_output, features = self.get_original_features(init_code, text_embeddings)
        handle_points_init = copy.deepcopy(handle_points)

        # #mask preparation
        # mask = self.prepare_mask(mask)
        # interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')
        # target_mask = self.prepare_mask(target_mask)
        # interp_target_mask = F.interpolate(target_mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')

        points_move, points_rotate = self.get_handle_target_points(points, self.enable_center_points)
        original_step_output, features = self.get_original_features(init_code, text_embeddings)
        #handle_points_init = copy.deepcopy(handle_points)
   
        #mask preparation
        mask = self.prepare_mask(mask)
        interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')
        target_mask = self.prepare_mask(target_mask)
        interp_target_mask = F.interpolate(target_mask, (init_code.shape[2], init_code.shape[3]), mode='nearest')

        # region <-> point
        shape = tuple(target_mask.shape[-2:])
        contours = cv2.findContours(target_mask.squeeze().cpu().numpy().astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours_sort_move = []
        contours_sort_rotate = []
        
        for item in points_move[0]:
            handle_point = item.unsqueeze(0).cpu().detach().numpy()
            for contour in contours:
                contour = contour[:, 0, :]
                contour_mask = np.zeros(shape, dtype=np.uint8)
                contour = contour[:, np.newaxis, :] if contour.ndim == 2 else contour
                cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=cv2.FILLED)
                source_region = np.column_stack(np.where(contour_mask)).astype(np.int32)
                indictor = np.all(handle_point == source_region, axis=1).any(axis=0)
                if indictor:
                    contours_sort_move.append(contour_mask)
        
        for item in points_rotate[0]:
            handle_point = item.unsqueeze(0).cpu().detach().numpy()
            for contour in contours:
                contour = contour[:, 0, :]
                contour_mask = np.zeros(shape, dtype=np.uint8)
                contour = contour[:, np.newaxis, :] if contour.ndim == 2 else contour
                cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=cv2.FILLED)
                source_region = np.column_stack(np.where(contour_mask)).astype(np.int32)
                indictor = np.all(handle_point == source_region, axis=1).any(axis=0)
                if indictor:
                    contours_sort_rotate.append(contour_mask)       
        
        try:
            assert len(contours_sort_move)==len(points_move[0]); assert len(contours_sort_rotate)==len(points_rotate[0])
        except:
            pdb.set_trace()

        intermediate_features = [init_code.detach().clone().cpu()] if return_intermediate_features else []
        valid_timestep = timesteps[start_time_step_idx:]
        set_mutual = True
        intermediate_images, intermediate_images_original, intermediate_images_t_idx = [], [], []

        did_drag = False
        #the revision starts from 741;
        #!The drag process;
        for i, t in enumerate(tqdm(valid_timestep, desc="Drag and Denoise")):
            #?self.do_drag may control the elary stop!
            if i < self.t2 and self.do_drag and (self.no_change_track_num != self.max_no_change_track_num):
                t_ = valid_timestep[i + 1]
                init_code, handle_points = self.dragnext_step(init_code, t, t_, text_embeddings, handle_points, target_points, features, handle_points_init,original_step_output, interp_mask, interp_target_mask)
                did_drag = True

            if i< self.t1:
                #?>>>>>This place is for reconstruction and inpaiting; 
                t_ = valid_timestep[i + 1]
                init_code, handle_points = self.RecInp(init_code, i, t, t_, text_embeddings,
                                                    features, original_step_output, interp_mask, target_mask, points_move, points_rotate, contours_sort_move, contours_sort_rotate)                
                did_drag = True
                #?>>>>>This place is for reconstruction and inpaiting; 
            
            else:
                if set_mutual:
                    set_mutual = False
                    self.set_latent_masactrl()
        
            #!The denoising process;
            with torch.no_grad():
                noise_pred_all = self.model.unet(torch.cat([original_init, init_code]), t, #concat the original_init and the init_code, where the original_init is the reference;
                                                 encoder_hidden_states=torch.cat([text_embeddings, text_embeddings]))
                noise_pred = noise_pred_all[1]
                noise_pred_original = noise_pred_all[0]
                init_code = self.model.scheduler.step(noise_pred, t, init_code, return_dict=False)[0]
                original_init = self.model.scheduler.step(noise_pred_original, t, original_init, return_dict=False)[0]

            if did_drag and return_intermediate_images:
                current_init_code = init_code.detach().clone()
                current_original_code = original_init.detach().clone()

                intermediate_images.append(current_init_code.cpu())
                intermediate_images_original.append(current_original_code.cpu())
                intermediate_images_t_idx.append(i)
            
            did_drag = False
            if return_intermediate_features:
                intermediate_features.append(init_code.detach().clone().cpu())

        if return_intermediate_images:
            intermediate_images = self.get_intermediate_images(intermediate_images, intermediate_images_original,
                                                               intermediate_images_t_idx, valid_timestep, text_embeddings)

        image = self.latent2image(init_code, return_type="pt")
        return image, intermediate_features, handle_points, intermediate_images