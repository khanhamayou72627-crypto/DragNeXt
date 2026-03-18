import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple
import math
import cv2
import numpy as np
import gradio as gr
from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

import datetime
import PIL
from PIL import Image
from PIL.ImageOps import exif_transpose
import torch
import torch.nn.functional as F

from diffusers import DDIMScheduler, AutoencoderKL
from pipeline import DragNextDragger

from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from .lora_utils import train_lora
import pdb

# -------------- general UI functionality --------------
def clear_all(length=512):
    return gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           gr.Image.update(value=None, height=length, width=length), \
           [], None, None


def mask_image(image,
               mask,
               color=[255, 0, 0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1 - alpha, 0, out)
    return out


def store_img(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    return image, [], masked_img, mask

def draw_target_region(img, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        masked_img = mask_image(image, 1 - mask, color=[0, 0, 0], alpha=0.3)
        masked_img = mask_image(masked_img, mask, color=[255, 0, 0], alpha=0.3)
    else:
        masked_img = image.copy()
    return image, [], masked_img, mask

def draw_editable_region(img, tar_mask, length=512):
    image, mask = img["image"], np.float32(img["mask"][:, :, 0]) / 255.
    height, width, _ = image.shape
    image = Image.fromarray(image)
    image = exif_transpose(image)
    image = image.resize((length, int(length * height / width)), PIL.Image.BILINEAR)
    mask = cv2.resize(mask, (length, int(length * height / width)), interpolation=cv2.INTER_NEAREST)
    image = np.array(image)

    if mask.sum() > 0:
        mask = np.uint8(mask > 0)
        tar_mask = np.uint8(tar_mask > 0)
        union_mask = np.uint8(mask | tar_mask)
        masked_img = mask_image(image, 1 - union_mask, color=[0, 0, 0], alpha=0.3)
        masked_img = mask_image(masked_img, union_mask, color=[0, 0, 255], alpha=0.3)
        masked_img = mask_image(masked_img, tar_mask, color=[255, 0, 0], alpha=0.3)

    else:
        masked_img = image.copy()
    return image, [], masked_img, mask

def calculate_angle(center_points, handle_points, target_points):
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
    center_points = torch.tensor(center_points, dtype=torch.float32)
    handle_points = torch.tensor(handle_points, dtype=torch.float32)
    target_points = torch.tensor(target_points, dtype=torch.float32)

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

    print(center_points, handle_points, target_points)
    print("handle angle:", angle_v1_v0_deg)
    print("target angle:", angle_v2_v0_deg)
    return angle_v1_v0_deg, angle_v2_v0_deg
    
def get_points(img,
               sel_pix,
               enable_center_points,
               enable_center_points_set,
               evt: gr.SelectData):
    # collect the selected point
    point_ = evt.index
    sel_pix.append(point_)
    enable_center_points_set.append(enable_center_points)
    
    points_move = []
    points_rotate = []

    for idx, point in enumerate(sel_pix):
        if enable_center_points_set[idx]:
            points_rotate.append(point)
        else:
            points_move.append(point)


    if len(points_move)>0:
        points = []
        for idx, point in enumerate(points_move):
            if idx % 2 == 0:
                cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
            else:
                cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
            points.append(tuple(point))
            if len(points) == 2:
                cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
                points = []
    
    if len(points_rotate)>0:
        points = []
        for idx, point in enumerate(points_rotate):
            if idx % 3 == 0:                   
                cv2.circle(img, tuple(point), 10, (0, 255, 0), -1)
                cv2.ellipse(img, tuple(point), (15, 15), 0, 0, 360, (0, 255, 0), 2) #顺时针画圆弧
            if idx % 3 == 1:
                cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
                cv2.line(img, tuple(points[0]), tuple(point), (0, 255, 0), 2)
            if idx % 3 == 2:
                cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
                cv2.line(img, tuple(points[0]), tuple(point), (0, 255, 0), 2)
     
            points.append(tuple(point))
            if len(points) == 3:
                angle_start, angle_end = calculate_angle(points[0], list(points[1]), list(points[2]))
                src = points[1]
                tar = (int((points[1][0]+points[2][0])/2), int((points[1][1]+points[2][1])/2))
                cv2.arrowedLine(img, src, tar, (255, 255, 255), 4, tipLength=0.5)
                angle = angle_end - angle_start

                if angle < -180:
                    angle = 360 + angle
                elif angle > 180:
                    angle = angle - 360

                if angle >0:
                    text = f"Rotation Degree=+{angle:.2f}"
                else:
                    text = f"Rotation Degree={angle:.2f}"
                
                points = []

    return img if isinstance(img, np.ndarray) else np.array(img)

def add_center_points(enable_center_points):
    enable_center_points = True
    return enable_center_points

def remove_center_points(enable_center_points):
    enable_center_points = False
    return enable_center_points

def show_cur_points(img,
                    sel_pix,
                    bgr=False):
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            red = (255, 0, 0) if not bgr else (0, 0, 255)
            cv2.circle(img, tuple(point), 5, red, -1)
        else:
            blue = (0, 0, 255) if not bgr else (255, 0, 0)
            cv2.circle(img, tuple(point), 5, blue, -1)
        points.append(tuple(point))
        if len(points) == 2:
            cv2.arrowedLine(img, points[0], points[1], (255, 255, 255), 4, tipLength=0.5)
            points = []
    return img if isinstance(img, np.ndarray) else np.array(img)

def undo_points(original_image,
                mask, target_mask, enable_center_points_set):
    if mask.sum() > 0 and target_mask.sum() > 0:
        mask = np.uint8(mask > 0)
        target_mask = np.uint8(target_mask > 0)
        union_mask = np.uint8(mask | target_mask)
        masked_img = mask_image(original_image, 1 - union_mask, color=[0, 0, 0], alpha=0.3)
        masked_img = mask_image(masked_img, union_mask, color=[0, 0, 255], alpha=0.3)
        masked_img = mask_image(masked_img, target_mask, color=[255, 0, 0], alpha=0.3)
    else:
        masked_img = original_image.copy()
    return masked_img, [], []


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def train_lora_interface(original_image,
                         prompt,
                         model_path,
                         vae_path,
                         lora_path,
                         lora_step,
                         lora_lr,
                         lora_batch_size,
                         lora_rank,
                         progress=gr.Progress(),
                         use_gradio_progress=True):
    if not os.path.exists(lora_path):
        os.makedirs(lora_path)

    clear_folder(lora_path)

    train_lora(
        original_image,
        prompt,
        model_path,
        vae_path,
        lora_path,
        lora_step,
        lora_lr,
        lora_batch_size,
        lora_rank,
        progress,
        use_gradio_progress)
    return "Training LoRA Done!"


def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def save_images_with_pillow(images, base_filename='image'):
    for index, img in enumerate(images):
        img_pil = Image.fromarray(img)
        folder_path = f'./save'
        filename = os.path.join(folder_path, "{}_{}.png".format(base_filename, index))
        img_pil.save(filename)
        print(f"Saved: {filename}")


def get_original_points(handle_points: List[torch.Tensor],
                        full_h: int,
                        full_w: int,
                        sup_res_w,
                        sup_res_h,
                        ) -> List[torch.Tensor]:
    """
    Convert local handle points and target points back to their original UI coordinates.

    Args:
        sup_res_h: Half original height of the UI canvas.
        sup_res_w: Half original width of the UI canvas.
        handle_points: List of handle points in local coordinates.
        full_h: Original height of the UI canvas.
        full_w: Original width of the UI canvas.

    Returns:
        original_handle_points: List of handle points in original UI coordinates.
    """
    original_handle_points = []

    for cur_point in handle_points:
        original_point = torch.round(
            torch.tensor([cur_point[1] * full_w / sup_res_w, cur_point[0] * full_h / sup_res_h]))
        original_handle_points.append(original_point)

    return original_handle_points


def save_all_data(mask, target_mask, points, original_image, image_with_points, prompt, output_image, enable_center_points_set, output_dir='./saved_data'):
    """
    Saves the mask and points to the specified directory.

    Args:
      mask: The mask data as a numpy array.
      points: The list of points collected from the user interaction.
      image_with_points: The image with points clicked by the user.
      output_dir: The directory where to save the data.
    """
    
    indicator = list(set(enable_center_points_set))
    if len(indicator) == 2:
        folder_name = "hybrid"
    if len(indicator)==1:
        if indicator[0] == True:
            folder_name = "rot"
        if indicator[0] == False:
            folder_name = "trans_deform"     
     
    output_dir_ = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir_, exist_ok=True)

    mask_path = os.path.join(output_dir_, f"editable_region_mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    target_mask_path = os.path.join(output_dir_, f"handle_region_mask.png")
    Image.fromarray(target_mask.astype(np.uint8) * 255).save(target_mask_path)

    points_path = os.path.join(output_dir_, f"points.json")
    with open(points_path, 'w') as f:
        json.dump({'points': points, 'rotation':enable_center_points_set[:len(points)]}, f)
    
    prompts_path = os.path.join(output_dir_, f"user_intention.json")
    with open(prompts_path, 'w') as f:
        json.dump({'user_intention':prompt}, f)

    image_drag_path = os.path.join(output_dir_, "image&drag.jpg")
    Image.fromarray(image_with_points).save(image_drag_path)

    image_path = os.path.join(output_dir_, "image.jpg")
    Image.fromarray(original_image).save(image_path)

    result_path = os.path.join(output_dir_, "result.jpg")
    Image.fromarray(output_image).save(result_path)
    
    return

def save_image_mask_points(mask, points, image_with_points, output_dir='./saved_data'):
    """
    Saves the mask and points to the specified directory.

    Args:
      mask: The mask data as a numpy array.
      points: The list of points collected from the user interaction.
      image_with_points: The image with points clicked by the user.
      output_dir: The directory where to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f"mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    points_path = os.path.join(output_dir, f"points.json")
    with open(points_path, 'w') as f:
        json.dump({'points': points}, f)

    image_with_points_path = os.path.join(output_dir, "image_with_points.jpg")
    Image.fromarray(image_with_points).save(image_with_points_path)

    return
    
def save_image_points(mask, points, original_image,  enable_center_points, output_dir='./saved_data'):
    """
    Saves the mask and points to the specified directory.

    Args:
      mask: The mask data as a numpy array.
      points: The list of points collected from the user interaction.
      image_with_points: The image with points clicked by the user.
      output_dir: The directory where to save the data.
    """
    img = original_image.copy()
    img = mask_image(img, 1 - mask, color=[0, 0, 0], alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f"mask.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)

    points_path = os.path.join(output_dir, f"points.json")
    with open(points_path, 'w') as f:
        json.dump({'points': points}, f)

    points_list = []
    if not enable_center_points:
        for idx, point in enumerate(points):
            if idx % 2 == 0:
                cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
            else:
                cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
            points_list.append(tuple(point))
            if len(points_list) == 2:
                cv2.arrowedLine(img, points_list[0], points_list[1], (255, 255, 255), 4, tipLength=0.5)
                points_list = []
    else:
        cv2.circle(img, tuple(points[0]), 10, (0, 255, 0), -1)
        for idx, point in enumerate(points[1:]):
            if idx % 2 == 0:
                cv2.circle(img, tuple(point), 5, (255, 0, 0), -1)
            else:
                cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
            
            points_list.append(tuple(point))
            # draw an arrow from handle point to target point
            if len(points_list) == 2:
                angle_start, angle_end = calculate_angle(points[0], list(points_list[0]), list(points_list[1]))
                cv2.ellipse(img, tuple(points[0]), (15, 15), 0, 0, 360, (0, 255, 0), 1) #顺时针画圆弧
                src = points_list[0]
                tar = points_list[1]
                #tar = (int((points_list[0][0]+points_list[1][0])/2), int((points_list[0][1]+points_list[1][1])/2))
                cv2.arrowedLine(img, src, tar, (255, 255, 255), 4, tipLength=0.5)
                angle = angle_end - angle_start

                if angle < -180:
                    angle = 360 + angle
                elif angle > 180:
                    angle = angle - 360

                if angle >0:
                    text = f"Rotation Degree=+{angle:.2f}"
                else:
                    text = f"Rotation Degree={angle:.2f}"
                
                text_position = (20, 50)
                #cv2.putText(img, text, tuple(text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                points_list = []
    
    image_with_points_path = os.path.join(output_dir, "image_without_mask.jpg")
    Image.fromarray(img).save(image_with_points_path)
    return

def save_drag_result(output_image, new_points, result_path):
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    result_dir = f'{result_path}'
    os.makedirs(result_dir, exist_ok=True)
    output_image_path = os.path.join(result_dir, 'output_image.png')
    cv2.imwrite(output_image_path, output_image)

    img_with_new_points = show_cur_points(np.ascontiguousarray(output_image), new_points, bgr=True)
    new_points_image_path = os.path.join(result_dir, 'image_with_new_points.png')
    cv2.imwrite(new_points_image_path, img_with_new_points)

    points_path = os.path.join(result_dir, f'new_points.json')
    with open(points_path, 'w') as f:
        json.dump({'points': new_points}, f)


def save_intermediate_images(intermediate_images, result_dir):
    for i in range(len(intermediate_images)):
        intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
        intermediate_images_path = os.path.join(result_dir, f'output_image_{i}.png')
        cv2.imwrite(intermediate_images_path, intermediate_images[i])


def create_video(image_folder, data_folder, fps=2, first_frame_duration=2, last_frame_extra_duration=2):
    """
    Creates an MP4 video from a sequence of images using OpenCV.
    """
    img_folder = Path(image_folder)
    img_num = len(list(img_folder.glob('*.png')))

    data_folder = Path(data_folder)
    original_path = data_folder / 'image_with_points.jpg'
    output_path = img_folder / 'dragging.mp4'
    # Collect all image paths
    img_files = [original_path]

    frame = cv2.imread(str(img_files[0]))
    height, width, layers = frame.shape
    size = (int(width), int(height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 format
    video = cv2.VideoWriter(str(output_path), fourcc, int(fps), size)

    for _ in range(int(fps * first_frame_duration)):
        video.write(frame)

    for i in range(img_num - 2):
        video.write(cv2.imread(str(img_folder / f'output_image_{i}.png')))

    last_frame = cv2.imread(str(img_folder / 'output_image.png'))
    for _ in range(int(fps * last_frame_extra_duration)):
        video.write(last_frame)

    video.release()


def run_dragnext(source_image,
                 image_with_clicks,
                 mask,
                 target_mask, # add target mask
                 prompt,
                 points,
                 inversion_strength,
                 lam,
                 latent_lr,
                 model_path,
                 vae_path,
                 lora_path,
                 drag_end_step,
                 drag_end_step_1,
                 track_per_step,
                 r1,
                 r2,
                 d,
                 max_drag_per_track,
                 max_track_no_change,
                 feature_idx=3,
                 result_save_path='',
                 enable_center_points=False,
                 enable_center_points_set=None,
                 return_intermediate_images=True,
                 drag_loss_threshold=0,
                 save_intermedia=False,
                 compare_mode=False,
                 once_drag=False,
                 ):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    height, width = source_image.shape[:2]
    n_inference_step = 50
    guidance_scale = 1.0
    seed = 42
    dragger = DragNextDragger(device, 
                          model_path, 
                          prompt, 
                          height, 
                          width, 
                          inversion_strength, 
                          r1, r2, d,
                          drag_end_step, 
                          drag_end_step_1, 
                          track_per_step, 
                          lam, latent_lr,
                          n_inference_step, 
                          guidance_scale, 
                          feature_idx, 
                          compare_mode, 
                          vae_path, 
                          lora_path, seed,
                          max_drag_per_track, 
                          drag_loss_threshold, 
                          once_drag, 
                          max_track_no_change, 
                          enable_center_points,
                          enable_center_points_set)
    
    source_image = preprocess_image(source_image, device)
    gen_image, intermediate_features, new_points_handle, intermediate_images = dragger.drag_next(source_image, points, mask, target_mask, return_intermediate_images=return_intermediate_images)
    new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)
    if save_intermedia:
        drag_image = [dragger.latent2image(i.cuda()) for i in intermediate_features]
        save_images_with_pillow(drag_image, base_filename='drag_image')
    gen_image = F.interpolate(gen_image, (height, width), mode='bilinear')
    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)

    new_points = []
    for i in range(len(new_points_handle)):
        new_cur_handle_points = new_points_handle[i].numpy().tolist()
        new_cur_handle_points = [int(point) for point in new_cur_handle_points]
        new_points.append(new_cur_handle_points)
        new_points.append(points[i * 2 + 1])

    print(f'points {points}')
    print(f'new points {new_points}')

    if return_intermediate_images:
        os.makedirs(result_save_path, exist_ok=True)
        for i in range(len(intermediate_images)):
            intermediate_images[i] = F.interpolate(intermediate_images[i], (height, width), mode='bilinear')
            intermediate_images[i] = intermediate_images[i].cpu().permute(0, 2, 3, 1).numpy()[0]
            intermediate_images[i] = (intermediate_images[i] * 255).astype(np.uint8)

        for i in range(len(intermediate_images)):
            intermediate_images[i] = cv2.cvtColor(intermediate_images[i], cv2.COLOR_RGB2BGR)
            intermediate_images_path = os.path.join(result_save_path, f'output_image_{i}.png')
            cv2.imwrite(intermediate_images_path, intermediate_images[i])

    return out_image, new_points
