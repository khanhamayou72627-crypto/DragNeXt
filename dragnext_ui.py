import pdb
import os
import base64
import gradio as gr
from utils.ui_utils import (add_center_points, 
                            remove_center_points, 
                            get_points, 
                            undo_points, 
                            show_cur_points,
                            clear_all, 
                            store_img, 
                            train_lora_interface, 
                            run_dragnext, 
                            save_image_mask_points, 
                            save_image_points, 
                            save_drag_result,
                            save_intermediate_images, 
                            create_video, 
                            draw_target_region, 
                            draw_editable_region, 
                            save_all_data)
LENGTH = 512

def create_markdown_section():
    logo_path = os.path.join(os.path.dirname(__file__), "figure", "logo.png")
    with open(logo_path, "rb") as logo_file:
        logo_base64 = base64.b64encode(logo_file.read()).decode("utf-8")

    gr.HTML(f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 16px; min-height: 96px; margin: 8px 0 20px;">
        <img
        src="data:image/png;base64,{logo_base64}"
        alt="DragNeXt logo"
        style="height: 64px; width: auto;"
        />
        <h1 style="margin: 0; font-size: 48px; font-weight: 800; line-height: 1;">DragNeXt</h1>
    </div>
    """)

def create_base_model_config_ui():
    with gr.Tab("Diffusion Model"):
        with gr.Row():
            local_models_dir = 'local_pretrained_models'
            os.makedirs(local_models_dir, exist_ok=True)
            local_models_choice = [os.path.join(local_models_dir, d) for d in os.listdir(local_models_dir) if os.path.isdir(os.path.join(local_models_dir, d))]
            model_path = gr.Dropdown(value="runwayml/stable-diffusion-v1-5",
                                     label="Diffusion Model Path",
                                     choices=[
                                            "runwayml/stable-diffusion-v1-5",
                                            "stabilityai/stable-diffusion-2-1-base",
                                            "stabilityai/stable-diffusion-xl-base-1.0",
                                             ] + local_models_choice)
            vae_path = gr.Dropdown(value="stabilityai/sd-vae-ft-mse",
                                   label="VAE choice",
                                   choices=["stabilityai/sd-vae-ft-mse", "default"] + local_models_choice)
    return model_path, vae_path

def create_lora_parameters_ui():
    with gr.Tab("LoRA Parameters"):
        with gr.Row():
            lora_step = gr.Number(value=70, 
                                  label="LoRA training steps", 
                                  precision=0)
            lora_lr = gr.Number(value=0.0005, 
                                label="LoRA learning rate")
            lora_batch_size = gr.Number(value=4, 
                                        label="LoRA batch size", 
                                        precision=0)
            lora_rank = gr.Number(value=16, 
                                  label="LoRA rank", 
                                  precision=0)
    return lora_step, lora_lr, lora_batch_size, lora_rank


def create_real_image_editing_ui():
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>📤 Draw Mask</h2>")
            canvas = gr.Image(type="numpy", 
                              tool="sketch", 
                              label="Draw your mask on the image",
                              show_label=True, 
                              height=LENGTH, 
                              width=LENGTH) 
            with gr.Row():               
                add_target_region_button = gr.Button("Add Target Region")
                add_edit_region_button = gr.Button("Add Edit Region")
            with gr.Row():
                train_lora_button = gr.Button("Train LoRA")            
            with gr.Row():
                lora_path = gr.Textbox(value=f"./lora_data/test", 
                                       label="LoRA Path", 
                                       placeholder="Enter path for LoRA data")                 
            with gr.Row():
                lora_status_bar = gr.Textbox(label="LoRA Training Status", 
                                             interactive=False)

        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>✏Click Points</h2>")
            input_image = gr.Image(type="numpy", 
                                   label="Click on the image to mark points", 
                                   show_label=True, 
                                   height=LENGTH, 
                                   width=LENGTH)
            with gr.Row():
                center_button = gr.Button("Enable Center Point")
                disenable_center_button = gr.Button("Disenable Center Point")
                undo_button = gr.Button("Undo Point")
                save_button = gr.Button('Save Current Data')
                save_wo_mask_button = gr.Button('Save without mask')
                save_all_button = gr.Button('Save All Data')
                data_dir = gr.Textbox(value='./dataset/test', 
                                      label="Data Directory", 
                                      placeholder="Enter directory path for mask and points")

        with gr.Column():
            gr.Markdown("<h2 style='text-align: center;'>🖼️ Editing Result</h2>")
            output_image = gr.Image(type="numpy", 
                                    label="View the editing results here",
                                    show_label=True, 
                                    height=LENGTH, 
                                    width=LENGTH)
            with gr.Row():
                run_button = gr.Button("Run")
                clear_all_button = gr.Button("Clear All")
                save_result = gr.Button("Save Result")
                show_points = gr.Button("Show Points")
                result_save_path = gr.Textbox(value='./result/test', 
                                              label="Result Folder", 
                                              placeholder="Enter path to save the results")
    return canvas, train_lora_button, lora_path, lora_status_bar, input_image, center_button, disenable_center_button, undo_button, save_button,  save_wo_mask_button, save_all_button, data_dir, output_image, run_button, clear_all_button, show_points, result_save_path, add_target_region_button, add_edit_region_button, save_result


def create_drag_parameters_ui():
    with gr.Tab("Drag Parameters"):
        with gr.Row():
            latent_lr = gr.Number(value=0.02, label="Learning rate")
            prompt = gr.Textbox(label="Prompt")
            drag_end_step = gr.Number(value=3, 
                                      label="End time step", 
                                      precision=0)
            drag_end_step_1 = gr.Number(value=5, 
                                        label="End time step 1", 
                                        precision=0)
            drag_per_step = gr.Number(value=10, 
                                      label="Point tracking number per each step", 
                                      precision=0)
    return latent_lr, prompt, drag_end_step, drag_end_step_1, drag_per_step


def create_advance_parameters_ui():
    with gr.Tab("Advanced Parameters"):
        with gr.Row():
            r1 = gr.Number(value=4, 
                           label="Motion supervision feature path size", 
                           precision=0)
            r2 = gr.Number(value=12, 
                           label="Point tracking feature patch size", 
                           precision=0)
            drag_distance = gr.Number(value=4, 
                                      label="The distance for motion supervision", 
                                      precision=0)
            feature_idx = gr.Number(value=3, 
                                    label="The index of the features [0,3]", 
                                    precision=0)
            max_drag_per_track = gr.Number(value=3, 
                                           label="Motion supervision times for each point tracking", 
                                           precision=0)

        with gr.Row():
            lam = gr.Number(value=0.2, 
                            label="Lambda", 
                            info="Regularization strength on unmasked areas")
            inversion_strength = gr.Slider(0, 1.0, 
                                           value=0.75, 
                                           label="Inversion strength")
            max_track_no_change = gr.Number(value=10, 
                                            label="Early stop",
                                            info="The maximum number of times points is unchanged.")
    return (r1, r2, drag_distance, feature_idx, max_drag_per_track, lam, inversion_strength, max_track_no_change)

def create_intermediate_save_ui():
    with gr.Tab("Get Intermediate Images"):
        with gr.Row():
            save_intermediates_images = gr.Checkbox(label='Save intermediate images')
            get_mp4 = gr.Button("Get video")
    return save_intermediates_images, get_mp4

def attach_canvas_event(canvas: gr.State, 
                        original_image: gr.State, 
                        selected_points: gr.State, 
                        input_image, mask, 
                        add_target_region_button, 
                        add_edit_region_button, 
                        target_mask):
    #? canvas is just like a placeholder!
    #! Modify to get target mask~
    add_target_region_button.click(
        draw_target_region,
        [canvas],
        [original_image, 
         selected_points, 
         input_image, 
         target_mask])

    add_edit_region_button.click(
        draw_editable_region,
        [canvas, target_mask],
        [original_image, 
         selected_points, 
         input_image, mask])

def attach_input_image_event(input_image, 
                             selected_points, 
                             enable_center_points, 
                             enable_center_points_set, 
                             center_button, 
                             disenable_center_button):  

    center_button.click(
        add_center_points,
        [enable_center_points],
        [enable_center_points])

    disenable_center_button.click(
        remove_center_points,
        [enable_center_points],
        [enable_center_points])

    input_image.select(
        get_points,
        [input_image, 
         selected_points, 
         enable_center_points, 
         enable_center_points_set],
        [input_image])

def attach_undo_button_event(undo_button, 
                             original_image, 
                             selected_points, 
                             mask, input_image, 
                             target_mask, 
                             enable_center_points_set):
    undo_button.click(
        undo_points,
        [original_image, 
         mask, 
         target_mask, 
         enable_center_points_set],
        [input_image, 
         selected_points,
         enable_center_points_set])

def attach_train_lora_button_event(train_lora_button, 
                                   original_image, 
                                   prompt, 
                                   model_path, 
                                   vae_path, 
                                   lora_path, 
                                   lora_step, 
                                   lora_lr, 
                                   lora_batch_size, 
                                   lora_rank, 
                                   lora_status_bar):
    train_lora_button.click(
        train_lora_interface,
        [original_image, 
         prompt, 
         model_path, 
         vae_path, 
         lora_path,
         lora_step, 
         lora_lr, 
         lora_batch_size, 
         lora_rank],
        [lora_status_bar])

def attach_run_button_event(run_button, 
                            original_image, 
                            input_image, 
                            mask, 
                            target_mask, 
                            prompt,
                            selected_points, 
                            inversion_strength, 
                            lam, 
                            latent_lr,
                            model_path, 
                            vae_path, 
                            lora_path, 
                            drag_end_step, 
                            drag_end_step_1, 
                            drag_per_step, 
                            output_image, 
                            r1, 
                            r2, 
                            d, 
                            feature_idx, 
                            new_points,
                            max_drag_per_track, 
                            max_track_no_change, 
                            result_save_path, 
                            save_intermediates_images, 
                            enable_center_points, 
                            enable_center_points_set):
    run_button.click(
        run_dragnext,
        [original_image, 
         input_image, 
         mask,
         target_mask, 
         prompt, 
         selected_points,
         inversion_strength, 
         lam, 
         latent_lr, 
         model_path, 
         vae_path,
         lora_path, 
         drag_end_step, 
         drag_end_step_1, 
         drag_per_step, 
         r1, 
         r2, 
         d,
         max_drag_per_track, 
         max_track_no_change, 
         feature_idx, 
         result_save_path, 
         enable_center_points, 
         enable_center_points_set,
           save_intermediates_images],
        [output_image, new_points])

def attach_show_points_event(show_points, 
                             output_image, 
                             selected_points):
    show_points.click(
        show_cur_points,
        [output_image, selected_points],
        [output_image])

def attach_clear_all_button_event(clear_all_button, 
                                  canvas, 
                                  input_image, 
                                  output_image, 
                                  selected_points, 
                                  original_image, mask):
    clear_all_button.click(
        clear_all,
        [gr.Number(value=LENGTH, visible=False, precision=0)],
        [canvas, 
         input_image, 
         output_image, 
         selected_points, 
         original_image, 
         mask])


def attach_save_button_event(save_button, 
                             save_wo_mask_button, 
                             save_all_button, 
                             mask, 
                             target_mask, 
                             selected_points, 
                             input_image, 
                             save_dir, 
                             enable_center_points, 
                             original_image,
                             prompt,
                             output_image,
                             enable_center_points_set):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_button.click(
        save_image_mask_points,
        inputs=[mask, selected_points, input_image, save_dir],
        outputs=[])

    save_wo_mask_button.click(
        save_image_points,
        inputs=[mask, 
                selected_points, 
                original_image, 
                enable_center_points, 
                save_dir],
        outputs=[])
    
    save_all_button.click(
        save_all_data,
        inputs=[mask, 
                target_mask, 
                selected_points, 
                original_image, 
                input_image, 
                prompt, 
                output_image, 
                enable_center_points_set, 
                save_dir],
        outputs=[])


def attach_save_result_event(save_result, output_image, new_points, result_path):
    """
    Attaches an event to the save button to trigger the save function.
    """
    save_result.click(
        save_drag_result,
        inputs=[output_image, 
                new_points, 
                result_path],
        outputs=[])

def attach_video_event(get_mp4_button, 
                       result_save_path, 
                       data_dir):
    get_mp4_button.click(
        create_video,
        inputs=[result_save_path, data_dir])

def main():
    with gr.Blocks() as demo:
        mask = gr.State(value=None)
        target_mask = gr.State(value=None)
        enable_center_points = gr.State(value=False)
        enable_center_points_set = gr.State([])
        selected_points = gr.State([])
        new_points = gr.State([])
        original_image = gr.State(value=None)
        create_markdown_section()
        intermediate_images = gr.State([])

        canvas, train_lora_button, lora_path, lora_status_bar, input_image, center_button, disenable_center_button, undo_button, save_button, save_wo_mask_button, save_all_button, data_dir, output_image, run_button, clear_all_button, show_points, result_save_path, add_target_region_button, add_edit_region_button, save_result = create_real_image_editing_ui()

        latent_lr, prompt, drag_end_step, drag_end_step_1, drag_per_step = create_drag_parameters_ui()

        model_path, vae_path = create_base_model_config_ui()
        lora_step, lora_lr, lora_batch_size, lora_rank = create_lora_parameters_ui()
        r1, r2, d, feature_idx, max_drag_per_track, lam, inversion_strength, max_track_no_change = create_advance_parameters_ui()
        save_intermediates_images, get_mp4_button = create_intermediate_save_ui()

        attach_canvas_event(canvas, 
                            original_image, 
                            selected_points, 
                            input_image, mask, 
                            add_target_region_button, 
                            add_edit_region_button, 
                            target_mask)
        attach_input_image_event(input_image, 
                                 selected_points, 
                                 enable_center_points, 
                                 enable_center_points_set, 
                                 center_button, 
                                 disenable_center_button)
        attach_undo_button_event(undo_button, 
                                 original_image, 
                                 selected_points, 
                                 mask, 
                                 input_image, 
                                 target_mask, 
                                 enable_center_points_set)
        attach_train_lora_button_event(train_lora_button, 
                                       original_image, 
                                       prompt, 
                                       model_path, 
                                       vae_path, 
                                       lora_path, 
                                       lora_step, 
                                       lora_lr, 
                                       lora_batch_size, 
                                       lora_rank, 
                                       lora_status_bar)
        attach_run_button_event(run_button, 
                                original_image, 
                                input_image, 
                                mask, 
                                target_mask, 
                                prompt, 
                                selected_points, 
                                inversion_strength, 
                                lam, 
                                latent_lr, 
                                model_path, 
                                vae_path, 
                                lora_path, 
                                drag_end_step, 
                                drag_end_step_1,
                                drag_per_step, 
                                output_image, 
                                r1, 
                                r2, 
                                d, 
                                feature_idx, 
                                new_points, 
                                max_drag_per_track,
                                max_track_no_change, 
                                result_save_path, 
                                save_intermediates_images, 
                                enable_center_points, 
                                enable_center_points_set)
        attach_show_points_event(show_points, output_image, new_points)
        attach_clear_all_button_event(clear_all_button, 
                                      canvas, 
                                      input_image, 
                                      output_image, 
                                      selected_points, 
                                      original_image, mask)
        attach_save_button_event(save_button, 
                                 save_wo_mask_button, 
                                 save_all_button, 
                                 mask, target_mask, 
                                 selected_points, 
                                 input_image, 
                                 data_dir,
                                 enable_center_points,
                                 original_image,
                                 prompt,
                                 output_image, 
                                 enable_center_points_set)
        attach_save_result_event(save_result, 
                                 output_image, 
                                 new_points, 
                                 result_save_path)
        attach_video_event(get_mp4_button, result_save_path, data_dir)

    demo.queue().launch(share=True, debug=True, server_port=7861)

if __name__ == '__main__':
    main()
