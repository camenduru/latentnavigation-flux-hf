import os
import uuid
import gradio as gr
import spaces
from clip_slider_pipeline import CLIPSliderFlux
from diffusers import FluxPipeline, AutoencoderTiny
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import load_image
from diffusers.utils import export_to_video
import random

# load pipelines
base_model = "black-forest-labs/FLUX.1-schnell"

taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
pipe = FluxPipeline.from_pretrained(base_model,
                                    vae=taef1,
                                    torch_dtype=torch.bfloat16)

pipe.transformer.to(memory_format=torch.channels_last)
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
# pipe.enable_model_cpu_offload()
clip_slider = CLIPSliderFlux(pipe, device=torch.device("cuda"))

MAX_SEED = 2**32-1

def save_images_with_unique_filenames(image_list, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    paths = []
    for image in image_list:
        unique_filename = f"{uuid.uuid4()}.png"
        file_path = os.path.join(save_directory, unique_filename)
        
        image.save(file_path)
        paths.append(file_path)
    
    return paths

def convert_to_centered_scale(num):
    if num % 2 == 0:  # even
        start = -(num // 2 - 1)
        end = num // 2
    else:  # odd
        start = -(num // 2)
        end = num // 2 
    return tuple(range(start, end + 1))

@spaces.GPU(duration=200)
def generate(prompt,
             concept_1,
             concept_2,
             scale,
             randomize_seed=True,
             seed=42,
             recalc_directions=True,
             iterations=200, 
             steps=3, 
             interm_steps=33, 
             guidance_scale=3.5,
             x_concept_1="", x_concept_2="", 
             avg_diff_x=None, 
             total_images=[],
             progress=gr.Progress()
    ):
    slider_x = [concept_2, concept_1]
    # check if avg diff for directions need to be re-calculated
    if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        
    if not sorted(slider_x) == sorted([x_concept_1, x_concept_2]) or recalc_directions:
        progress(0, desc="Calculating directions...")
        avg_diff = clip_slider.find_latent_direction(slider_x[0], slider_x[1], num_iterations=iterations)
        x_concept_1, x_concept_2 = slider_x[0], slider_x[1]

    images = []
    high_scale = scale
    low_scale = -1 * scale
    for i in progress.tqdm(range(interm_steps), desc="Generating images"):
        cur_scale = low_scale + (high_scale - low_scale) * i / (interm_steps - 1)
        image = clip_slider.generate(prompt, 
                                     width=768,
                                     height=768,
                                     guidance_scale=guidance_scale, 
                                     scale=cur_scale,  seed=seed, num_inference_steps=steps, avg_diff=avg_diff) 
        images.append(image)
    canvas = Image.new('RGB', (256*interm_steps, 256))
    for i, im in enumerate(images):
        canvas.paste(im.resize((256,256)), (256 * i, 0))

    comma_concepts_x = f"{slider_x[1]}, {slider_x[0]}"

    scale_total = convert_to_centered_scale(interm_steps)
    scale_min = scale_total[0]
    scale_max = scale_total[-1]
    scale_middle = scale_total.index(0)
    post_generation_slider_update = gr.update(label=comma_concepts_x, value=0, minimum=scale_min, maximum=scale_max, interactive=True)
    avg_diff_x = avg_diff.cpu()
    
    return x_concept_1,x_concept_2, avg_diff_x, export_to_video(images, f"{uuid.uuid4()}.mp4", fps=5), canvas, images, images[scale_middle], post_generation_slider_update, seed

def update_pre_generated_images(slider_value, total_images):
    number_images = len(total_images)
    if(number_images > 0):
        scale_tuple = convert_to_centered_scale(number_images)
        return total_images[scale_tuple.index(slider_value)][0]
    else:
        return None
    
def reset_recalc_directions():
    return True


intro = """
<div style="display: flex;align-items: center;justify-content: center">
    <img src="https://huggingface.co/spaces/LatentNavigation/latentnavigation-flux/resolve/main/Group 4-16.png" width="120" style="display: inline-block">
    <h1 style="margin-left: 12px;text-align: center;margin-bottom: 7px;display: inline-block;font-size:2.25em">Latent Navigation</h1>
</div>
<div style="display: flex;align-items: center;justify-content: center">
    <h3 style="display: inline-block;margin-left: 10px;margin-top: 6px;font-weight: 500">Exploring CLIP text space with FLUX.1 schnell 🪐</h3>
</div>
<p style="font-size: 0.95rem;margin: 0rem;line-height: 1.2em;margin-top:1em;display: inline-block">
    <a href="https://www.ethansmith2000.com/post/traversing-through-clip-space-pca-and-latent-directions" target="_blank">based on & inspired by CLIP directions by Ethan Smith</a>
     |
    <a href="https://github.com/linoytsaban/semantic-sliders" target="_blank">code</a>
     | 
    <a href="https://huggingface.co/spaces/LatentNavigation/latentnavigation-flux?duplicate=true" target="_blank" style="
        display: inline-block;
    ">
    <img style="margin-top: -1em;margin-bottom: 0em;position: absolute;" src="https://bit.ly/3CWLGkA" alt="Duplicate Space"></a>
</p>
"""
css='''
#strip, #video{max-height: 256px; min-height: 80px}
#video .empty{min-height: 80px}
#strip img{object-fit: cover}
.gradio-container{max-width: 960px !important}
'''
examples = [["a dog in the park", "winter", "summer", 1.5], ["a house", "USA suburb", "Europe", 2.5], ["a tomato", "rotten", "super fresh", 2.5]]

with gr.Blocks(css=css) as demo:

    gr.HTML(intro)
    
    x_concept_1 = gr.State("")
    x_concept_2 = gr.State("")
    total_images = gr.Gallery(visible=False)

    avg_diff_x = gr.State()

    recalc_directions = gr.State(False)
    
    with gr.Row():
        with gr.Column():
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", info="Describe what to be steered by the directions", placeholder="A dog in the park")
                with gr.Row():
                    concept_1 = gr.Textbox(label="1st direction to steer", placeholder="winter")
                    concept_2 = gr.Textbox(label="2nd direction to steer", placeholder="summer")
            x = gr.Slider(minimum=0, value=1.75, step=0.1, maximum=4.0, label="Strength", info="maximum strength on each direction (unstable beyond 2.5)")
            submit = gr.Button("Generate directions")
        with gr.Column():
            with gr.Group(elem_id="group"):
                post_generation_image = gr.Image(label="Generated Images", type="filepath", elem_id="interactive")
                post_generation_slider = gr.Slider(minimum=-10, maximum=10, value=0, step=1, label="From 1st to 2nd direction")
    with gr.Row():
        with gr.Column(scale=4):
            image_seq = gr.Image(label="Strip", elem_id="strip", height=80)
        with gr.Column(scale=2, min_width=100):
            output_image = gr.Video(label="Looping video", elem_id="video", loop=True, autoplay=True)
    with gr.Accordion(label="Advanced options", open=False):
        interm_steps = gr.Slider(label = "Num of intermediate images", minimum=3, value=7, maximum=65, step=2)
        with gr.Row():
            iterations = gr.Slider(label = "Num iterations for clip directions", minimum=0, value=200, maximum=400, step=1)
            steps = gr.Slider(label = "Num inference steps", minimum=1, value=3, maximum=4, step=1)
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                value=3.5,
            )
            with gr.Column():
                randomize_seed = gr.Checkbox(True, label="Randomize seed")
                seed = gr.Slider(minimum=0, maximum=MAX_SEED, step=1, label="Seed", interactive=True, randomize=True)

    examples_gradio = gr.Examples(
        examples=examples,
        inputs=[prompt, concept_1, concept_2, x],
        fn=generate,
        outputs=[x_concept_1, x_concept_2, avg_diff_x, output_image, image_seq, total_images, post_generation_image, post_generation_slider, seed],
        cache_examples="lazy"
    )

    submit.click(
        fn=generate,
        inputs=[prompt, concept_1, concept_2, x, randomize_seed, seed, recalc_directions, iterations, steps, interm_steps, guidance_scale, x_concept_1, x_concept_2, avg_diff_x, total_images],
        outputs=[x_concept_1, x_concept_2, avg_diff_x, output_image, image_seq, total_images, post_generation_image, post_generation_slider, seed]
    )
    iterations.change(
        fn=reset_recalc_directions,
        outputs=[recalc_directions]
    )
    seed.change(
        fn=reset_recalc_directions,
        outputs=[recalc_directions]
    )
    post_generation_slider.change(
        fn=update_pre_generated_images,
        inputs=[post_generation_slider, total_images],
        outputs=[post_generation_image],
        queue=False,
        show_progress="hidden",
        concurrency_limit=None
    )
        
if __name__ == "__main__":
    demo.launch()