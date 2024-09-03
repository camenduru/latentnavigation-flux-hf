import gradio as gr
import spaces
from clip_slider_pipeline import CLIPSliderFlux
from diffusers import FluxPipeline, AutoencoderTiny
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers.utils import export_to_gif

def process_controlnet_img(image):
    controlnet_img = np.array(image)
    controlnet_img = cv2.Canny(controlnet_img, 100, 200)
    controlnet_img = HWC3(controlnet_img)
    controlnet_img = Image.fromarray(controlnet_img)

# load pipelines
taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16).to("cuda")
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell",
                                    vae=taef1,
                                    torch_dtype=torch.bfloat16)

pipe.transformer.to(memory_format=torch.channels_last)
pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
#pipe.enable_model_cpu_offload()
clip_slider = CLIPSliderFlux(pipe, device=torch.device("cuda"))


base_model = 'black-forest-labs/FLUX.1-schnell'
controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny-alpha'
# controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
# pipe_controlnet = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
# t5_slider_controlnet = T5SliderFlux(sd_pipe=pipe_controlnet,device=torch.device("cuda"))

def convert_to_centered_scale(num):
    if num <= 0:
        raise ValueError("Input must be a positive integer")
    
    if num % 2 == 0:  # even
        start = -(num // 2 - 1)
        end = num // 2
    else:  # odd
        start = -(num // 2)
        end = num // 2
    
    return tuple(range(start, end + 1))

@spaces.GPU(duration=200)
def generate(concept_1, concept_2, scale, prompt, seed=42, recalc_directions=True, iterations=200, steps=4, interm_steps=9, guidance_scale=3.5,
             x_concept_1="", x_concept_2="", 
             avg_diff_x=None, 
             img2img_type = None, img = None, 
             controlnet_scale= None, ip_adapter_scale=None,
             total_images=[],
             progress=gr.Progress(track_tqdm=True)
             ):
    slider_x = [concept_1, concept_2]
    # check if avg diff for directions need to be re-calculated
    print("slider_x", slider_x)
    print("x_concept_1", x_concept_1, "x_concept_2", x_concept_2)
    #torch.manual_seed(seed)
    
    if not sorted(slider_x) == sorted([x_concept_1, x_concept_2]) or recalc_directions:
        #avg_diff = clip_slider.find_latent_direction(slider_x[0], slider_x[1], num_iterations=iterations).to(torch.float16)
        avg_diff = clip_slider.find_latent_direction(slider_x[0], slider_x[1], num_iterations=iterations)
        x_concept_1, x_concept_2 = slider_x[0], slider_x[1]

    images = []
    high_scale = scale
    low_scale = -1 * scale
    for i in range(interm_steps):
        cur_scale = low_scale + (high_scale - low_scale) * i / (interm_steps - 1)
        image = clip_slider.generate(prompt, 
                                     width=768,
                                     height=768,
                                     #guidance_scale=guidance_scale, 
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
    post_generation_slider_update = gr.update(label=comma_concepts_x, minimum=scale_min, maximum=scale_max, interactive=True)
    avg_diff_x = avg_diff.cpu()
    
    return gr.update(label=comma_concepts_x, interactive=True, value=scale), x_concept_1, x_concept_2, avg_diff_x, export_to_gif(images, "clip.gif", fps=5), canvas, images, images[scale_middle], post_generation_slider_update 

@spaces.GPU
def update_scales(x,prompt,seed, steps, interm_steps, guidance_scale,
                  avg_diff_x, 
                  img2img_type = None, img = None,
                  controlnet_scale= None, ip_adapter_scale=None, total_images=[], progress=gr.Progress(track_tqdm=True)):
    print("Hola", x)
    avg_diff = avg_diff_x.cuda()

    # for spectrum generation
    images = []

    high_scale = x
    low_scale = -1 * x
    
    if img2img_type=="controlnet canny" and img is not None:
        control_img = process_controlnet_img(img)
        image = t5_slider_controlnet.generate(prompt, guidance_scale=guidance_scale, image=control_img, controlnet_conditioning_scale =controlnet_scale, scale=x, seed=seed, num_inference_steps=steps, avg_diff=avg_diff) 
    elif img2img_type=="ip adapter" and img is not None:
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, ip_adapter_image=img, scale=x,seed=seed, num_inference_steps=steps, avg_diff=avg_diff) 
    else:
        for i in range(interm_steps):
            cur_scale = low_scale + (high_scale - low_scale) * i / (steps - 1)
            image = clip_slider.generate(prompt, 
                                         width=768,
                                         height=768,
                                         #guidance_scale=guidance_scale, 
                                         scale=cur_scale,  seed=seed, num_inference_steps=steps, avg_diff=avg_diff) 
            images.append(image)
        canvas = Image.new('RGB', (256*interm_steps, 256))
        for i, im in enumerate(images):
            canvas.paste(im.resize((256,256)), (256 * i, 0))
    
    scale_total = convert_to_centered_scale(interm_steps)
    scale_min = scale_total[0]
    scale_max = scale_total[-1]
    scale_middle = scale_total.index(0)
    post_generation_slider_update = gr.update(minimum=scale_min, maximum=scale_max, visible=True)
    
    return export_to_gif(images, "clip.gif", fps=5), canvas, images, images[scale_middle], post_generation_slider_update

def update_pre_generated_images(slider_value, total_images):
    number_images = len(total_images)
    if(number_images > 0):
        scale_tuple = convert_to_centered_scale(number_images)
        return total_images[scale_tuple.index(slider_value)]
    else:
        return None
    
def reset_recalc_directions():
    return True

css_old = '''
#group {
    position: relative;
    width: 600px; /* Increased width */
    height: 600px; /* Increased height */
    margin-bottom: 20px;
    background-color: white;
}
#x {
    position: absolute;
    bottom: 20px; /* Moved further down */
    left: 30px; /* Adjusted left margin */
    width: 540px; /* Increased width to match the new container size */
}
#y {
    position: absolute;
    bottom: 200px; /* Increased bottom margin to ensure proper spacing from #x */
    left: 20px; /* Adjusted left margin */
    width: 540px; /* Increased width to match the new container size */
    transform: rotate(-90deg);
    transform-origin: left bottom;
}
#image_out {
    position: absolute;
    width: 80%; /* Adjust width as needed */
    right: 10px;
    top: 10px; /* Increased top margin to clear space occupied by #x */
}
'''
intro = """
<div style="display: flex;align-items: center;justify-content: center">
    <img src="https://huggingface.co/spaces/LatentNavigation/latentnavigation-flux/resolve/main/Group 4-16.png" width="100" style="display: inline-block">
    <h1 style="margin-left: 12px;text-align: center;margin-bottom: 7px;display: inline-block">Latent Navigation</h1>
</div>
<div style="display: flex;align-items: center;justify-content: center">
    <h3 style="display: inline-block;margin-left: 10px;margin-top: 6px;font-weight: 500">Exploring CLIP text space with FLUX.1 schnell 🪐</h3>
</div>
<p style="font-size: 0.95rem;margin: 0rem;line-height: 1.2em;margin-top:1em;display: inline-block">
    <a href="https://github.com/linoytsaban/semantic-sliders" target="_blank">code</a>
     | 
    <a href="https://huggingface.co/spaces/LatentNavigation/latentnavigation-flux?duplicate=true" target="_blank" style="
        display: inline-block;
    ">
    <img style="margin-top: -1em;margin-bottom: 0em;position: absolute;" src="https://bit.ly/3CWLGkA" alt="Duplicate Space"></a>
</p>
"""
css='''
#strip, #gif{min-height: 50px}
'''
examples = [["winter", "summer", 1.25, "a dog in the park"]]
image_seq = gr.Image(label="Strip", elem_id="strip")
output_image = gr.Image(label="Gif", elem_id="gif")
post_generation_image = gr.Image(label="Generated Images")
post_generation_slider = gr.Slider(minimum=-2, maximum=2, value=0, step=1, interactive=False)
with gr.Blocks(css=css) as demo:

    gr.HTML(intro)
    
    x_concept_1 = gr.State("")
    x_concept_2 = gr.State("")
    total_images = gr.State([])
    # y_concept_1 = gr.State("")
    # y_concept_2 = gr.State("")

    avg_diff_x = gr.State()
    #avg_diff_y = gr.State()

    recalc_directions = gr.State(False)
    
    #with gr.Tab("text2image"):
    with gr.Row():
        with gr.Column():
            with gr.Row():
                concept_1 = gr.Textbox(label="1st concept to compare", placeholder="winter")
                concept_2 = gr.Textbox(label="2nd concept to compare", placeholder="summer")
            #slider_x = gr.Dropdown(label="Slider concept range", allow_custom_value=True, multiselect=True, max_choices=2)
            #slider_y = gr.Dropdown(label="Slider Y concept range", allow_custom_value=True, multiselect=True, max_choices=2)
            prompt = gr.Textbox(label="Prompt", placeholder="A dog in the park")
            x = gr.Slider(minimum=0, value=1.25, step=0.1, maximum=2.5, label="Strength", info="maximum strength on each direction")
            submit = gr.Button("Generate directions")
            gr.Examples(
                examples=examples,
                inputs=[concept_1, concept_2, x, prompt],
                fn=generate,
                outputs=[x, x_concept_1, x_concept_2, avg_diff_x, output_image, image_seq, total_images, post_generation_image, post_generation_slider],
                cache_examples="lazy"
            )
        with gr.Column():
            with gr.Group(elem_id="group"):
                post_generation_image.render()
                post_generation_slider.render()
              #y = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
            with gr.Row():
                with gr.Column(scale=4, min_width=50):
                    image_seq.render()
                    
                with gr.Column(scale=2, min_width=50):
                    output_image.render()
            # with gr.Row():
            #     generate_butt = gr.Button("generate")
    
    with gr.Accordion(label="advanced options", open=False):
        iterations = gr.Slider(label = "num iterations", minimum=0, value=200, maximum=400)
        steps = gr.Slider(label = "num inference steps", minimum=1, value=4, maximum=10)
        interm_steps = gr.Slider(label = "num of intermediate images", minimum=1, value=5, maximum=65)
        guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=0.1,
                maximum=10.0,
                step=0.1,
                value=5,
            )

        seed  = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
        
       
    # with gr.Tab(label="image2image"):
    #     with gr.Row():
    #         with gr.Column():
    #             image = gr.ImageEditor(type="pil", image_mode="L", crop_size=(512, 512))
    #             slider_x_a = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
    #             slider_y_a = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
    #             img2img_type = gr.Radio(["controlnet canny", "ip adapter"], label="", info="", visible=False, value="controlnet canny")
    #             prompt_a = gr.Textbox(label="Prompt")
    #             submit_a = gr.Button("Submit")
    #         with gr.Column():
    #             with gr.Group(elem_id="group"):
    #               x_a = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="x", interactive=False)
    #               y_a = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
    #               output_image_a = gr.Image(elem_id="image_out")
    #             with gr.Row():
    #                 generate_butt_a = gr.Button("generate")
        
    #     with gr.Accordion(label="advanced options", open=False):
    #         iterations_a = gr.Slider(label = "num iterations", minimum=0, value=200, maximum=300)
    #         steps_a = gr.Slider(label = "num inference steps", minimum=1, value=8, maximum=30)
    #         guidance_scale_a = gr.Slider(
    #                 label="Guidance scale",
    #                 minimum=0.1,
    #                 maximum=10.0,
    #                 step=0.1,
    #                 value=5,
    #             )
    #         controlnet_conditioning_scale = gr.Slider(
    #                 label="controlnet conditioning scale",
    #                 minimum=0.5,
    #                 maximum=5.0,
    #                 step=0.1,
    #                 value=0.7,
    #             )
    #         ip_adapter_scale = gr.Slider(
    #                 label="ip adapter scale",
    #                 minimum=0.5,
    #                 maximum=5.0,
    #                 step=0.1,
    #                 value=0.8,
    #                 visible=False
    #             )
    #         seed_a  = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
        
    # submit.click(fn=generate,
    #                  inputs=[slider_x, slider_y, prompt, seed, iterations, steps, guidance_scale, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y],
    #                  outputs=[x, y, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, output_image])
    submit.click(fn=generate,
                     inputs=[concept_1, concept_2, x, prompt, seed, recalc_directions, iterations, steps, interm_steps, guidance_scale, x_concept_1, x_concept_2, avg_diff_x, total_images],
                     outputs=[x, x_concept_1, x_concept_2, avg_diff_x, output_image, image_seq, total_images, post_generation_image, post_generation_slider])

    iterations.change(fn=reset_recalc_directions, outputs=[recalc_directions])
    seed.change(fn=reset_recalc_directions, outputs=[recalc_directions])
    #x.release(fn=update_scales, inputs=[x, prompt, seed, steps, interm_steps, guidance_scale, avg_diff_x, total_images], outputs=[output_image, image_seq, total_images, post_generation_image, post_generation_slider], trigger_mode='always_last')
    # generate_butt_a.click(fn=update_scales, inputs=[x_a,y_a, prompt_a, seed_a, steps_a, guidance_scale_a, avg_diff_x, avg_diff_y, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale], outputs=[output_image_a])
    # submit_a.click(fn=generate,
    #                  inputs=[slider_x_a, slider_y_a, prompt_a, seed_a, iterations_a, steps_a, guidance_scale_a, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale],
    #                  outputs=[x_a, y_a, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, output_image_a])
    post_generation_slider.change(fn=update_pre_generated_images, inputs=[post_generation_slider, total_images], outputs=[post_generation_image], queue=False, show_progress="hidden", concurrency_limit=None)
        
if __name__ == "__main__":
    demo.launch()