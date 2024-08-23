import gradio as gr
import spaces
from clip_slider_pipeline import T5SliderFlux
from diffusers import FluxPipeline
import torch
import time
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel


def process_controlnet_img(image):
    controlnet_img = np.array(image)
    controlnet_img = cv2.Canny(controlnet_img, 100, 200)
    controlnet_img = HWC3(controlnet_img)
    controlnet_img = Image.fromarray(controlnet_img)

# load pipelines
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", 
                                    torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload()
t5_slider = T5SliderFlux(pipe, device=torch.device("cuda"))

base_model = 'black-forest-labs/FLUX.1-schnell'
controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny-alpha'
# controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
# pipe_controlnet = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
# t5_slider_controlnet = T5SliderFlux(sd_pipe=pipe_controlnet,device=torch.device("cuda"))


@spaces.GPU(duration=200)
def generate(slider_x, slider_y, prompt, seed, iterations, steps, guidance_scale,
             x_concept_1, x_concept_2, y_concept_1, y_concept_2, 
             avg_diff_x, 
             avg_diff_y,correlation,
             img2img_type = None, img = None, 
             controlnet_scale= None, ip_adapter_scale=None,
             
             ):
    
    start_time = time.time()
    # check if avg diff for directions need to be re-calculated
    print("slider_x", slider_x)
    print("x_concept_1", x_concept_1, "x_concept_2", x_concept_2)
    
    if not sorted(slider_x) == sorted([x_concept_1, x_concept_2]):
        avg_diff = t5_slider.find_latent_direction(slider_x[0], slider_x[1], num_iterations=iterations).to(torch.float16)
        x_concept_1, x_concept_2 = slider_x[0], slider_x[1]
    
    
    if not sorted(slider_y) == sorted([y_concept_1, y_concept_2]):
        avg_diff_2nd = t5_slider.find_latent_direction(slider_y[0], slider_y[1], num_iterations=iterations).to(torch.float16)
        y_concept_1, y_concept_2 = slider_y[0], slider_y[1]
    end_time = time.time()
    print(f"direction time: {end_time - start_time:.2f} ms")
    
    start_time = time.time()
    
    if img2img_type=="controlnet canny" and img is not None:
        control_img = process_controlnet_img(img)
        image = t5_slider_controlnet.generate(prompt, correlation_weight_factor=correlation, guidance_scale=guidance_scale, image=control_img, controlnet_conditioning_scale =controlnet_scale, scale=0, scale_2nd=0, seed=seed, num_inference_steps=steps, avg_diff=avg_diff, avg_diff_2nd=avg_diff_2nd)
    elif img2img_type=="ip adapter" and img is not None:
        image = t5_slider.generate(prompt, guidance_scale=guidance_scale, correlation_weight_factor=correlation, ip_adapter_image=img, scale=0, scale_2nd=0, seed=seed, num_inference_steps=steps, avg_diff=avg_diff, avg_diff_2nd=avg_diff_2nd)
    else: # text to image
        image = t5_slider.generate(prompt, guidance_scale=guidance_scale, correlation_weight_factor=correlation, scale=0, scale_2nd=0, seed=seed, num_inference_steps=steps, avg_diff=avg_diff, avg_diff_2nd=avg_diff_2nd)
    
    end_time = time.time()
    print(f"generation time: {end_time - start_time:.2f} ms")
    
    comma_concepts_x = ', '.join(slider_x)
    comma_concepts_y = ', '.join(slider_y)

    avg_diff_x = avg_diff.cpu()
    avg_diff_y = avg_diff_2nd.cpu()
  
    return gr.update(label=comma_concepts_x, interactive=True),gr.update(label=comma_concepts_y, interactive=True), x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, image

@spaces.GPU
def update_scales(x,y,prompt,seed, steps, guidance_scale,
                  avg_diff_x, avg_diff_y,
                  img2img_type = None, img = None,
                  controlnet_scale= None, ip_adapter_scale=None,):
    avg_diff = avg_diff_x.cuda()
    avg_diff_2nd = avg_diff_y.cuda()
    if img2img_type=="controlnet canny" and img is not None:
        control_img = process_controlnet_img(img)
        image = t5_slider_controlnet.generate(prompt, guidance_scale=guidance_scale, image=control_img, controlnet_conditioning_scale =controlnet_scale, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    elif img2img_type=="ip adapter" and img is not None:
        image = t5_slider.generate(prompt, guidance_scale=guidance_scale, ip_adapter_image=img, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    else:     
        image = t5_slider.generate(prompt, guidance_scale=guidance_scale, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    return image



@spaces.GPU
def update_x(x,y,prompt,seed, steps, 
             avg_diff_x, avg_diff_y,
             img2img_type = None,
             img = None):
    avg_diff = avg_diff_x.cuda()
    avg_diff_2nd = avg_diff_y.cuda()
    image = t5_slider.generate(prompt, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    return image

@spaces.GPU
def update_y(x,y,prompt,seed, steps, 
             avg_diff_x, avg_diff_y,
             img2img_type = None,
             img = None):
    avg_diff = avg_diff_x.cuda()
    avg_diff_2nd = avg_diff_y.cuda()
    image = t5_slider.generate(prompt, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    return image



css = '''
#group {
    position: relative;
    width: 420px;
    height: 420px;
    margin-bottom: 20px;
    background-color: white
}
#x {
    position: absolute;
    bottom: 0;
    left: 25px;
    width: 400px;
}
#y {
    position: absolute;
    bottom: 20px;
    left: 67px;
    width: 400px;
    transform: rotate(-90deg);
    transform-origin: left bottom;
}
#image_out{position:absolute; width: 80%; right: 10px; top: 40px}
'''
with gr.Blocks(css=css) as demo:
    
    x_concept_1 = gr.State("")
    x_concept_2 = gr.State("")
    y_concept_1 = gr.State("")
    y_concept_2 = gr.State("")

    avg_diff_x = gr.State()
    avg_diff_y = gr.State()
    
    with gr.Tab("text2image"):
        with gr.Row():
            with gr.Column():
                slider_x = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                slider_y = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                prompt = gr.Textbox(label="Prompt")
                submit = gr.Button("find directions")
            with gr.Column():
                with gr.Group(elem_id="group"):
                  x = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="x", interactive=False)
                  y = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
                  output_image = gr.Image(elem_id="image_out")
                with gr.Row():
                    generate_butt = gr.Button("generate")
        
        with gr.Accordion(label="advanced options", open=False):
            iterations = gr.Slider(label = "num iterations", minimum=0, value=200, maximum=400)
            steps = gr.Slider(label = "num inference steps", minimum=1, value=4, maximum=10)
            guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
            correlation = gr.Slider(
                    label="correlation",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    value=0.6,
                )
            seed  = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
        
       
    with gr.Tab(label="image2image"):
        with gr.Row():
            with gr.Column():
                image = gr.ImageEditor(type="pil", image_mode="L", crop_size=(512, 512))
                slider_x_a = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                slider_y_a = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                img2img_type = gr.Radio(["controlnet canny", "ip adapter"], label="", info="", visible=False, value="controlnet canny")
                prompt_a = gr.Textbox(label="Prompt")
                submit_a = gr.Button("Submit")
            with gr.Column():
                with gr.Group(elem_id="group"):
                  x_a = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="x", interactive=False)
                  y_a = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
                  output_image_a = gr.Image(elem_id="image_out")
                with gr.Row():
                    generate_butt_a = gr.Button("generate")
        
        with gr.Accordion(label="advanced options", open=False):
            iterations_a = gr.Slider(label = "num iterations", minimum=0, value=200, maximum=300)
            steps_a = gr.Slider(label = "num inference steps", minimum=1, value=8, maximum=30)
            guidance_scale_a = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
            controlnet_conditioning_scale = gr.Slider(
                    label="controlnet conditioning scale",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                    value=0.7,
                )
            ip_adapter_scale = gr.Slider(
                    label="ip adapter scale",
                    minimum=0.5,
                    maximum=5.0,
                    step=0.1,
                    value=0.8,
                    visible=False
                )
            seed_a  = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
        
    submit.click(fn=generate,
                     inputs=[slider_x, slider_y, prompt, seed, iterations, steps, guidance_scale, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y,correlation],
                     outputs=[x, y, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, output_image])

    generate_butt.click(fn=update_scales, inputs=[x,y, prompt, seed, steps, guidance_scale, avg_diff_x, avg_diff_y], outputs=[output_image])
    generate_butt_a.click(fn=update_scales, inputs=[x_a,y_a, prompt_a, seed_a, steps_a, guidance_scale_a, avg_diff_x, avg_diff_y, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale], outputs=[output_image_a])
    submit_a.click(fn=generate,
                     inputs=[slider_x_a, slider_y_a, prompt_a, seed_a, iterations_a, steps_a, guidance_scale_a, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, correlation, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale],
                     outputs=[x_a, y_a, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x, avg_diff_y, output_image_a])

        
if __name__ == "__main__":
    demo.launch()