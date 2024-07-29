import gradio as gr
import spaces
import torch
from clip_slider_pipeline import CLIPSliderXL
from diffusers import StableDiffusionXLPipeline, ControlNetModel, StableDiffusionXLControlNetPipeline, EulerDiscreteScheduler,  AutoencoderKL
import time
import numpy as np
import cv2
from PIL import Image


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def process_controlnet_img(image):
    controlnet_img = np.array(image)
    controlnet_img = cv2.Canny(controlnet_img, 100, 200)
    controlnet_img = HWC3(controlnet_img)
    controlnet_img = Image.fromarray(controlnet_img)

# load pipelines
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash", vae=vae).to("cuda", torch.float16)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
clip_slider = CLIPSliderXL(pipe, device=torch.device("cuda"))

pipe_adapter = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash").to("cuda", torch.float16)
pipe_adapter.scheduler = EulerDiscreteScheduler.from_config(pipe_adapter.scheduler.config)
#pipe_adapter.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
# scale = 0.8
# pipe_adapter.set_ip_adapter_scale(scale)
clip_slider_ip = CLIPSliderXL(sd_pipe=pipe_adapter, device=torch.device("cuda"))

controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-canny-sdxl-1.0", # insert here your choice of controlnet
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
    "sd-community/sdxl-flash",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
clip_slider_controlnet = CLIPSliderXL(sd_pipe=pipe_controlnet,device=torch.device("cuda"))


@spaces.GPU(duration=120)
def generate(slider_x, slider_y, prompt, seed, iterations, steps, guidance_scale,
             x_concept_1, x_concept_2, y_concept_1, y_concept_2, 
             avg_diff_x_1, avg_diff_x_2,
             avg_diff_y_1, avg_diff_y_2,
             img2img_type = None, img = None, 
             controlnet_scale= None, ip_adapter_scale=None):
    
    start_time = time.time()
    # check if avg diff for directions need to be re-calculated
    print("slider_x", slider_x)
    print("x_concept_1", x_concept_1, "x_concept_2", x_concept_2)
    
    if not sorted(slider_x) == sorted([x_concept_1, x_concept_2]):
        avg_diff = clip_slider.find_latent_direction(slider_x[0], slider_x[1], num_iterations=iterations)
        avg_diff_0 = avg_diff[0].to(torch.float16)
        avg_diff_1 = avg_diff[1].to(torch.float16)
        x_concept_1, x_concept_2 = slider_x[0], slider_x[1]
    
    print("avg_diff_0", avg_diff_0.dtype)
    
    if not sorted(slider_y) == sorted([y_concept_1, y_concept_2]):
        avg_diff_2nd = clip_slider.find_latent_direction(slider_y[0], slider_y[1], num_iterations=iterations)
        avg_diff_2nd_0 = avg_diff_2nd[0].to(torch.float16)
        avg_diff_2nd_1 = avg_diff_2nd[1].to(torch.float16)
        y_concept_1, y_concept_2 = slider_y[0], slider_y[1]
    end_time = time.time()
    print(f"direction time: {end_time - start_time:.2f} ms")
    
    start_time = time.time()
    
    if img2img_type=="controlnet canny" and img is not None:
        control_img = process_controlnet_img(img)
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, image=control_img, controlnet_conditioning_scale =controlnet_scale, scale=0, scale_2nd=0, seed=seed, num_inference_steps=steps, avg_diff=(avg_diff_0,avg_diff_1), avg_diff_2nd=(avg_diff_2nd_0,avg_diff_2nd_1))
    elif img2img_type=="ip adapter" and img is not None:
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, ip_adapter_image=img, scale=0, scale_2nd=0, seed=seed, num_inference_steps=steps, avg_diff=(avg_diff_0,avg_diff_1), avg_diff_2nd=(avg_diff_2nd_0,avg_diff_2nd_1))
    else: # text to image
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, scale=0, scale_2nd=0, seed=seed, num_inference_steps=steps, avg_diff=(avg_diff_0,avg_diff_1), avg_diff_2nd=(avg_diff_2nd_0,avg_diff_2nd_1))
    
    end_time = time.time()
    print(f"generation time: {end_time - start_time:.2f} ms")
    
    comma_concepts_x = ', '.join(slider_x)
    comma_concepts_y = ', '.join(slider_y)

    avg_diff_x_1 = avg_diff_0.cpu()
    avg_diff_x_2 = avg_diff_1.cpu()
    avg_diff_y_1 = avg_diff_2nd_0.cpu()
    avg_diff_y_2 = avg_diff_2nd_1.cpu()
  
    return gr.update(label=comma_concepts_x, interactive=True),gr.update(label=comma_concepts_y, interactive=True), x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, image

@spaces.GPU
def update_scales(x,y,prompt,seed, steps, guidance_scale,
                  avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, 
                  img2img_type = None, img = None,
                  controlnet_scale= None, ip_adapter_scale=None):
    avg_diff = (avg_diff_x_1.cuda(), avg_diff_x_2.cuda())
    avg_diff_2nd = (avg_diff_y_1.cuda(), avg_diff_y_2.cuda())
    if img2img_type=="controlnet canny" and img is not None:
        control_img = process_controlnet_img(img)
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, image=control_img, controlnet_conditioning_scale =controlnet_scale, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    elif img2img_type=="ip adapter" and img is not None:
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, ip_adapter_image=img, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    else:     
        image = clip_slider.generate(prompt, guidance_scale=guidance_scale, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    return image

@spaces.GPU
def update_x(x,y,prompt,seed, steps, 
             avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2,
             img2img_type = None,
             img = None):
    avg_diff = (avg_diff_x_1.cuda(), avg_diff_x_2.cuda())
    avg_diff_2nd = (avg_diff_y_1.cuda(), avg_diff_y_2.cuda())
    image = clip_slider.generate(prompt, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    return image

@spaces.GPU
def update_y(x,y,prompt, seed, steps, 
            avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2,
            img2img_type = None,
            img = None):
    avg_diff = (avg_diff_x_1.cuda(), avg_diff_x_2.cuda())
    avg_diff_2nd = (avg_diff_y_1.cuda(), avg_diff_y_2.cuda())
    image = clip_slider.generate(prompt, scale=x, scale_2nd=y, seed=seed, num_inference_steps=steps, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
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

    avg_diff_x_1 = gr.State()
    avg_diff_x_2 = gr.State()
    avg_diff_y_1 = gr.State()
    avg_diff_y_2 = gr.State()
    
    with gr.Tab("text2image"):
        with gr.Row():
            with gr.Column():
                slider_x = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                slider_y = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                prompt = gr.Textbox(label="Prompt")
                submit = gr.Button("find directions")
            with gr.Column():
                with gr.Group(elem_id="group"):
                  x = gr.Slider(minimum=-7, value=0, maximum=7, elem_id="x", interactive=False)
                  y = gr.Slider(minimum=-7, value=0, maximum=7, elem_id="y", interactive=False)
                  output_image = gr.Image(elem_id="image_out")
            generate_butt = gr.Button("generate")
        
        with gr.Accordion(label="advanced options", open=False):
            iterations = gr.Slider(label = "num iterations", minimum=0, value=200, maximum=400)
            steps = gr.Slider(label = "num inference steps", minimum=1, value=8, maximum=30)
            guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
            seed  = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
        
       
    with gr.Tab(label="image2image"):
        with gr.Row():
            with gr.Column():
                image = gr.ImageEditor(type="pil", image_mode="L", crop_size=(512, 512))
                slider_x_a = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                slider_y_a = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
                img2img_type = gr.Radio(["controlnet canny", "ip adapter"], label="", info="")
                prompt_a = gr.Textbox(label="Prompt")
                submit_a = gr.Button("Submit")
            with gr.Column():
                with gr.Group(elem_id="group"):
                  x_a = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="x", interactive=False)
                  y_a = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
                  output_image_a = gr.Image(elem_id="image_out")
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
                )
            seed_a  = gr.Slider(minimum=0, maximum=np.iinfo(np.int32).max, label="Seed", interactive=True, randomize=True)
        
    submit.click(fn=generate,
                     inputs=[slider_x, slider_y, prompt, seed, iterations, steps, guidance_scale, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2],
                     outputs=[x, y, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, output_image])
    
    generate_butt.click(fn=update_scales, inputs=[x,y, prompt, seed, steps, guidance_scale, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2], outputs=[output_image])
    generate_butt_a.click(fn=update_scales, inputs=[x_a,y_a, prompt_a, seed_a, steps_a, guidance_scale_a, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale], outputs=[output_image_a])
    #x.change(fn=update_scales, inputs=[x,y, prompt, seed, steps, guidance_scale, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2], outputs=[output_image])
    #y.change(fn=update_scales, inputs=[x,y, prompt, seed, steps, guidance_scale, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2], outputs=[output_image])
    submit_a.click(fn=generate,
                     inputs=[slider_x_a, slider_y_a, prompt_a, seed_a, iterations_a, steps_a, guidance_scale_a, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale],
                     outputs=[x_a, y_a, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, output_image_a])
    #x_a.change(fn=update_scales, inputs=[x_a,y_a, prompt_a, seed_a, steps_a, guidance_scale_a, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale], outputs=[output_image_a])
    #y_a.change(fn=update_scales, inputs=[x_a,y_a, prompt, seed_a, steps_a, guidance_scale_a, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, img2img_type, image, controlnet_conditioning_scale, ip_adapter_scale], outputs=[output_image_a])

        
if __name__ == "__main__":
    demo.launch()