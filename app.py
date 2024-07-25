import gradio as gr
import spaces
import torch
from clip_slider_pipeline import CLIPSliderXL
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler,  AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
flash_pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash", vae=vae).to("cuda", torch.float16)
flash_pipe.scheduler = EulerDiscreteScheduler.from_config(flash_pipe.scheduler.config)
clip_slider = CLIPSliderXL(flash_pipe, device=torch.device("cuda"), iterations=50)

@spaces.GPU
def generate(slider_x, slider_y, prompt, 
             x_concept_1, x_concept_2, y_concept_1, y_concept_2, 
             avg_diff_x_1, avg_diff_x_2,
             avg_diff_y_1, avg_diff_y_2):

    # check if avg diff for directions need to be re-calculated
    if not sorted(slider_x) == sorted([x_concept_1, x_concept_2]):
        avg_diff = clip_slider.find_latent_direction(slider_x[0], slider_x[1])
        x_concept_1, x_concept_2 = slider_x[0], slider_x[1]

    if not sorted(slider_y) == sorted([y_concept_1, y_concept_2]):
        avg_diff_2nd = clip_slider.find_latent_direction(slider_y[0], slider_y[1])
        y_concept_1, y_concept_2 = slider_y[0], slider_y[1]
    
    image = clip_slider.generate(prompt, scale=0, scale_2nd=0, num_inference_steps=8, avg_diff=avg_diff, avg_diff_2nd=avg_diff_2nd)
    comma_concepts_x = ', '.join(slider_x)
    comma_concepts_y = ', '.join(slider_y)

    avg_diff_x_1 = avg_diff[0].cpu()
    avg_diff_x_2 = avg_diff[1].cpu()
    avg_diff_y_1 = avg_diff_2nd[0].cpu()
    avg_diff_y_2 = avg_diff_2nd[1].cpu()
  
    return gr.update(label=comma_concepts_x, interactive=True),gr.update(label=comma_concepts_y, interactive=True), x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, image

@spaces.GPU
def update_x(x,y,prompt, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2):
    avg_diff = [avg_diff_x_1.cuda(), avg_diff_x_2.cuda()] 
    avg_diff_2nd = [avg_diff_y_1.cuda(), avg_diff_y_2.cuda()]
    image = clip_slider.generate(prompt, scale=x, scale_2nd=y, num_inference_steps=8, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
    return image

@spaces.GPU
def update_y(x,y,prompt, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2):
    avg_diff = [avg_diff_x_1.cuda(), avg_diff_x_2.cuda()] 
    avg_diff_2nd = [avg_diff_y_1.cuda(), avg_diff_y_2.cuda()]
    image = clip_slider.generate(prompt, scale=x, scale_2nd=y, num_inference_steps=8, avg_diff=avg_diff,avg_diff_2nd=avg_diff_2nd) 
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
    
    with gr.Row():
        with gr.Column():
            slider_x = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
            slider_y = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
            prompt = gr.Textbox(label="Prompt")
            submit = gr.Button("Submit")
        with gr.Group(elem_id="group"):
          x = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="x", interactive=False)
          y = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
          output_image = gr.Image(elem_id="image_out")
    
    submit.click(fn=generate,
                 inputs=[slider_x, slider_y, prompt, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2],
                 outputs=[x, y, x_concept_1, x_concept_2, y_concept_1, y_concept_2, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2, output_image])
    x.change(fn=update_x, inputs=[x,y, prompt, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2], outputs=[output_image])
    y.change(fn=update_y, inputs=[x,y, prompt, avg_diff_x_1, avg_diff_x_2, avg_diff_y_1, avg_diff_y_2], outputs=[output_image])

if __name__ == "__main__":
    demo.launch()