import gradio as gr
import spaces

@spaces.GPU
def generate(slider_x, slider_y, prompt):
    comma_concepts_x = ', '.join(slider_x)
    comma_concepts_y = ', '.join(slider_y)
    return gr.update(label=comma_concepts_x, interactive=True),gr.update(label=comma_concepts_y, interactive=True), gr.update()

def update_x(slider_x):
  return gr.update()

def update_y(slider_y):
  return gr.update()
  
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
    with gr.Group(elem_id="group"):
      x = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="x", interactive=False)
      y = gr.Slider(minimum=-10, value=0, maximum=10, elem_id="y", interactive=False)
      output_image = gr.Image(elem_id="image_out")

    slider_x = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
    slider_y = gr.Dropdown(label="Slider X concept range", allow_custom_value=True, multiselect=True, max_choices=2)
    prompt = gr.Textbox(label="Prompt")
    submit = gr.Button("Submit")
    submit.click(fn=generate,
                 inputs=[slider_x, slider_y, prompt],
                 outputs=[x, y, output_image])
    x.change(fn=update_x, inputs=[slider_x], outputs=[output_image])
    y.change(fn=update_y, inputs=[slider_y], outputs=[output_image])
if __name__ == "__main__":
    demo.launch()
