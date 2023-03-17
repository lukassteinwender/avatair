import gradio as gr
from diffusers import DiffusionPipeline
import torch

def diffusion(prompt):
    torch.cuda.empty_cache()


    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V1.4_Fantasy.ai",
    
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cuda")

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy
    
    image = pipe(prompt).images[0]
    image.save("output.jpg")
    return image

demo = gr.Interface(
    fn=diffusion,
    inputs=gr.Textbox(lines=2, placeholder="type prompt here"),
    outputs="image",
)

demo.launch()