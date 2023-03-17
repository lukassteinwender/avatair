import gradio as gr
from diffusers import DiffusionPipeline
import torch

def diffusion(prompt):
    torch.cuda.empty_cache()


    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V1.4_Fantasy.ai",
        negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
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