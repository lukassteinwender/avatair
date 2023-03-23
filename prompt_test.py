import gradio as gr
from diffusers import DiffusionPipeline
import torch

def diffusion(slide1, slide2):
    torch.cuda.empty_cache()

    if slide1 == 0: abstr = "A ultra abstract "
    if slide1 == 0.2: abstr = "A very abstract " 
    if slide1 == 0.4: abstr = "A abstract "
    if slide1 == 0.6: abstr = "A realistic "
    if slide1 == 0.8: abstr = "A very realistic "
    if slide1 == 1: abstr = "A ultra realistic "

    if slide2 == 0: age = "10 y.o. "
    if slide2 == 0.2: age = "20 y.o. " 
    if slide2 == 0.4: age = "30 y.o. "
    if slide2 == 0.6: age = "50 y.o. "
    if slide2 == 0.8: age = "60 y.o. "
    if slide2 == 1: age = "70 y.o. "

    prompt = abstr + age +  "woman is looking at the camera with a smile on her face and a grey background, by NHK Animation, digital art, trending on artstation, illustration"
    negative_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    print("Running prompt: " + prompt)

    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V1.4_Fantasy.ai",
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cuda")

    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy
    
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
    image.save("output.jpg")
    return image

demo = gr.Interface(
    fn=diffusion,
    inputs=[gr.Slider(0, 1, step=0.2, value=0.6, label="Abstraction", info="abstract <-> realistic"), gr.Slider(0, 1, step=0.2, value=0.4, label="Age", info="young <-> old")],
    outputs="image"
)

demo.launch()