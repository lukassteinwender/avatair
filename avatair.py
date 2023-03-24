import gradio as gr
from diffusers import DiffusionPipeline
import torch

def diffusion(check):
    torch.cuda.empty_cache()
    
    print(check)

    opt_val = 0.3

    if 0 <= opt_val <= 0.19: abstr = "A ultra abstract "
    if 0.2 <= opt_val <= 0.39: abstr = "A abstract "
    if 0.4 <= opt_val <= 0.59: abstr = "A realistic "
    if 0.6 <= opt_val <= 0.79: abstr = "A very realistic "
    if 0.8 <= opt_val <= 1: abstr = "A ultra realistic "

    if 0 <= opt_val <= 0.19: age = "10 y.o. "
    if 0.2 <= opt_val <= 0.39: age = "20 y.o. " 
    if 0.4 <= opt_val <= 0.59: age = "30 y.o. "
    if 0.6 <= opt_val <= 0.79: age = "50 y.o. "
    if 0.8 <= opt_val <= 1: age = "70 y.o. "

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
    inputs=[gr.Checkbox(label="Avatar gut?"),],
    outputs="image"
)

demo.launch()