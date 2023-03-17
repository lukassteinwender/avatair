from diffusers import DiffusionPipeline
import torch

torch.cuda.empty_cache()


pipe = DiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V1.4_Fantasy.ai",
    
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda")

def dummy(images, **kwargs):
    return images, False
pipe.safety_checker = dummy

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image.save("output.jpg")