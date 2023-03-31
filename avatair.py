import gradio as gr
import torch
from diffusers import DiffusionPipeline
import torch
import threading
import warnings
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

val = 1.00

def objective(x):
    x == 1.00
    while x==0.00 or x==1.00: 
        x = torch.rand_like(x)
    global val
    val = x
    return x

# define the function to query the human for feedback on each value
def query_human(x):

    # wait for user in gradio-UI to generate next one
    event.wait()
    event.clear()
    
    # TODO input scale from 0-6 how "good" the avatar is
    response = 'n'
    if response.lower() == 'y':
        return 1
    else:
        return 0

# Main BO-Script
def bayesian_opti():
    
    # set the seed for reproducibility
    torch.manual_seed(123)

    # set the range of values to search over
    bounds = torch.tensor([[0.0], [1.0]])

    # randomly sample some initial data points
    train_x = torch.rand(2, 1)
    train_y = torch.tensor([objective(x) for x in train_x]).unsqueeze(-1)

    # define the model
    model = SingleTaskGP(train_x, train_y)

    # loop until the human is satisfied with the value
    while True:
        # define the acquisition function
        acq_func = UpperConfidenceBound(model, beta=2.0)

        # optimize the acquisition function to get the next point to evaluate
        x_next, y_opt = optimize_acqf(acq_func, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

        # evaluate the objective function at the next point
        y_next = objective(x_next)

        # query the human for feedback
        is_good = query_human(x_next.item())

        # add the new data point to the model
        train_x = torch.cat([train_x, x_next])
        train_y = torch.cat([train_y, y_next], dim=0)
        model = SingleTaskGP(train_x, train_y)

        # check if the human is satisfied with the value
        if is_good:
            print(f"The optimal value is {x_next.item():.2f}.")
            break

# Photo generation on user-interaction
def diffusion(check):
    
    # empty cuda-cache
    torch.cuda.empty_cache()

    # call BO
    event.set()

    global val
    opt_val = val

    print(opt_val)

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

    # set the prompts
    prompt = abstr + age +  "woman is looking at the camera with a smile on her face and a grey background, by NHK Animation, digital art, trending on artstation, illustration"
    negative_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    print("Running prompt: " + prompt)

    # stable-diffusion photo-generation script
    pipe = DiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V1.4",
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cuda")
    
    def dummy(images, **kwargs):
        return images, False
    pipe.safety_checker = dummy
    
    image = pipe(prompt=prompt, negative_prompt=negative_prompt).images[0]
    return image

# initializing Gradio-UI
def main():
    demo = gr.Interface(
        fn=diffusion,
        inputs=[gr.Checkbox(label="Avatar gut?"),],
        outputs="image"
    )
    demo.launch()

# start threads main and bo parallel
warnings.filterwarnings("ignore", category=UserWarning, module=".*botorch.*")
event = threading.Event()
threading.Thread(target=bayesian_opti).start()
main()
