import gradio as gr
import os
import numpy as np
import torch
import threading
import warnings
from diffusers import DiffusionPipeline
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
import socket
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt
from operator import itemgetter

tkwargs = {
    "dtype": torch.double,
   # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cuda")
}

BATCH_SIZE = 1 # Number of design parameter points to query at next iteration
NUM_RESTARTS = 10 # Used for the acquisition function number of restarts in optimization
RAW_SAMPLES = 1024 # Initial restart location candidates
N_ITERATIONS = 5 # Number of optimization iterations
MC_SAMPLES = 512 # Number of samples to approximate acquisition function
N_INITIAL = 10
SEED = 2 # Seed to initialize the initial samples obtained

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

problem_dim = 8 # dimension of the inputs X z.B.: ALter, Abstraktheit, (alles was wir ändern)
num_objs = 2 # dimension of the objectives Y z.B.: Vertraunswürdigkeit, Schönheit (alles was die Leute bewerten)

ABSTR_VAL = 1.00
AGE_VAL = 1.00

ref_point = torch.tensor([-1. for _ in range(num_objs)]).cuda()
problem_bounds = torch.zeros(2, problem_dim, **tkwargs)

problem_bounds [1] = 1

def objective(x):
    x = x.cpu().numpy()

    fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x) 
    print(f"fs: {fs}")
    print(f"datatype: {type(fs)}")
  
    fs = fs[:num_objs]
    print(f"return value: {torch.tensor(fs, dtype=torch.float64).cuda().shape[-1]}") # return.shape[-1] = 10
    return torch.tensor(fs, dtype=torch.float64).cuda()

def generate_initial_data(n_samples):
    # generate training data
    train_x = draw_sobol_samples(
        bounds=problem_bounds, n=1, q=n_samples, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0)
    train_obj = objective(train_x)
    print(f"train_x: {train_x.shape[0]}") # train_x.shape[0] = N_INITIAL und train_x.shape[-1] = 3

    train_obj = []
    for i, x in enumerate(train_x):
        print(f"initial sample: {i + 1}")
        train_obj.append(objective(x))
        print(f"objFunction: {objective(x).shape[-1]}") # objFunction.shape [-1] = 3

    train_obj = torch.stack(train_obj).cuda()
    print(f"train_obj: {train_obj.shape[-1]}") 

    return train_x, train_obj

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qehvi(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=problem_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem_bounds)
    return new_x

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
    
def mobo_execute(seed, iterations, initial_samples):
    torch.manual_seed(seed)
    
    hv = Hypervolume(ref_point = ref_point)
    # Hypervolumes
    hvs_qehvi = []


    # Initial Samples
    # train_x_qehvi, train_obj_qehvi = load_data()
    train_x_qehvi, train_obj_qehvi = generate_initial_data(n_samples=initial_samples)
    print(f"train_qehvi: {train_obj_qehvi.shape[-1]}")

    print("generated initial data")

    # Initialize GP models
    mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    print("initialized GP models")



    # Compute Pareto front and hypervolume
    pareto_mask = is_non_dominated(train_obj_qehvi)

    print("is_non_dominated(train_obj_qehvi) done")

    
    pareto_y = train_obj_qehvi[pareto_mask]
    pareto_y = -pareto_y

    print("train_obj_qehvi done")
    print(f"pareto_y: {pareto_y}")
    
    print(f"pareto_y_shape: {pareto_y.shape[-1]}") # pareto_y.shape[-1]: 3, pareto_y.shape[0]: 10
    print(f"ref_point_shape: {ref_point.shape[0]}") # ref_point.shape[-1]: 10, ref_point.shape[0]: 10
    print(f"mask_shape: {pareto_mask.shape[0]}") # pareto_mask.shape[-1]: 22, pareto_mask.shape[0]: 22



    volume = hv.compute(pareto_y)
    hvs_qehvi.append(volume)

    # Go through the iterations 

    print(f"iterations: {iterations}")

    for iteration in range(1, iterations + 1):

        print("Waiting for Input...")
        event.wait()
        event.clear()

        print("Iteration: " + str(iteration))
        #print(mll_qehvi)
        # Fit Models
        fit_gpytorch_model(mll_qehvi)

        print("after fit_gpytorch_model")
        # Define qEI acquisition modules using QMC sampler
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=(BATCH_SIZE))
        print("after qehvi_sampler")


        # Optimize acquisition functions and get new observations
        new_x_qehvi = optimize_qehvi(model_qehvi, train_obj_qehvi, qehvi_sampler)
        new_obj_qehvi = objective(new_x_qehvi[0])

        print("after optimize acq")
        print("new_x_qehvi")
        print(new_x_qehvi)
        print("train_x_qehvi")
        print(train_x_qehvi)

        print("train_obj_qehvi")
        print(train_obj_qehvi)
        print("new_obj_qehvi")
        print(new_obj_qehvi)

        # Update training points
        train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
        train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi.unsqueeze(0)])

        print("after update training")

        # Compute hypervolumes
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        volume = hv.compute(pareto_y)
        hvs_qehvi.append(volume)
        # Mark: das hier benötigen wir erst mal nicht
        # save_xy(train_x_qehvi, train_obj_qehvi, hvs_qehvi)
        print("mask", pareto_mask)
        print("pareto y", pareto_y)
        print("volume", volume)

        trainValues = train_x_qehvi
        actualValues = trainValues[-1:]
        print("training x", trainValues)
        print("RESULT: ", actualValues)
        print("training obj", train_obj_qehvi)
        
        global ABSTR_VAL
        ABSTR_VAL = actualValues.data[0][0]

        global AGE_VAL
        AGE_VAL = actualValues.data[0][1]

        event2.set()

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
    
    return hvs_qehvi, train_x_qehvi, train_obj_qehvi

# Photo generation on user-interaction
def diffusion(check, btn1, btn2, btn3, btn4, btn5, btn6, btn7):
    
    # empty cuda-cache
    torch.cuda.empty_cache()

    # call BO
    event.set()
    event2.wait()
    event2.clear()
    
    global ABSTR_VAL
    global AGE_VAL

    sugarcheck= False

    print("ABSTR_VAL", ABSTR_VAL)
    print("AGE_VAL", AGE_VAL)

    if 0.00 <= ABSTR_VAL < 0.2: abstr = "A ultra abstract "; sugarcheck = False
    if 0.2 <= ABSTR_VAL < 0.40: abstr = "A abstract "; sugarcheck = False
    if 0.4 <= ABSTR_VAL < 0.60: abstr = "A realistic "; sugarcheck = False
    if 0.6 <= ABSTR_VAL < 0.80: abstr = "A very realistic " ; sugarcheck = False
    if 0.8 <= ABSTR_VAL < 1.00: abstr = "A realistic "; sugarcheck = True

    if 0.00 <= AGE_VAL < 0.10: age = "10 y.o. "
    if 0.1 <= AGE_VAL < 0.15: age = "15 y.o. " 
    if 0.15 <= AGE_VAL < 0.20: age = "20 y.o. "
    if 0.2 <= AGE_VAL < 0.25: age = "25 y.o. "
    if 0.25 <= AGE_VAL < 0.30: age = "30 y.o. "
    if 0.30 <= AGE_VAL < 0.35: age = "35 y.o. "
    if 0.35 <= AGE_VAL < 0.40: age = "40 y.o. "
    if 0.4 <= AGE_VAL < 0.45: age = "45 y.o. "
    if 0.45 <= AGE_VAL < 0.50: age = "50 y.o. "
    if 0.5 <= AGE_VAL < 0.55: age = "55 y.o. "
    if 0.55 <= AGE_VAL < 0.60: age = "60 y.o. "
    if 0.6 <= AGE_VAL < 0.65: age = "65 y.o. "
    if 0.65 <= AGE_VAL < 0.70: age = "70 y.o. "
    if 0.7 <= AGE_VAL < 0.75: age = "75 y.o. "
    if 0.75 <= AGE_VAL < 0.80: age = "80 y.o. "
    if 0.8 <= AGE_VAL < 0.85: age = "85 y.o. "
    if 0.85 <= AGE_VAL < 0.90: age = "90 y.o. "
    if 0.9 <= AGE_VAL < 0.95: age = "95 y.o. "
    if 0.95 <= AGE_VAL < 1.00: age = "100 y.o. "



    # set the sugar
    if sugarcheck == True: 
        sugar = "(high detailed skin:1.2), 8k uhd, dslr,soft lighting, high quality, film grain"
    else:
        sugar = "by NHK Animation, digital art, trending on artstation, illustration"

    # set the prompts
    prompt = abstr + age +  "woman is looking at the camera with a smile on her face and a grey background, " + sugar
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
        outputs="image",
        description="Ein Interface zum erstellen individueller Avatare"
    )
    demo.launch()
    



    

# start threads main and bo parallel
warnings.filterwarnings("ignore", category=UserWarning, module=".*botorch.*")
event = threading.Event()
event2 = threading.Event()
t1 = threading.Thread(target=mobo_execute, args=(SEED, N_ITERATIONS, N_INITIAL))
t1.start()
main()
