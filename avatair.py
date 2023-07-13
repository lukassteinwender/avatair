import gradio as gr
import os
import numpy as np
import torch
import random
import threading
import warnings
import config
import prompting
from scripts import *
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
N_ITERATIONS = config.initial * 8 # Number of optimization iterations
MC_SAMPLES = 512 # Number of samples to approximate acquisition function
N_INITIAL = config.initial
SEED = 2 # Seed to initialize the initial samples obtained

SCALES = config.scales

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

problem_dim = 5 # dimension of the inputs X z.B.: ALter, Abstraktheit, (alles was wir ändern)
# dimension of the objectives Y z.B.: Vertraunswürdigkeit, Schönheit (alles was die Leute bewerten)
if SCALES == 1: num_objs = 5
if SCALES == 2: num_objs = 5
if SCALES == 3: num_objs = 1 # dimension of the objectives Y z.B.: Vertraunswürdigkeit, Schönheit (alles was die Leute bewerten)

INITIAL_CHECK = False
ITERATION_COUNT = 0

SCALE_1 = 0.3000
if(SCALES == 1 or SCALES == 2):
    SCALE_2 = 0.2000
    SCALE_3 = 0.2000
    SCALE_4 = 0.2000
    SCALE_5 = 0.2000

ABSTR_VAL = 1.00
AGE_VAL = 1.00
ETHN_VAL = 0.1000
GENDER_VAL = 0.3000
testval = 0.3000

ref_point = torch.tensor([-1. for _ in range(num_objs)]).cuda()
problem_bounds = torch.zeros(2, problem_dim, **tkwargs)

problem_bounds [1] = 1

def objective(x):
    x = x.cpu().numpy()
    global INITIAL_CHECK

    if INITIAL_CHECK == False:
        fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x) * random.randint(0,1000)
    # print(f"datatype: {type(fs)}")
    else:
        fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x) * random.randint(0,1000)
        global SCALE_1
        if(SCALES == 1 or SCALES == 2):
            global SCALE_2
            global SCALE_3
            global SCALE_4
            global SCALE_5
        fs[0] = SCALE_1
        if(SCALES == 1 or SCALES == 2):
            fs[1] = SCALE_2
            fs[2] = SCALE_3
            fs[3] = SCALE_4
            fs[4] = SCALE_5
    print(f"fs BEFORE : {fs}")
    fs = fs[:num_objs]
    print(f"fs AFTER : {fs}")

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
    
def mobo_execute(seed, iterations, initial_samples):
    
    event3.wait()
    event3.clear()
    
    torch.manual_seed(seed)

    hv = Hypervolume(ref_point = ref_point)
    # Hypervolumes
    hvs_qehvi = []

    # Initial Samples
    # train_x_qehvi, train_obj_qehvi = load_data()
    train_x_qehvi, train_obj_qehvi = generate_initial_data(n_samples=initial_samples)
    print(f"train_qehvi: {train_obj_qehvi.shape[-1]}")

    print("generated initial data")

    global INITIAL_CHECK

    INITIAL_CHECK = True

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
        global ITERATION_COUNT 
        ITERATION_COUNT = iteration
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
        # # # print("training obj", train_obj_qehvi)
        
        global ABSTR_VAL
        ABSTR_VAL = actualValues.data[0][0]

        global AGE_VAL
        AGE_VAL = actualValues.data[0][1]

        global ETHN_VAL
        ETHN_VAL = actualValues.data[0][2]

        global GENDER_VAL
        GENDER_VAL = actualValues.data[0][3]

        global testval
        testval = actualValues.data[0][4]

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

        event2.set()

    return hvs_qehvi, train_x_qehvi, train_obj_qehvi


# initializing Gradio-UI
def main():
    with gr.Blocks(title="AvatAIr") as demo:
        gr.Markdown("**AvatAIr**")
        with gr.Row():
            with gr.Column():
                infotext = gr.TextArea(value="Willkommen bei AvatAIr, \n\nSie werden im Laufe des Programmes immer wieder neue Avatare generiert bekommen. Diese bitten wir Sie, mit Hilfe von Slidern, welche Sie gleich sehen werden nach und nach zu bewerten. \n\nImmer wenn Ihre Bewertung fertig ist generiert das Programm einen neuen, auf Ihre Bewertung angepassten Avatar. Dieser Prozess wird mehrmals wiederholt, bis Ihnen zum Schluss des Programmes ihr finales Ergebnis präsentiert wird. \n\nDie Generierung der Avatare kann je nach Leistung des Systems etwas Zeit in Anspruch nehmen. Wir bitten deshalb um Geduld.\n\nVielen Dank.", interactive=False, show_label=False)
                global SCALES
                if(SCALES == 1):
                    inp1 = gr.Slider(0.0, 1.0, step=0.0001, value=0.11, label="acceptance", info="0 = low | 1 = high", visible=False)
                    inp2 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="likeability", info="0 = low | 1 = high", visible=False)
                    inp3 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="empathy", info="0 = low | 1 = high", visible=False)
                    inp4 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="anthropomorphism", info="0 = low | 1 = high", visible=False)
                    inp5 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="trust", info="0 = low | 1 = high", visible=False)
                if(SCALES == 2):
                    inp1 = gr.Slider(0.0, 1.0, step=0.0001, value=0.11, label="openness", info="0 = low | 1 = high", visible=False)
                    inp2 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="conscientiousness", info="0 = low | 1 = high", visible=False)
                    inp3 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="extraversion", info="0 = low | 1 = high", visible=False)
                    inp4 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="agreeableness", info="0 = low | 1 = high", visible=False)
                    inp5 = gr.Slider(0.0, 1.0, step=0.0001, value=0.28, label="neuroticism", info="0 = low | 1 = high", visible=False)
                if(SCALES == 3):
                    inp1 = gr.Slider(0.0, 1.0, step=0.0001, value=0.11, label="efficiency", info="0 = low | 1 = high", visible=False)
            out = gr.Image()
            out.style(height=512, width=512)
        
        def diffusion(scale1, scale2, scale3, scale4, scale5):
    
            # empty cuda-cache
            torch.cuda.empty_cache()

            global SCALE_1
            if(SCALES == 1 or SCALES == 2):
                global SCALE_2
                global SCALE_3
                global SCALE_4
                global SCALE_5

            SCALE_1 = scale1
            if(SCALES == 1 or SCALES == 2):
                SCALE_2 = scale2
                SCALE_3 = scale3
                SCALE_4 = scale4
                SCALE_5 = scale5

            # call BO
            event.set()
            event3.set()
            event2.wait()
            event2.clear()
            
            global ABSTR_VAL
            global AGE_VAL
            global ETHN_VAL
            global GENDER_VAL
            global testval

            sugarcheck= False

            print("ABSTR_VAL: ", ABSTR_VAL)
            print("AGE_VAL: ", AGE_VAL)
            print("ETHN_VAL: ", ETHN_VAL)
            print("GENDER_VAL: ", GENDER_VAL)
            print("testval: ", testval)

            # stable-diffusion photo-generation script
            torch.manual_seed(random.randint(0, 1000))
            
            model_id = "stabilityai/stable-diffusion-xl-base-0.9"
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker = None,
                requires_safety_checker = False
            )
            pipe = pipe.to("cuda")
            pipe.enable_vae_tiling()

            # wenn wir die setup pages haben können wir hier die art der prompterzeugung festlegen, also latent oder defined
            prompt = prompting.generate_definedprompt(ABSTR_VAL, AGE_VAL, ETHN_VAL, GENDER_VAL)
            print("Running prompt: " + prompt)
            negative_prompt = prompting.generate_negativePrompt()

            if(config.pictures == 1):
                image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=512, height=512).images[0]
            else:
                images = pipe(prompt=[prompt] * config.pictures, negative_prompt=[negative_prompt] * config.pictures, width=512, height=512).images
                grid = image_grid(images, rows=1, cols=config.pictures)
                image = grid

            global N_INITIAL
            global ITERATION_COUNT
            if(ITERATION_COUNT < N_INITIAL):
                if(SCALES == 1 or SCALES == 2):
                    return {
                        inp1: gr.update(visible=True),
                        inp2: gr.update(visible=True),
                        inp3: gr.update(visible=True),
                        inp4: gr.update(visible=True),
                        inp5: gr.update(visible=True),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(visible=False),
                        btn: gr.update(value="Generate new avatar")
                         
                    }
                else: 
                    return {
                        inp1: gr.update(visible=True),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(visible=False),
                        btn: gr.update(value="Generate new avatar")
                    }
            else:
                if(SCALES == 1 or SCALES == 2):
                    return {
                        inp1: gr.update(visible=False),
                        inp2: gr.update(visible=False),
                        inp3: gr.update(visible=False),
                        inp4: gr.update(visible=False),
                        inp5: gr.update(visible=False),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(value="Hier ist Ihr Ergebnis. \n\nVielen Dank für Ihre Teilnahme!", visible=True),
                        btn: gr.update(visible=False)
                    }
                else:
                    return {
                        inp1: gr.update(visible=False),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(value="Hier ist Ihr Ergebnis. \n\nVielen Dank für Ihre Teilnahme!", visible=True),
                        btn: gr.update(visible=False)
                    }
            
        btn = gr.Button("Run")
        if(SCALES == 1 or SCALES  == 2):
            btn.click(fn=diffusion, inputs=[inp1, inp2, inp3, inp4, inp5], outputs=[inp1,inp2,inp3,inp4,inp5,out,btn,infotext])
        else:
            btn.click(fn=diffusion, inputs=[inp1], outputs=[inp1,out,btn,infotext])
    demo.launch()

# start threads main and bo parallel
warnings.filterwarnings("ignore", category=UserWarning, module=".*botorch.*")
event = threading.Event()
event2 = threading.Event()
event3 = threading.Event()
t1 = threading.Thread(target=mobo_execute, args=(SEED, N_ITERATIONS, N_INITIAL))
t1.start()
main()
