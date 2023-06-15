import gradio as gr
import os
import numpy as np
import torch
import random
import threading
import warnings
import config
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
        fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x) 
    # print(f"datatype: {type(fs)}")
    else:
        fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x) 
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

            if 0.00 <= ABSTR_VAL < 0.20: abstr = "A ultra abstract "; sugarcheck = False
            if 0.20 <= ABSTR_VAL < 0.40: abstr = "A abstract "; sugarcheck = False
            if 0.40 <= ABSTR_VAL < 0.60: abstr = "A realistic "; sugarcheck = False
            if 0.60 <= ABSTR_VAL < 0.80: abstr = "A very realistic " ; sugarcheck = False
            if 0.80 <= ABSTR_VAL <= 1.00: abstr = "A realistic "; sugarcheck = True

            if 0.00 <= AGE_VAL < 0.01: age = "1 y.o. "
            if 0.01 <= AGE_VAL < 0.02: age = "2 y.o. "
            if 0.02 <= AGE_VAL < 0.03: age = "3 y.o. "
            if 0.03 <= AGE_VAL < 0.04: age = "4 y.o. "
            if 0.04 <= AGE_VAL < 0.05: age = "5 y.o. "
            if 0.05 <= AGE_VAL < 0.06: age = "6 y.o. "
            if 0.06 <= AGE_VAL < 0.07: age = "7 y.o. "
            if 0.07 <= AGE_VAL < 0.08: age = "8 y.o. "
            if 0.08 <= AGE_VAL < 0.09: age = "9 y.o. "
            if 0.09 <= AGE_VAL < 0.10: age = "10 y.o. "
            if 0.10 <= AGE_VAL < 0.11: age = "11 y.o. "
            if 0.11 <= AGE_VAL < 0.12: age = "12 y.o. "
            if 0.12 <= AGE_VAL < 0.13: age = "13 y.o. "
            if 0.13 <= AGE_VAL < 0.14: age = "14 y.o. "
            if 0.14 <= AGE_VAL < 0.15: age = "15 y.o. "
            if 0.15 <= AGE_VAL < 0.16: age = "16 y.o. "
            if 0.16 <= AGE_VAL < 0.17: age = "17 y.o. "
            if 0.17 <= AGE_VAL < 0.18: age = "18 y.o. "
            if 0.18 <= AGE_VAL < 0.19: age = "19 y.o. "
            if 0.19 <= AGE_VAL < 0.20: age = "20 y.o. "
            if 0.20 <= AGE_VAL < 0.21: age = "21 y.o. "
            if 0.21 <= AGE_VAL < 0.22: age = "22 y.o. "
            if 0.22 <= AGE_VAL < 0.23: age = "23 y.o. "
            if 0.23 <= AGE_VAL < 0.24: age = "24 y.o. "
            if 0.24 <= AGE_VAL < 0.25: age = "25 y.o. "
            if 0.25 <= AGE_VAL < 0.26: age = "26 y.o. "
            if 0.26 <= AGE_VAL < 0.27: age = "27 y.o. "
            if 0.27 <= AGE_VAL < 0.28: age = "28 y.o. "
            if 0.28 <= AGE_VAL < 0.29: age = "29 y.o. "
            if 0.29 <= AGE_VAL < 0.30: age = "30 y.o. "
            if 0.30 <= AGE_VAL < 0.31: age = "31 y.o. "
            if 0.31 <= AGE_VAL < 0.32: age = "32 y.o. "
            if 0.32 <= AGE_VAL < 0.33: age = "33 y.o. "
            if 0.33 <= AGE_VAL < 0.34: age = "34 y.o. "
            if 0.34 <= AGE_VAL < 0.35: age = "35 y.o. "
            if 0.35 <= AGE_VAL < 0.36: age = "36 y.o. "
            if 0.36 <= AGE_VAL < 0.37: age = "37 y.o. "
            if 0.37 <= AGE_VAL < 0.38: age = "38 y.o. "
            if 0.38 <= AGE_VAL < 0.39: age = "39 y.o. "
            if 0.39 <= AGE_VAL < 0.40: age = "40 y.o. "
            if 0.40 <= AGE_VAL < 0.41: age = "41 y.o. "
            if 0.41 <= AGE_VAL < 0.42: age = "42 y.o. "
            if 0.42 <= AGE_VAL < 0.43: age = "43 y.o. "
            if 0.43 <= AGE_VAL < 0.44: age = "44 y.o. "
            if 0.44 <= AGE_VAL < 0.45: age = "45 y.o. "
            if 0.45 <= AGE_VAL < 0.46: age = "46 y.o. "
            if 0.46 <= AGE_VAL < 0.47: age = "47 y.o. "
            if 0.47 <= AGE_VAL < 0.48: age = "48 y.o. "
            if 0.48 <= AGE_VAL < 0.49: age = "49 y.o. "
            if 0.49 <= AGE_VAL < 0.50: age = "50 y.o. "
            if 0.50 <= AGE_VAL < 0.51: age = "51 y.o. "
            if 0.51 <= AGE_VAL < 0.52: age = "52 y.o. "
            if 0.52 <= AGE_VAL < 0.53: age = "53 y.o. "
            if 0.53 <= AGE_VAL < 0.54: age = "54 y.o. "
            if 0.54 <= AGE_VAL < 0.55: age = "55 y.o. "
            if 0.55 <= AGE_VAL < 0.56: age = "56 y.o. "
            if 0.56 <= AGE_VAL < 0.57: age = "57 y.o. "
            if 0.57 <= AGE_VAL < 0.58: age = "58 y.o. "
            if 0.58 <= AGE_VAL < 0.59: age = "59 y.o. "
            if 0.59 <= AGE_VAL < 0.60: age = "60 y.o. "
            if 0.60 <= AGE_VAL < 0.61: age = "61 y.o. "
            if 0.61 <= AGE_VAL < 0.62: age = "62 y.o. "
            if 0.62 <= AGE_VAL < 0.63: age = "63 y.o. "
            if 0.63 <= AGE_VAL < 0.64: age = "64 y.o. "
            if 0.64 <= AGE_VAL < 0.65: age = "65 y.o. "
            if 0.65 <= AGE_VAL < 0.66: age = "66 y.o. "
            if 0.66 <= AGE_VAL < 0.67: age = "67 y.o. "
            if 0.67 <= AGE_VAL < 0.68: age = "68 y.o. "
            if 0.68 <= AGE_VAL < 0.69: age = "69 y.o. "
            if 0.69 <= AGE_VAL < 0.70: age = "70 y.o. "
            if 0.70 <= AGE_VAL < 0.71: age = "71 y.o. "
            if 0.71 <= AGE_VAL < 0.72: age = "72 y.o. "
            if 0.72 <= AGE_VAL < 0.73: age = "73 y.o. "
            if 0.73 <= AGE_VAL < 0.74: age = "74 y.o. "
            if 0.74 <= AGE_VAL < 0.75: age = "75 y.o. "
            if 0.75 <= AGE_VAL < 0.76: age = "76 y.o. "
            if 0.76 <= AGE_VAL < 0.77: age = "77 y.o. "
            if 0.77 <= AGE_VAL < 0.78: age = "78 y.o. "
            if 0.78 <= AGE_VAL < 0.79: age = "79 y.o. "
            if 0.79 <= AGE_VAL < 0.80: age = "80 y.o. "
            if 0.80 <= AGE_VAL < 0.81: age = "81 y.o. "
            if 0.81 <= AGE_VAL < 0.82: age = "82 y.o. "
            if 0.82 <= AGE_VAL < 0.83: age = "83 y.o. "
            if 0.83 <= AGE_VAL < 0.84: age = "84 y.o. "
            if 0.84 <= AGE_VAL < 0.85: age = "85 y.o. "
            if 0.85 <= AGE_VAL < 0.86: age = "86 y.o. "
            if 0.86 <= AGE_VAL < 0.87: age = "87 y.o. "
            if 0.87 <= AGE_VAL < 0.88: age = "88 y.o. "
            if 0.88 <= AGE_VAL < 0.89: age = "89 y.o. "
            if 0.89 <= AGE_VAL < 0.90: age = "90 y.o. "
            if 0.90 <= AGE_VAL < 0.91: age = "91 y.o. "
            if 0.91 <= AGE_VAL < 0.92: age = "92 y.o. "
            if 0.92 <= AGE_VAL < 0.93: age = "93 y.o. "
            if 0.93 <= AGE_VAL < 0.94: age = "94 y.o. "
            if 0.94 <= AGE_VAL < 0.95: age = "95 y.o. "
            if 0.95 <= AGE_VAL < 0.96: age = "96 y.o. "
            if 0.96 <= AGE_VAL < 0.97: age = "97 y.o. "
            if 0.97 <= AGE_VAL < 0.98: age = "98 y.o. "
            if 0.98 <= AGE_VAL < 0.99: age = "99 y.o. "
            if 0.99 <= AGE_VAL < 1.00: age = "100 y.o. "
            


            if 0.00 <= ETHN_VAL < 0.0263: ethn = "german "
            if 0.0263 <= ETHN_VAL < 0.0526: ethn = "french "
            if 0.0526 <= ETHN_VAL < 0.0789: ethn = "italian "
            if 0.0789 <= ETHN_VAL < 0.1052: ethn = "polish "
            if 0.1052 <= ETHN_VAL < 0.1315: ethn = "english "
            if 0.1315 <= ETHN_VAL < 0.1578: ethn = "irish "
            if 0.1578 <= ETHN_VAL < 0.1841: ethn = "mexican "
            if 0.1841 <= ETHN_VAL < 0.2104: ethn = "salvadorian "
            if 0.2104 <= ETHN_VAL < 0.2367: ethn = "puerto rican "
            if 0.2367 <= ETHN_VAL < 0.2630: ethn = "dominican "
            if 0.2630 <= ETHN_VAL < 0.2893: ethn = "cuban "
            if 0.2893 <= ETHN_VAL < 0.3156: ethn = "colombian "
            if 0.3156 <= ETHN_VAL < 0.3419: ethn = "africnan american "
            if 0.3419 <= ETHN_VAL < 0.3682: ethn = "jamaican "
            if 0.3682 <= ETHN_VAL < 0.3945: ethn = "haitian "
            if 0.3945 <= ETHN_VAL < 0.4208: ethn = "nigerian "
            if 0.4208 <= ETHN_VAL < 0.4471: ethn = "ethiopian "
            if 0.4471 <= ETHN_VAL < 0.4734: ethn = "somalian "
            if 0.4734 <= ETHN_VAL < 0.4997: ethn = "chinese "
            if 0.4997 <= ETHN_VAL < 0.5260: ethn = "vietnamese "
            if 0.5260 <= ETHN_VAL < 0.5523: ethn = "filipino "
            if 0.5523 <= ETHN_VAL < 0.5786: ethn = "korean "
            if 0.5786 <= ETHN_VAL < 0.6049: ethn = "asian indian "
            if 0.6049 <= ETHN_VAL < 0.6312: ethn = "japanese "
            if 0.6312 <= ETHN_VAL < 0.6575: ethn = "american indian "
            if 0.6575 <= ETHN_VAL < 0.6838: ethn = "alaskan native "
            if 0.6838 <= ETHN_VAL < 0.7101: ethn = "lebanese "
            if 0.7101 <= ETHN_VAL < 0.7364: ethn = "iranian "
            if 0.7364 <= ETHN_VAL < 0.7627: ethn = "egyptian "
            if 0.7627 <= ETHN_VAL < 0.7890: ethn = "syrian "
            if 0.7890 <= ETHN_VAL < 0.8153: ethn = "moroccan "
            if 0.8153 <= ETHN_VAL < 0.8416: ethn = "israeli "
            if 0.8416 <= ETHN_VAL < 0.8679: ethn = "native hawaiian "
            if 0.8679 <= ETHN_VAL < 0.8942: ethn = "samoan "
            if 0.8942 <= ETHN_VAL < 0.9205: ethn = "chamorro "
            if 0.9205 <= ETHN_VAL < 0.9468: ethn = "tongan "
            if 0.9468 <= ETHN_VAL < 0.9731: ethn = "fijian "
            if 0.9731 <= ETHN_VAL < 0.9994: ethn = "marshallese "
            if 0.9994 <= ETHN_VAL <= 1.0000: ethn = "swedish "



            if 0.00 <= GENDER_VAL < 0.20: gender = "female person "
            if 0.20 <= GENDER_VAL < 0.40: gender = "(female:0.25) person "
            if 0.40 <= GENDER_VAL < 0.60: gender = "Androgynous "
            if 0.60 <= GENDER_VAL < 0.80: gender = "(male:0.25) person "
            if 0.80 <= GENDER_VAL < 1.00: gender = "male person "




            # set the sugar
            if sugarcheck == True: 
                sugar = "(high detailed skin:1.2), 8k uhd, dslr,soft lighting, high quality, film grain"
            else:
                sugar = "by NHK Animation, digital art, trending on artstation, illustration"

            # set the prompts
            prompt = abstr + age + ethn + gender + "is looking at the camera with a smile on the face and a grey background, " + sugar
            negative_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
            print("Running prompt: " + prompt)

            # stable-diffusion photo-generation script
            torch.manual_seed(random.randint(0, 1000))
            
            pipe = DiffusionPipeline.from_pretrained(
                "SG161222/Realistic_Vision_V1.4",
                torch_dtype=torch.float32,
                safety_checker = None,
                requires_safety_checker = False
            )
            pipe = pipe.to("cuda")

            image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=512, height=512).images[0]

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
