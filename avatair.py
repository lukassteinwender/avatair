import gradio as gr
import os
import sys
import numpy as np
import torch
import random
import threading
import warnings
import config
import pyautogui
import prompting
import scripts
import logging
from scripts import *
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
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
from huggingface_hub import login
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

date = time.strftime("%Y_%m_%d-%H_%M_%S")
directory = os.path.dirname(os.path.realpath(__file__))
os.mkdir(directory + '\pic_log' + '\pic_' + date)
logfolder = directory + '\log' + '\log_' + date + '.log'
logging.basicConfig(filename=logfolder, encoding='utf-8', level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.info('Starting avatair...\n')

# important global values for the bayesian optimization
BATCH_SIZE = 1 # Number of design parameter points to query at next iteration
NUM_RESTARTS = 10 # Used for the acquisition function number of restarts in optimization
RAW_SAMPLES = 1024 # Initial restart location candidates
N_ITERATIONS = config.initial * 4 # Number of optimization iterations
MC_SAMPLES = 512 # Number of samples to approximate acquisition function
N_INITIAL = config.initial
SEED = random.randint(0,10000) # Seed to initialize the initial samples obtained

SCALES = config.scales

start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())

problem_dim = 22 # dimension of the inputs X z.B.: ALter, Abstraktheit, (alles was wir ändern)
# dimension of the objectives Y z.B.: Vertraunswürdigkeit, Schönheit (alles was die Leute bewerten)
if SCALES == 1: num_objs = 5
if SCALES == 2: num_objs = 5
if SCALES == 3: num_objs = 1 # dimension of the objectives Y z.B.: Vertraunswürdigkeit, Schönheit (alles was die Leute bewerten)

INITIAL_CHECK = False
ATT_CHECK_VAL = random.randint(0,100)
ITERATION_COUNT = 0

SCALE_1 = random.randint(0,1000) / 1000
if(SCALES == 1 or SCALES == 2):
    SCALE_2 = random.randint(0,1000) / 1000
    SCALE_3 = random.randint(0,1000) / 1000
    SCALE_4 = random.randint(0,1000) / 1000
    SCALE_5 = random.randint(0,1000) / 1000

ABSTR_VAL = 1.00
AGE_VAL = 1.00
GENDER_VAL = 0.3000
GLASSES_VAL = 0.1000
FACEWIDTH_VAL = 0.3000
FACIALHAIR_VAL = 0.3000
HAIRSTRUCTURE_VAL = 0.3000
STATUR_VAL = 0.3000
NOSE_VAL = 0.3000
MOUTH_VAL = 0.3000
EYESIZE_VAL = 0.3000
FACEWIDTH_VAL = 0.3000
EARS_VAL = 0.3000
SKINCOLOR_VAL_R= 0.3000
SKINCOLOR_VAL_G= 0.3000
SKINCOLOR_VAL_B= 0.3000
HAIRLENGTH_VAL= 0.3000
HAIRCOLOR_VAL_R= 0.3000
HAIRCOLOR_VAL_G= 0.3000
HAIRCOLOR_VAL_B= 0.3000
EYECOLOR_VAL_R= 0.3000
EYECOLOR_VAL_G= 0.3000
EYECOLOR_VAL_B= 0.3000

ref_point = torch.tensor([-1. for _ in range(num_objs)]).cuda()
problem_bounds = torch.zeros(2, problem_dim, **tkwargs)

problem_bounds [1] = 1

def objective(x):
    x = x.cpu().numpy()
    global INITIAL_CHECK
    global SCALES
    global SCALE_1
    if(SCALES == 1 or SCALES == 2):
        global SCALE_2
        global SCALE_3
        global SCALE_4
        global SCALE_5
    if INITIAL_CHECK == False:
        fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x)
    else:
        fs = 0.2 * (x - 0.3)**2 - 0.4 * np.sin(15.0 * x)
        
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

def restartAv():
    eventStop.wait()
    eventStop.clear()
    time.sleep(1)
    os.execv(sys.executable, ['python'] + sys.argv)

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

        global GLASSES_VAL
        GLASSES_VAL = actualValues.data[0][2]

        global GENDER_VAL
        GENDER_VAL = actualValues.data[0][3]

        global FACEWIDTH_VAL
        FACEWIDTH_VAL = actualValues.data[0][4]

        global FACIALHAIR_VAL
        FACIALHAIR_VAL = actualValues.data[0][5]

        global HAIRSTRUCTURE_VAL
        HAIRSTRUCTURE_VAL = actualValues.data[0][6]

        global STATUR_VAL
        STATUR_VAL = actualValues.data[0][7]

        global NOSE_VAL
        NOSE_VAL = actualValues.data[0][8]

        global MOUTH_VAL
        MOUTH_VAL = actualValues.data[0][9]

        global EYESIZE_VAL
        EYESIZE_VAL = actualValues.data[0][10]

        global EARS_VAL
        EARS_VAL = actualValues.data[0][11]
        
        global SKINCOLOR_VAL_R
        SKINCOLOR_VAL_R = actualValues.data[0][12]

        global SKINCOLOR_VAL_G
        SKINCOLOR_VAL_G = actualValues.data[0][13]

        global SKINCOLOR_VAL_B
        SKINCOLOR_VAL_B = actualValues.data[0][14]
        
        global HAIRLENGTH_VAL
        HAIRLENGTH_VAL = actualValues.data[0][15]

        global HAIRCOLOR_VAL_R
        HAIRCOLOR_VAL_R = actualValues.data[0][16]

        global HAIRCOLOR_VAL_G
        HAIRCOLOR_VAL_G = actualValues.data[0][17]

        global HAIRCOLOR_VAL_B
        HAIRCOLOR_VAL_B = actualValues.data[0][18]

        global EYECOLOR_VAL_R
        EYECOLOR_VAL_R = actualValues.data[0][19]

        global EYECOLOR_VAL_G
        EYECOLOR_VAL_G = actualValues.data[0][20]

        global EYECOLOR_VAL_B
        EYECOLOR_VAL_B = actualValues.data[0][21]

        logging.info('Optimized values:' + '\nabstraction: ' + str(ABSTR_VAL) + '\nage: ' + str(AGE_VAL) + '\nethnicity: ' + str(GLASSES_VAL) + '\ngender: ' + str(GENDER_VAL) + '\nface width: ' + str(FACEWIDTH_VAL) + '\nfacial hair: ' + str(FACIALHAIR_VAL) + '\nhair structure: ' + str(HAIRSTRUCTURE_VAL) + '\nstatur: ' + str(STATUR_VAL) + '\nnose: ' + str(NOSE_VAL) + '\nmouth: ' + str(MOUTH_VAL) + '\neye size: ' + str(EYESIZE_VAL) + '\nears: ' + str(EARS_VAL) + '\nskincolor_R: ' + str(SKINCOLOR_VAL_R) + '\nskincolor_G: ' + str(SKINCOLOR_VAL_G) + '\nskincolor_B: ' + str(SKINCOLOR_VAL_B) + '\nhair length: ' + str(HAIRLENGTH_VAL) + '\nhaircolor_R: ' + str(HAIRCOLOR_VAL_R) + '\nhaircolor_G: ' + str(HAIRCOLOR_VAL_G) + '\nhaircolor_B: ' + str(HAIRCOLOR_VAL_B) + '\neyecolor_R: ' + str(EYECOLOR_VAL_R) + '\neyecolor_G: ' + str(EYECOLOR_VAL_G) + '\neyecolor_B: ' + str(EYECOLOR_VAL_B) + '\n')

        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

        event2.set()

    return hvs_qehvi, train_x_qehvi, train_obj_qehvi


# initializing Gradio-UI
def main():
    global ATT_CHECK_VAL
    att_check_info = 'Pull the slider to ' + str(ATT_CHECK_VAL)
    with gr.Blocks(title="AvatAIr") as demo:
        gr.Markdown("**AvatAIr**")
        with gr.Row():
            with gr.Column():
                infotext = gr.TextArea(value="Welcome to AvatAIr, \n\nYou will get new avatars generated throughout the program. We ask you to rate them one by one using sliders, which you will see in a moment. \n\nEvery time your rating is finished the program will generate a new avatar adapted to your rating. This process is repeated several times until you are presented with your final result at the end of the program. \nThe generation of avatars may take some time depending on the performance of the system. Therefore we ask for your patience.\n\nThank you.", interactive=False, show_label=False)
                global SCALES
                if(SCALES == 1):
                    inp1 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="acceptance", info="0 = low | 1 = high | general agreement that the avatar is satisfactory or right", visible=False)
                    inp2 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="likeability", info="0 = low | 1 = high | how much do you like the avatar", visible=False)
                    inp3 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="empathy", info="0 = low | 1 = high | how empathic does the avatar effect", visible=False)
                    inp4 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="anthropomorphism", info="0 = low | 1 = high | how human-like is the avatar", visible=False)
                    inp5 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="trust", info="0 = low | 1 = high | how much do you trust the avatar", visible=False)
                if(SCALES == 2):
                    inp1 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="openness", info="0 = low | 1 = high | how open/acceptant does the avatar effect", visible=False)
                    inp2 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="conscientiousness", info="0 = low | 1 = high | does the avatar feel conscientious", visible=False)
                    inp3 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="extraversion", info="0 = low | 1 = high | how extroverted does the avatar look", visible=False)
                    inp4 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="agreeableness", info="0 = low | 1 = high | how sympathetic/cooperative/warm does the avatar feel", visible=False)
                    inp5 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="neuroticism", info="0 = low | 1 = high | how emotional stable would you rate the avatar", visible=False)
                if(SCALES == 3):
                    inp1 = gr.Slider(0.0, 1.0, step=0.0001, value=round(random.uniform(0.0000, 1.0000), 2), label="efficiency", info="0 = low | 1 = high | how good does the avatar work", visible=False)
                attention = gr.Slider(1, 100, step=1, value=0, label=att_check_info, visible=False)
                text_input = gr.Textbox(label="feedback (optional)", visible=False)
            out = gr.Image(visible=False)
            out.style(height=512, width=512)
        with gr.Row():
            btn = gr.Button(value="Run", scale=1)
            btnEnd = gr.Button(value="Interrupt survey", variant='stop', scale=2, visible=False)
            btnNoReturn = gr.Button(value="No, return to survey", scale=1, visible=False)
            btnYesEnd = gr.Button(value="Yes, interrupt survey", variant='stop', scale=2, visible=False)
            btnSubmitEnd = gr.Button(value="Submit reason", scale=2, visible=False)
            btnStartOver = gr.Button(value="Restart survey (optional)", scale=2, visible=False)
        
        def endquestion(scale1, scale2, scale3, scale4, scale5, att, txt):
            if(SCALES == 1 or SCALES == 2):
                return {
                    inp1: gr.update(visible=False),
                    inp2: gr.update(visible=False),
                    inp3: gr.update(visible=False),
                    inp4: gr.update(visible=False),
                    inp5: gr.update(visible=False),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=False),
                    btn: gr.update(visible=False),
                    btnEnd: gr.update(visible=False),
                    btnNoReturn: gr.update(visible=True),
                    btnYesEnd: gr.update(visible=True),
                    infotext: gr.update(value="Are you sure about closing the survey? \nYou will not get rewarded and your progress will be worthless.", visible=True)
                }
            else:
                return {
                    inp1: gr.update(visible=False),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=False),
                    btn: gr.update(visible=False),
                    btnEnd: gr.update(visible=False),
                    btnNoReturn: gr.update(visible=True),
                    btnYesEnd: gr.update(visible=True),
                    infotext: gr.update(value="Are you sure about closing the survey? \nYou will not get rewarded and your progress will be worthless.", visible=True)
                }
        
        def startOver(scale1, scale2, scale3, scale4, scale5, att, txt):
            logging.info('Restarted program, watch new log-file. \n')
            try:
                if(SCALES == 1 or SCALES == 2):
                    return {
                        btnStartOver:gr.update(visible=False),
                        infotext: gr.update(value="Restarting the survey, please reload the page in a few seconds if it isn't reloading by itself.", visible=True)
                    }
                else:
                    return {
                        btnStartOver:gr.update(visible=False),
                        infotext: gr.update(value="Restarting the survey, please reload the page in a few seconds if it isn't reloading by itself..", visible=True)
                    }
            finally:
                eventStop.set()

        def submitEnd(scale1, scale2, scale3, scale4, scale5, att, txt):
            logging.info('End-Reason:' + txt + '\n')
            if(SCALES == 1 or SCALES == 2):
                return {
                    inp1: gr.update(visible=False),
                    inp2: gr.update(visible=False),
                    inp3: gr.update(visible=False),
                    inp4: gr.update(visible=False),
                    inp5: gr.update(visible=False),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=False),
                    btn: gr.update(visible=False),
                    btnEnd: gr.update(visible=False),
                    btnNoReturn: gr.update(visible=False),
                    btnStartOver:gr.update(visible=True),
                    btnYesEnd: gr.update(visible=False),
                    infotext: gr.update(value="Thank you, you can close this page now.", visible=True),
                    text_input: gr.update(visible=False),
                    btnSubmitEnd: gr.update(visible=False)
                }
            else:
                return {
                    inp1: gr.update(visible=False),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=False),
                    btn: gr.update(visible=False),
                    btnEnd: gr.update(visible=False),
                    btnNoReturn: gr.update(visible=False),
                    btnYesEnd: gr.update(visible=False),
                    btnStartOver:gr.update(visible=True),
                    infotext: gr.update(value="Thank you, you can close this page now.", visible=True),
                    text_input: gr.update(visible=False),
                    btnSubmitEnd: gr.update(visible=False)
                }
        
        def returnbutton(scale1, scale2, scale3, scale4, scale5, att, txt):
            if(SCALES == 1 or SCALES == 2):
                return {
                    inp1: gr.update(visible=True),
                    inp2: gr.update(visible=True),
                    inp3: gr.update(visible=True),
                    inp4: gr.update(visible=True),
                    inp5: gr.update(visible=True),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=True),
                    btn: gr.update(visible=True),
                    btnEnd: gr.update(visible=True),
                    btnNoReturn: gr.update(visible=False),
                    btnYesEnd: gr.update(visible=False),
                    infotext: gr.update(visible=False)
                }
            else:
                return {
                    inp1: gr.update(visible=True),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=True),
                    btn: gr.update(visible=True),
                    btnEnd: gr.update(visible=True),
                    btnNoReturn: gr.update(visible=False),
                    btnYesEnd: gr.update(visible=False),
                    infotext: gr.update(visible=False)
                }
            
        def end(scale1, scale2, scale3, scale4, scale5, att, txt):
            logging.info('Survey interrupted by user!' + '\n')
            if(SCALES == 1 or SCALES == 2):
                return {
                    inp1: gr.update(visible=False),
                    inp2: gr.update(visible=False),
                    inp3: gr.update(visible=False),
                    inp4: gr.update(visible=False),
                    inp5: gr.update(visible=False),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=False),
                    btn: gr.update(visible=False),
                    btnEnd: gr.update(visible=False),
                    btnNoReturn: gr.update(visible=False),
                    btnYesEnd: gr.update(visible=False),
                    text_input: gr.update(visible=True),
                    infotext: gr.update(value="Thank you for your participation. \n\nCould you tell us the reason for interrupting the survey?", visible=True),
                    btnSubmitEnd: gr.update(visible=True)
                }
            else:
                return {
                    inp1: gr.update(visible=False),
                    attention: gr.update(visible=False),
                    out: gr.update(visible=False),
                    btn: gr.update(visible=False),
                    btnEnd: gr.update(visible=False),
                    btnNoReturn: gr.update(visible=False),
                    btnYesEnd: gr.update(visible=False),
                    text_input: gr.update(visible=True),
                    infotext: gr.update(value="Thank you for your participation. \n\nCould you tell us the reason for interrupting the survey?", visible=True),
                    btnSubmitEnd: gr.update(visible=True)
                }

        def diffusion(scale1, scale2, scale3, scale4, scale5, att, txt):
    
            # empty cuda-cache
            torch.cuda.empty_cache()
            global ITERATION_COUNT
            global ATT_CHECK_VAL

            if(ITERATION_COUNT in config.attention):
                if(ATT_CHECK_VAL == int(att)):
                    logging.info('Attention-Check successful\n')
                else:
                    logging.info('Attention-Check FAILED (should be: ' + str(ATT_CHECK_VAL) + ', is: ' + str(int(att)) + ')\n')
                ATT_CHECK_VAL = random.randint(0,100)

            logging.info('Running iteration ' + str(ITERATION_COUNT + 1) + '\n')

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

            if(SCALES == 1):
                logging.info('Slider Values:' + '\nacceptance: ' + str(scale1) + '\nlikeability: ' + str(scale2) + '\nempathy: ' + str(scale3) + '\nanthropomorphism: ' + str(scale4) + '\ntrust: ' + str(scale5) + '\n')

            if(SCALES == 2):
                logging.info('Slider Values:' + '\nopenness: ' + str(scale1) + '\nconscientiousness: ' + str(scale2) + '\nextraversion: ' + str(scale3) + '\nagreeableness: ' + str(scale4) + '\nneuroticism: ' + str(scale5) + '\n')

            if(SCALES == 3):
                logging.info('Slider Values:' + '\nefficiency: ' + str(scale1) + '\n')
            # call BO
            event.set()
            event3.set()
            event2.wait()
            event2.clear()
            
            global ABSTR_VAL
            global AGE_VAL
            global GLASSES_VAL
            global GENDER_VAL
            global FACEWIDTH_VAL
            global FACIALHAIR_VAL
            global HAIRSTRUCTURE_VAL
            global STATUR_VAL
            global NOSE_VAL
            global MOUTH_VAL
            global EYESIZE_VAL
            global EARS_VAL
            global SKINCOLOR_VAL_R
            global SKINCOLOR_VAL_G
            global SKINCOLOR_VAL_B
            global HAIRLENGTH_VAL
            global HAIRCOLOR_VAL_R
            global HAIRCOLOR_VAL_G
            global HAIRCOLOR_VAL_B
            global EYECOLOR_VAL_R
            global EYECOLOR_VAL_G
            global EYECOLOR_VAL_B

            sugarcheck = False

            # stable-diffusion photo-generation script
            torch.manual_seed(random.randint(0, 1000))
            
            model_id = config.model

            if(model_id == "SG161222/Realistic_Vision_V1.4"):
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    safety_checker = None,
                    requires_safety_checker = False,
                    use_safetensors=False
                )
                pipe = pipe.to("cuda")
                pipe.enable_vae_tiling()
            if(model_id == "stabilityai/stable-diffusion-xl-base-1.0"):
                pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
                pipe.enable_model_cpu_offload()
            
            # Prompterzeugung festlegen, latent oder defined
            if (config.promptmodel == "defined"):
                prompt = prompting.generate_definedprompt(ABSTR_VAL, AGE_VAL, GENDER_VAL, GLASSES_VAL, SKINCOLOR_VAL_R, SKINCOLOR_VAL_G, SKINCOLOR_VAL_B, FACEWIDTH_VAL, FACIALHAIR_VAL,  HAIRLENGTH_VAL, HAIRSTRUCTURE_VAL, HAIRCOLOR_VAL_R, HAIRCOLOR_VAL_G, HAIRCOLOR_VAL_B, STATUR_VAL, NOSE_VAL, MOUTH_VAL, EYECOLOR_VAL_R, EYECOLOR_VAL_G, EYECOLOR_VAL_B, EYESIZE_VAL, EARS_VAL)
                logging.info('Running prompt: ' + prompt + '\n')
                print("Running prompt: " + prompt)
                negative_prompt = prompting.generate_defined_negativePrompt()
            elif (config.promptmodel == "latent"):
                prompt = prompting.generate_latentprompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL)
                logging.info('Running prompt: ' + prompt + '\n')
                print("Running prompt: " + prompt)
                negative_prompt = prompting.generate_latent_negativePrompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL)
            global INITIAL_CHECK
            INITIAL_CHECK = True
            

            if(config.pictures == 1):
                image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=512, height=512).images[0]
            else:
                images = pipe(prompt=[prompt] * config.pictures, negative_prompt=[negative_prompt] * config.pictures, width=512, height=512).images
                grid = scripts.image_grid(images, rows=1, cols=config.pictures)
                image = grid

            img = image
            global directory
            global date
            image_path = directory + '\pic_log' + '\pic_' + date + '\image_' + str(ITERATION_COUNT) + '.jpg'
            img.save(image_path)

            global N_INITIAL
            if(ITERATION_COUNT < N_INITIAL):
                if(ITERATION_COUNT in config.attention):
                    logging.info('Running attention-check\n')
                    att_check_info = 'Pull the slider to ' + str(ATT_CHECK_VAL)
                    if(SCALES == 1 or SCALES == 2):
                        return {
                            inp1: gr.update(visible=True),
                            inp2: gr.update(visible=True),
                            inp3: gr.update(visible=True),
                            inp4: gr.update(visible=True),
                            inp5: gr.update(visible=True),
                            out: gr.update(value=image, visible=True),
                            infotext: gr.update(visible=False),
                            btn: gr.update(value="Generate new avatar"),
                            btnEnd: gr.update(visible=True),
                            btnNoReturn: gr.update(visible=False),
                            btnYesEnd: gr.update(visible=False),
                            attention: gr.update(visible=True, label=att_check_info)
                        }
                    else: 
                        return {
                            inp1: gr.update(visible=True),
                            attention: gr.update(visible=True, label=att_check_info),
                            out: gr.update(value=image, visible=True),
                            infotext: gr.update(visible=False),
                            btnEnd: gr.update(visible=True),
                            btnNoReturn: gr.update(visible=False),
                            btnYesEnd: gr.update(visible=False),
                            btn: gr.update(value="Generate new avatar")
                        }
                elif(SCALES == 1 or SCALES == 2):
                    return {
                        inp1: gr.update(visible=True),
                        inp2: gr.update(visible=True),
                        inp3: gr.update(visible=True),
                        inp4: gr.update(visible=True),
                        inp5: gr.update(visible=True),
                        btnEnd: gr.update(visible=True),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(visible=False),
                        attention: gr.update(visible=False),
                        btnNoReturn: gr.update(visible=False),
                        btnYesEnd: gr.update(visible=False),
                        btn: gr.update(value="Generate new avatar")
                         
                    }
                else: 
                    return {
                        inp1: gr.update(visible=True),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(visible=False),
                        btnEnd: gr.update(visible=True),
                        attention: gr.update(visible=False),
                        btnNoReturn: gr.update(visible=False),
                        btnYesEnd: gr.update(visible=False),
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
                        btnStartOver: gr.update(visible=True),
                        attention: gr.update(visible=False),
                        btn: gr.update(visible=False),
                        btnEnd: gr.update(visible=False),
                        out: gr.update(value=image, visible=True),
                        btnNoReturn: gr.update(visible=False),
                        btnYesEnd: gr.update(visible=False),
                        infotext: gr.update(value="This is your result. \nThank you for your participation.", visible=True),
                    }
                else:
                    return {
                        inp1: gr.update(visible=False),
                        btnStartOver: gr.update(visible=True),
                        attention: gr.update(visible=False),
                        out: gr.update(value=image, visible=True),
                        btn: gr.update(visible=False),
                        btnEnd: gr.update(visible=False),
                        btnNoReturn: gr.update(visible=False),
                        btnYesEnd: gr.update(visible=False),
                        infotext: gr.update(value="This is your result. \nThank you for your participation.", visible=True),
                    }

        if(SCALES == 1 or SCALES  == 2):
            btn.click(fn=diffusion, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext, text_input,btnStartOver])
            btnEnd.click(fn=endquestion, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext, text_input,btnStartOver])
            btnYesEnd.click(fn=end, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnNoReturn,btnSubmitEnd,infotext, text_input,btnStartOver])
            btnNoReturn.click(fn=returnbutton, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext, text_input,btnStartOver])
            btnSubmitEnd.click(fn=submitEnd, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnSubmitEnd,btnYesEnd,btnNoReturn,infotext, text_input,btnStartOver])
            btnStartOver.click(fn=startOver, inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver])
        else:
            btn.click(fn=diffusion, inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnNoReturn,btnSubmitEnd,infotext,attention, text_input,btnStartOver])
            btnEnd.click(fn=endquestion, inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver])
            btnYesEnd.click(fn=end , inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver])
            btnNoReturn.click(fn=returnbutton, inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver])
            btnSubmitEnd.click(fn=submitEnd, inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver])
            btnStartOver.click(fn=startOver, inputs=[inp1, attention, text_input], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver])
    pyautogui.hotkey('f5')
    demo.launch()

# start threads main and bo parallel
if (config.token != ""):
    login(token=config.token)
warnings.filterwarnings("ignore", category=UserWarning, module=".*botorch.*")
event = threading.Event()
event2 = threading.Event()
event3 = threading.Event()
eventStop = threading.Event()
t1 = threading.Thread(target=mobo_execute, args=(SEED, N_ITERATIONS, N_INITIAL))
t2 = threading.Thread(target=restartAv)
t1.start()
t2.start()
main()