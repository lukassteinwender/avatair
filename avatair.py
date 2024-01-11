import gradio as gr
import os
import csv
import io
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
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
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

class CsvFormatter(logging.Formatter):
    
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, delimiter =';', quoting=csv.QUOTE_MINIMAL)

    def format(self, record):
        x = record.msg.split(";")
        self.writer.writerow(x)
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)
        return data.strip()

tkwargs = {
    "dtype": torch.double,
   # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cuda")
}

date = time.strftime("%Y_%m_%d-%H_%M_%S")
directory = os.path.dirname(os.path.realpath(__file__))
os.mkdir(directory + '\pic_log' + '\pic_' + date)
logfolder = directory + '\log' + '\log_' + date + '.csv'
logging.basicConfig(filename=logfolder, encoding='utf-8', level=logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.root.handlers[0].setFormatter(CsvFormatter())

if (config.promptmodel == "defined"):
    if(config.scales == 1):
        logging.info('iteration;prompt;negative_prompt;acceptance_sl;likeability_sl;empathy_sl;anthropomorphism_sl;trust_sl;abstraction;age;ethnicity;gender;face width;facial hair;hair structure;statur;nose;mouth;eye size;ears;skincolor_R;skincolor_G;skincolor_B;hair length;haircolor_R;haircolor_G;haircolor_B;eyecolor_R;eyecolor_G;eyecolor_B;run_att_check;att_check_successful;user_id;cfg_initial;cfg_scales;cfg_pictures;cfg_attention;cfg_model;cfg_promptmodel')
    if(config.scales == 2):
        logging.info('iteration;prompt;negative_prompt;openness_sl;conscientiousness_sl;extraversion_sl;agreeableness_sl;neuroticism_sl;abstraction;age;ethnicity;gender;face width;facial hair;hair structure;statur;nose;mouth;eye size;ears;skincolor_R;skincolor_G;skincolor_B;hair length;haircolor_R;haircolor_G;haircolor_B;eyecolor_R;eyecolor_G;eyecolor_B;run_att_check;att_check_successful;user_id;cfg_initial;cfg_scales;cfg_pictures;cfg_attention;cfg_model;cfg_promptmodel')
    if(config.scales == 3):
        logging.info('iteration;prompt;negative_prompt;efficiency_sl;abstraction;age;ethnicity;gender;face width;facial hair;hair structure;statur;nose;mouth;eye size;ears;skincolor_R;skincolor_G;skincolor_B;hair length;haircolor_R;haircolor_G;haircolor_B;eyecolor_R;eyecolor_G;eyecolor_B;run_att_check;att_check_successful;user_id;cfg_initial;cfg_scales;cfg_pictures;cfg_attention;cfg_model;cfg_promptmodel')
elif (config.promptmodel == "latent"):
    if(config.scales == 1):
        logging.info('iteration;prompt;negative_prompt;acceptance_sl;likeability_sl;empathy_sl;anthropomorphism_sl;trust_sl;openness;conscientiousness;extraversion;agreeableness;neuroticism;acceptance;likeability;empathy;anthropomorphism;trust;run_att_check;att_check_successful;user_id;cfg_initial;cfg_scales;cfg_pictures;cfg_attention;cfg_model;cfg_promptmodel')
    if(config.scales == 2):
        logging.info('iteration;prompt;negative_prompt;openness_sl;conscientiousness_sl;extraversion_sl;agreeableness_sl;neuroticism_sl;openness;conscientiousness;extraversion;agreeableness;neuroticism;acceptance;likeability;empathy;anthropomorphism;trust;run_att_check;att_check_successful;user_id;cfg_initial;cfg_scales;cfg_pictures;cfg_attention;cfg_model;cfg_promptmodel')
    if(config.scales == 3):
        logging.info('iteration;prompt;negative_prompt;efficiency_sl;openness;conscientiousness;extraversion;agreeableness;neuroticism;acceptance;likeability;empathy;anthropomorphism;trust;run_att_check;att_check_successful;user_id;cfg_initial;cfg_scales;cfg_pictures;cfg_attention;cfg_model;cfg_promptmodel')    

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

# dimension of the inputs X z.B.: ALter, Abstraktheit, (alles was wir ändern)
if (config.promptmodel == "defined"):
    problem_dim = 22 
elif (config.promptmodel == "latent"):
    problem_dim = 10

# dimension of the objectives Y z.B.: Vertraunswürdigkeit, Schönheit (alles was die Leute bewerten)
if SCALES == 1: num_objs = 5
if SCALES == 2: num_objs = 5
if SCALES == 3: num_objs = 1

INITIAL_CHECK = False
ATT_CHECK_VAL = random.randint(0,100)
ITERATION_COUNT = 0

RUN_ATT_CHECK = False
ATT_CHECK_SUCCESSFUL = True

SCALE_1 = 0
if(SCALES == 1 or SCALES == 2):
    SCALE_2 = 0
    SCALE_3 = 0
    SCALE_4 = 0
    SCALE_5 = 0

if (config.promptmodel == "defined"):
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
elif (config.promptmodel == "latent"):
    OPEN_VAL= 0.3000
    CON_VAL= 0.3000
    EXTRA_VAL= 0.3000
    AGREE_VAL= 0.3000
    NEURO_VAL= 0.3000
    ACCEPT_VAL= 0.3000
    LIKE_VAL= 0.3000
    EMP_VAL= 0.3000
    ANTHRO_VAL= 0.3000
    TRUST_VAL= 0.3000

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
        fs = 0.2 * (x - 0.3)**0
        fs[0] = 1 - SCALE_1
        if(SCALES == 1 or SCALES == 2):
            fs[1] = 1 - SCALE_2
            fs[2] = 1 - SCALE_3
            fs[3] = 1 - SCALE_4
            fs[4] = 1 - SCALE_5
    fs = fs[:num_objs]

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
    time.sleep(2)
    os.execv(sys.executable, ['python'] + sys.argv)

def mobo_execute(seed, iterations, initial_samples):
    
    global ITERATION_COUNT 

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
        
        if (config.promptmodel == "defined"):
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
        elif (config.promptmodel == "latent"):

            global OPEN_VAL
            OPEN_VAL = actualValues.data[0][0]

            global CON_VAL
            CON_VAL = actualValues.data[0][1]

            global EXTRA_VAL
            EXTRA_VAL = actualValues.data[0][2]

            global AGREE_VAL
            AGREE_VAL = actualValues.data[0][3]

            global NEURO_VAL
            NEURO_VAL = actualValues.data[0][4]

            global ACCEPT_VAL
            ACCEPT_VAL = actualValues.data[0][5]

            global LIKE_VAL
            LIKE_VAL = actualValues.data[0][6]

            global EMP_VAL
            EMP_VAL = actualValues.data[0][7]

            global ANTHRO_VAL
            ANTHRO_VAL = actualValues.data[0][8]
            
            global TRUST_VAL
            TRUST_VAL = actualValues.data[0][9]

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
                user_id = gr.Textbox(label="User ID", visible=True)
            out = gr.Image(visible=False, scale=0, min_width=512)
            #out.style(height=512, width=512) # pre Gradio 4.0
        with gr.Row():
            btn = gr.Button(value="Run", scale=1)
            btnEnd = gr.Button(value="Interrupt survey", variant='stop', scale=2, visible=False)
            btnNoReturn = gr.Button(value="No, return to survey", scale=1, visible=False)
            btnYesEnd = gr.Button(value="Yes, interrupt survey", variant='stop', scale=2, visible=False)
            btnSubmitEnd = gr.Button(value="Submit reason", scale=2, visible=False)
            btnStartOver = gr.Button(value="Restart survey (optional)", scale=2, visible=False)
        
        def endquestion(scale1, scale2, scale3, scale4, scale5, att, txt, u_id):
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
        
        def startOver(scale1, scale2, scale3, scale4, scale5, att, txt, u_id):
            #logging.info('restart,TRUE')
            
            try:
                if(SCALES == 1 or SCALES == 2):
                    return {
                        out: gr.update(visible=False),
                        btnStartOver:gr.update(visible=False),
                        infotext: gr.update(value="Restarting the survey, please reload the page in a few seconds if it isn't reloading by itself.", visible=True)
                    }
                else:
                    return {
                        out: gr.update(visible=False),
                        btnStartOver:gr.update(visible=False),
                        infotext: gr.update(value="Restarting the survey, please reload the page in a few seconds if it isn't reloading by itself..", visible=True)
                    }
            finally:
                eventStop.set()

        def submitEnd(scale1, scale2, scale3, scale4, scale5, att, txt, u_id):
            #logging.info('end_reason,' + txt)
            
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
        
        def returnbutton(scale1, scale2, scale3, scale4, scale5, att, txt, u_id):
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
            
        def end(scale1, scale2, scale3, scale4, scale5, att, txt, u_id):
            #logging.info('interrupt,TRUE')
            
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

        def diffusion(scale1, scale2, scale3, scale4, scale5, att, txt, u_id):
    
            # empty cuda-cache
            torch.cuda.empty_cache()
            global ITERATION_COUNT
            global ATT_CHECK_VAL
            global RUN_ATT_CHECK
            global ATT_CHECK_SUCCESSFUL

            if(ITERATION_COUNT in config.attention):
                if(ATT_CHECK_VAL == int(att)):
                    #logging.info('attention_check_successful,TRUE')
                    ATT_CHECK_SUCCESSFUL = True
                    
                else:
                    #logging.info('attention_check_successful,FALSE')
                    ATT_CHECK_SUCCESSFUL = False
                    
                ATT_CHECK_VAL = random.randint(0,100)

            #logging.info('iteration,' + str(ITERATION_COUNT + 1))
            
            print("USER-ID " + u_id)

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

            if (config.promptmodel == "defined"):
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
            elif (config.promptmodel == "latent"):
                global OPEN_VAL
                global CON_VAL
                global EXTRA_VAL
                global AGREE_VAL
                global NEURO_VAL
                global ACCEPT_VAL
                global LIKE_VAL
                global EMP_VAL
                global ANTHRO_VAL
                global TRUST_VAL
            
            # Prompterzeugung festlegen, latent oder defined
            if (config.promptmodel == "defined"):
                prompt = prompting.generate_definedprompt(ABSTR_VAL, AGE_VAL, GENDER_VAL, GLASSES_VAL, SKINCOLOR_VAL_R, SKINCOLOR_VAL_G, SKINCOLOR_VAL_B, FACEWIDTH_VAL, FACIALHAIR_VAL,  HAIRLENGTH_VAL, HAIRSTRUCTURE_VAL, HAIRCOLOR_VAL_R, HAIRCOLOR_VAL_G, HAIRCOLOR_VAL_B, STATUR_VAL, NOSE_VAL, MOUTH_VAL, EYECOLOR_VAL_R, EYECOLOR_VAL_G, EYECOLOR_VAL_B, EYESIZE_VAL, EARS_VAL)
                #logging.info('prompt,' + prompt.replace(","," "))
                
                print("Prompt: " + prompt)
                print(" ")
                negative_prompt = prompting.generate_defined_negativePrompt()
                print("Negative prompt:" + negative_prompt)
                #logging.info('negative_prompt,' + negative_prompt.replace(","," "))
                
            elif (config.promptmodel == "latent"):
                prompt = prompting.generate_latentprompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL)
                #logging.info('prompt,' + prompt.replace(","," "))
                
                print("Prompt: " + prompt)
                print(" ")
                negative_prompt = prompting.generate_latent_negativePrompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL)
                print("Negative prompt: " + negative_prompt)
                #logging.info('negative_prompt,' + negative_prompt.replace(","," "))

            if (config.promptmodel == "defined"):
                if(SCALES == 1):
                    logging.info(str(ITERATION_COUNT) + ';' + prompt + ';' + negative_prompt + ';' + str(scale1) + ';' + str(scale2) + ';' + str(scale3) + ';' + str(scale4) + ';' + str(scale5) + ';' + str("{:.2f}".format(ABSTR_VAL.item())) + ';' + str("{:.2f}".format(AGE_VAL.item())) + ';' + str("{:.2f}".format(GLASSES_VAL.item())) + ';' + str("{:.2f}".format(GENDER_VAL.item())) + ';' + str("{:.2f}".format(FACEWIDTH_VAL.item())) + ';' + str("{:.2f}".format(FACIALHAIR_VAL.item())) + ';' + str("{:.2f}".format(HAIRSTRUCTURE_VAL.item())) + ';' + str("{:.2f}".format(STATUR_VAL.item())) + ';' + str("{:.2f}".format(NOSE_VAL.item())) + ';' + str("{:.2f}".format(MOUTH_VAL.item())) + ';' + str("{:.2f}".format(EYESIZE_VAL.item())) + ';' + str("{:.2f}".format(EARS_VAL.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_B.item())) + ';' + str("{:.2f}".format(HAIRLENGTH_VAL.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_B.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_B.item())) + ';' + str(RUN_ATT_CHECK) + ';' + str(ATT_CHECK_SUCCESSFUL) + ';' + str(u_id)+ ';' + str(config.initial)+ ';' + str(config.scales)+ ';' + str(config.pictures)+ ';' + str(config.attention)+ ';' + str(config.model)+ ';' + str(config.promptmodel))
                if(SCALES == 2):
                    logging.info(str(ITERATION_COUNT) + ';' + prompt + ';' + negative_prompt + ';' + str(scale1) + ';' + str(scale2) + ';' + str(scale3) + ';' + str(scale4) + ';' + str(scale5) + ';' + str("{:.2f}".format(ABSTR_VAL.item())) + ';' + str("{:.2f}".format(AGE_VAL.item())) + ';' + str("{:.2f}".format(GLASSES_VAL.item())) + ';' + str("{:.2f}".format(GENDER_VAL.item())) + ';' + str("{:.2f}".format(FACEWIDTH_VAL.item())) + ';' + str("{:.2f}".format(FACIALHAIR_VAL.item())) + ';' + str("{:.2f}".format(HAIRSTRUCTURE_VAL.item())) + ';' + str("{:.2f}".format(STATUR_VAL.item())) + ';' + str("{:.2f}".format(NOSE_VAL.item())) + ';' + str("{:.2f}".format(MOUTH_VAL.item())) + ';' + str("{:.2f}".format(EYESIZE_VAL.item())) + ';' + str("{:.2f}".format(EARS_VAL.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_B.item())) + ';' + str("{:.2f}".format(HAIRLENGTH_VAL.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_B.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_B.item())) + ';' + str(RUN_ATT_CHECK) + ';' + str(ATT_CHECK_SUCCESSFUL) + ';' + str(u_id)+ ';' + str(config.initial)+ ';' + str(config.scales)+ ';' + str(config.pictures)+ ';' + str(config.attention)+ ';' + str(config.model)+ ';' + str(config.promptmodel))
                if(SCALES == 3):
                    logging.info(str(ITERATION_COUNT) + ';' + prompt + ';' + negative_prompt + ';' + str(scale1) + ';' + str("{:.2f}".format(ABSTR_VAL.item())) + ';' + str("{:.2f}".format(AGE_VAL.item())) + ';' + str("{:.2f}".format(GLASSES_VAL.item())) + ';' + str("{:.2f}".format(GENDER_VAL.item())) + ';' + str("{:.2f}".format(FACEWIDTH_VAL.item())) + ';' + str("{:.2f}".format(FACIALHAIR_VAL.item())) + ';' + str("{:.2f}".format(HAIRSTRUCTURE_VAL.item())) + ';' + str("{:.2f}".format(STATUR_VAL.item())) + ';' + str("{:.2f}".format(NOSE_VAL.item())) + ';' + str("{:.2f}".format(MOUTH_VAL.item())) + ';' + str("{:.2f}".format(EYESIZE_VAL.item())) + ';' + str("{:.2f}".format(EARS_VAL.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(SKINCOLOR_VAL_B.item())) + ';' + str("{:.2f}".format(HAIRLENGTH_VAL.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(HAIRCOLOR_VAL_B.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_R.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_G.item())) + ';' + str("{:.2f}".format(EYECOLOR_VAL_B.item())) + ';' + str(RUN_ATT_CHECK) + ';' + str(ATT_CHECK_SUCCESSFUL) + ';' + str(u_id)+ ';' + str(config.initial)+ ';' + str(config.scales)+ ';' + str(config.pictures)+ ';' + str(config.attention)+ ';' + str(config.model)+ ';' + str(config.promptmodel))
            elif (config.promptmodel == "latent"):
                if(SCALES == 1):
                    logging.info(str(ITERATION_COUNT) + ';' + prompt + ';' + negative_prompt + ';' + str(scale1) + ';' + str(scale2) + ';' + str(scale3) + ';' + str(scale4) + ';' + str(scale5) + ';' + str("{:.2f}".format(OPEN_VAL.item())) + ';' + str("{:.2f}".format(CON_VAL.item())) + ';' + str("{:.2f}".format(EXTRA_VAL.item())) + ';' + str("{:.2f}".format(AGREE_VAL.item())) + ';' + str("{:.2f}".format(NEURO_VAL.item())) + ';' + str("{:.2f}".format(ACCEPT_VAL.item())) + ';' + str("{:.2f}".format(LIKE_VAL.item())) + ';' + str("{:.2f}".format(EMP_VAL.item())) + ';' + str("{:.2f}".format(ANTHRO_VAL.item())) + ';' + str("{:.2f}".format(TRUST_VAL.item())) + ';' + str(RUN_ATT_CHECK) + ';' + str(ATT_CHECK_SUCCESSFUL) + ';' + str(u_id)+ ';' + str(config.initial)+ ';' + str(config.scales)+ ';' + str(config.pictures)+ ';' + str(config.attention)+ ';' + str(config.model)+ ';' + str(config.promptmodel))
                if(SCALES == 2):
                    logging.info(str(ITERATION_COUNT) + ';' + prompt + ';' + negative_prompt + ';' + str(scale1) + ';' + str(scale2) + ';' + str(scale3) + ';' + str(scale4) + ';' + str(scale5) + ';' + str("{:.2f}".format(OPEN_VAL.item())) + ';' + str("{:.2f}".format(CON_VAL.item())) + ';' + str("{:.2f}".format(EXTRA_VAL.item())) + ';' + str("{:.2f}".format(AGREE_VAL.item())) + ';' + str("{:.2f}".format(NEURO_VAL.item())) + ';' + str("{:.2f}".format(ACCEPT_VAL.item())) + ';' + str("{:.2f}".format(LIKE_VAL.item())) + ';' + str("{:.2f}".format(EMP_VAL.item())) + ';' + str("{:.2f}".format(ANTHRO_VAL.item())) + ';' + str("{:.2f}".format(TRUST_VAL.item())) + ';' + str(RUN_ATT_CHECK) + ';' + str(ATT_CHECK_SUCCESSFUL) + ';' + str(u_id)+ ';' + str(config.initial)+ ';' + str(config.scales)+ ';' + str(config.pictures)+ ';' + str(config.attention)+ ';' + str(config.model)+ ';' + str(config.promptmodel))
                if(SCALES == 3):
                    logging.info(str(ITERATION_COUNT) + ';' + prompt + ';' + negative_prompt + ';' + str("{:.2f}".format(OPEN_VAL.item())) + ';' + str("{:.2f}".format(CON_VAL.item())) + ';' + str("{:.2f}".format(EXTRA_VAL.item())) + ';' + str("{:.2f}".format(AGREE_VAL.item())) + ';' + str("{:.2f}".format(NEURO_VAL.item())) + ';' + str("{:.2f}".format(ACCEPT_VAL.item())) + ';' + str("{:.2f}".format(LIKE_VAL.item())) + ';' + str("{:.2f}".format(EMP_VAL.item())) + ';' + str("{:.2f}".format(ANTHRO_VAL.item())) + ';' + str("{:.2f}".format(TRUST_VAL.item())) + ';' + str(RUN_ATT_CHECK) + ';' + str(ATT_CHECK_SUCCESSFUL) + ';' + str(u_id)+ ';' + str(config.initial)+ ';' + str(config.scales)+ ';' + str(config.pictures)+ ';' + str(config.attention)+ ';' + str(config.model)+ ';' + str(config.promptmodel))
               

            sugarcheck = False

            # stable-diffusion photo-generation script
            torch.manual_seed(random.randint(0, 1000))
            
            steps=20
            if(config.stablediffusion == "xl"):
                pipe = DiffusionPipeline.from_pretrained(config.model, torch_dtype=torch.float16, use_safetensors=True)
                pipe.enable_model_cpu_offload()
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(config.model, torch_dtype=torch.float16)
                #pipe.enable_model_cpu_offload()
                pipe = pipe.to("cuda")
                pipe.enable_vae_tiling()
                
            global INITIAL_CHECK
            INITIAL_CHECK = True
            

            if(config.pictures == 1):
                image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=steps, width=512, height=512).images[0]
            else:
                images = pipe(prompt=[prompt] * config.pictures, negative_prompt=[negative_prompt] * config.pictures, num_inference_steps=steps, width=512, height=512).images
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
                    RUN_ATT_CHECK = True
                    #logging.info('run_attention_check,TRUE')
                    
                    att_check_info = 'Pull the slider to ' + str(ATT_CHECK_VAL)
                    if(SCALES == 1 or SCALES == 2):
                        return {
                            inp1: gr.update(visible=True),
                            inp2: gr.update(visible=True),
                            inp3: gr.update(visible=True),
                            inp4: gr.update(visible=True),
                            inp5: gr.update(visible=True),
                            user_id: gr.update(visible=False),
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
                            user_id: gr.update(visible=False),
                            attention: gr.update(visible=True, label=att_check_info),
                            out: gr.update(value=image, visible=True),
                            infotext: gr.update(visible=False),
                            btnEnd: gr.update(visible=True),
                            btnNoReturn: gr.update(visible=False),
                            btnYesEnd: gr.update(visible=False),
                            btn: gr.update(value="Generate new avatar")
                        }
                elif(SCALES == 1 or SCALES == 2):
                    #logging.info('run_attention_check,FALSE')
                    RUN_ATT_CHECK = False
                    
                    return {
                        inp1: gr.update(visible=True),
                        inp2: gr.update(visible=True),
                        inp3: gr.update(visible=True),
                        inp4: gr.update(visible=True),
                        inp5: gr.update(visible=True),
                        user_id: gr.update(visible=False),
                        btnEnd: gr.update(visible=True),
                        out: gr.update(value=image, visible=True),
                        infotext: gr.update(visible=False),
                        attention: gr.update(visible=False),
                        btnNoReturn: gr.update(visible=False),
                        btnYesEnd: gr.update(visible=False),
                        btn: gr.update(value="Generate new avatar")
                         
                    }
                else:
                    #logging.info('run_attention_check,FALSE')
                    RUN_ATT_CHECK = False
                    
                    return {
                        inp1: gr.update(visible=True),
                        user_id: gr.update(visible=False),
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
                        user_id: gr.update(visible=False),
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
                        user_id: gr.update(visible=False),
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
            btn.click(fn=diffusion, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input, user_id], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext, text_input,btnStartOver, user_id])
            btnEnd.click(fn=endquestion, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input, user_id], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext, text_input,btnStartOver, user_id])
            btnYesEnd.click(fn=end, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input, user_id], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnNoReturn,btnSubmitEnd,infotext, text_input,btnStartOver, user_id])
            btnNoReturn.click(fn=returnbutton, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input, user_id], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext, text_input,btnStartOver, user_id])
            btnSubmitEnd.click(fn=submitEnd, inputs=[inp1, inp2, inp3, inp4, inp5, attention, text_input, user_id], outputs=[inp1,inp2,inp3,inp4,inp5,attention,out,btn,btnEnd,btnSubmitEnd,btnYesEnd,btnNoReturn,infotext, text_input,btnStartOver, user_id])
            btnStartOver.click(fn=startOver, inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver, user_id])
        else:
            btn.click(fn=diffusion, inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnNoReturn,btnSubmitEnd,infotext,attention, text_input,btnStartOver, user_id])
            btnEnd.click(fn=endquestion, inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver, user_id])
            btnYesEnd.click(fn=end , inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver, user_id])
            btnNoReturn.click(fn=returnbutton, inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver, user_id])
            btnSubmitEnd.click(fn=submitEnd, inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver, user_id])
            btnStartOver.click(fn=startOver, inputs=[inp1, attention, text_input, user_id], outputs=[inp1,out,btn,btnEnd,btnYesEnd,btnSubmitEnd,btnNoReturn,infotext,attention, text_input,btnStartOver, user_id])
    pyautogui.hotkey('f5')
    demo.launch()

# start threads main and bo parallel
if (config.token != ""):
    login(token=config.token)
warnings.filterwarnings("ignore", category=UserWarning, module=".*botorch.*")
warnings.filterwarnings("ignore", category=UserWarning, module=".*gradio.*")
event = threading.Event()
event2 = threading.Event()
event3 = threading.Event()
eventStop = threading.Event()
t1 = threading.Thread(target=mobo_execute, args=(SEED, N_ITERATIONS, N_INITIAL))
t2 = threading.Thread(target=restartAv)
t1.start()
t2.start()
main()
