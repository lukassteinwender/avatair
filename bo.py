import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# define the objective function to optimize
def objective(x):
    x == 1.00
    while x==0.00 or x==1.00: 
        x = torch.rand_like(x)
    return x

# define the function to query the human for feedback on each value
def query_human(x):
    response = input(f"Is {x:.2f} a good value? (y/n): ")
    if response.lower() == 'y':
        return 1
    else:
        return 0

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