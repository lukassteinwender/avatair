# -------------------------------------------------------------------------------------------
# AvatAIr  Config
# -------------------------------------------------------------------------------------------

# Number of iterations that will run until the program is finished:
initial = 20

# Which scales should be used ( 1 = acceptance, likeability, empathy, anthropomorphism, trust | 2 = openness, conscientiousness, extraversion, agreeableness, neuroticism | 3 = efficiency )
scales = 1

# How many pictures are generated during one iteration
pictures = 1

# After how many iterations should the attention check be entered? (Array (e.g., [1,2,3]), [-1] for no check)
attention = [1,3]

# Diffusers - Model ( RV: "SG161222/Realistic_Vision_V1.4" )
model = "SG161222/Realistic_Vision_V1.4"

# Your token for the huggingface-login (needed for some models. If not needed, leave as "")
token = ""

# The promptmodel that will be used for the model ("defined" or "latent")
promptmodel = "defined"
