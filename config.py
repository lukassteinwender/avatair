# -------------------------------------------------------------------------------------------
# AvatAIr  Config
# -------------------------------------------------------------------------------------------

# Number of iterations that will run until the program is finished:
initial = 5

# Which scales should be used ( 1 = acceptance, likeability, empathy, anthropomorphism, trust | 2 = openness, conscientiousness, extraversion, agreeableness, neuroticism | 3 = efficiency )
scales = 1

# How much pictures are generated during one iteration
pictures = 1

# After how much iterations should enter the attention check ? (Array (e.g [1,2,3]), [-1] for no check)
attention = [1,3]

# Diffusers - Model ( RV: "SG161222/Realistic_Vision_V1.4" )
model = "SG161222/Realistic_Vision_V1.4"

# Your token for the huggingface-login (needed for some models, if not needed leave as "")
token = ""