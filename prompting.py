
import color

def generate_definedprompt(ABSTR_VAL, AGE_VAL, GENDER_VAL, GLASSES_VAL, SKINCOLOR_VAL_R, SKINCOLOR_VAL_G, SKINCOLOR_VAL_B, FACEWIDTH_VAL, FACIALHAIR_VAL,  HAIRLENGTH_VAL, HAIRSTRUCTURE_VAL, HAIRCOLOR_VAL_R, HAIRCOLOR_VAL_G, HAIRCOLOR_VAL_B, STATUR_VAL, NOSE_VAL, MOUTH_VAL, EYECOLOR_VAL_R, EYECOLOR_VAL_G, EYECOLOR_VAL_B, EYESIZE_VAL, EARS_VAL):

        
        #conversion to rgb values 0-255 & generate color as text
        skincolor_r = round(SKINCOLOR_VAL_R.item()*255)
        skincolor_g = round(SKINCOLOR_VAL_G.item()*255)
        skincolor_b = round(SKINCOLOR_VAL_B.item()*255)
        SKINCOLOR_VAL = color.colour_to_text((skincolor_r, skincolor_g, skincolor_b))

        haircolor_r = round(HAIRCOLOR_VAL_R.item()*255)
        haircolor_g = round(HAIRCOLOR_VAL_G.item()*255)
        haircolor_b = round(HAIRCOLOR_VAL_B.item()*255)
        HAIRCOLOR_VAL = color.colour_to_text((haircolor_r, haircolor_g, haircolor_b))

        eyecolor_r = round(EYECOLOR_VAL_R.item()*255)
        eyecolor_g = round(EYECOLOR_VAL_G.item()*255)
        eyecolor_b = round(EYECOLOR_VAL_B.item()*255)
        EYECOLOR_VAL = color.colour_to_text((eyecolor_r, eyecolor_g, eyecolor_b))

        
        #set the the abstractness level
        if 0.00 <= ABSTR_VAL < 0.20: abstr = "A ultra abstract "; sugarcheck = False
        if 0.20 <= ABSTR_VAL < 0.40: abstr = "A abstract "; sugarcheck = False
        if 0.40 <= ABSTR_VAL < 0.60: abstr = "A realistic "; sugarcheck = False
        if 0.60 <= ABSTR_VAL < 0.80: abstr = "A very realistic " ; sugarcheck = False
        if 0.80 <= ABSTR_VAL <= 1.00: abstr = "A realistic "; sugarcheck = True

        #set Age
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

        #set Gender
        if 0.00 <= GENDER_VAL < 0.34: gender = "female person "
        if 0.34 <= GENDER_VAL < 0.67: gender = "Androgynous "
        if 0.67 <= GENDER_VAL < 1.00: gender = "male person "
        
        #set Glasses
        if 0.00 <= GLASSES_VAL < 0.50: glasses = "with glasses, "
        if 0.50 <= GLASSES_VAL <= 1.00: glasses = "without glasses, "

        #set the face width
        if 0.00 <= FACEWIDTH_VAL < 0.34: facewidth = "thin "
        if 0.34 <= FACEWIDTH_VAL < 0.67: facewidth = "medium-sized "
        if 0.67 <= FACEWIDTH_VAL <= 1.00: facewidth = "wide "

        #set the facial hair
        if 0.00 <= FACIALHAIR_VAL < 0.34: facialhair = "clean shaven "
        if 0.34 <= FACIALHAIR_VAL < 0.67: facialhair = "little bearded"
        if 0.67 <= FACIALHAIR_VAL <= 1.00: facialhair = "bearded "

        #set the hair structure
        if 0.00 <= HAIRSTRUCTURE_VAL < 0.20: hairstructure = "stick straight "
        if 0.20 <= HAIRSTRUCTURE_VAL < 0.40: hairstructure = "straight "
        if 0.40 <= HAIRSTRUCTURE_VAL < 0.60: hairstructure = "wavy "
        if 0.60 <= HAIRSTRUCTURE_VAL < 0.80: hairstructure = "big curls "
        if 0.80 <= HAIRSTRUCTURE_VAL <= 1.00: hairstructure = "small curls "

        #set the statur
        if 0.00 <= STATUR_VAL < 0.20: statur = "very thin stature, "
        if 0.20 <= STATUR_VAL < 0.40: statur = "thin stature, "
        if 0.40 <= STATUR_VAL < 0.60: statur = "regular stature, "
        if 0.60 <= STATUR_VAL < 0.80: statur = "obese stature, "
        if 0.80 <= STATUR_VAL <= 1.00: statur = "very obese stature, "

        #set the nose, mouth, eyesize and ears
        if 0.00 <= NOSE_VAL < 0.25: nose = "no nose, "
        if 0.25 <= NOSE_VAL < 0.50: nose = "small nose, "
        if 0.50 <= NOSE_VAL < 0.75: nose = "medium-sized nose, "
        if 0.75 <= NOSE_VAL <= 1.00: nose = "big nose, "
        
        if 0.00 <= MOUTH_VAL < 0.25: mouth = "no mouth, "
        if 0.25 <= MOUTH_VAL < 0.50: mouth = "small mouth, "
        if 0.50 <= MOUTH_VAL < 0.75: mouth = "medium-sized mouth, "
        if 0.75 <= MOUTH_VAL <= 1.00: mouth = "big mouth, "

        if 0.00 <= EYESIZE_VAL < 0.25: eyesize = "no "
        if 0.25 <= EYESIZE_VAL < 0.50: eyesize = "small "
        if 0.50 <= EYESIZE_VAL < 0.75: eyesize = "medium-sized "
        if 0.75 <= EYESIZE_VAL <= 1.00: eyesize = "big "

        if 0.00 <= EARS_VAL < 0.25: ears= "no ears "
        if 0.25 <= EARS_VAL < 0.50: ears= "small ears "
        if 0.50 <= EARS_VAL < 0.75: ears= "medium-sized ears "
        if 0.75 <= EARS_VAL <= 1.00: ears= "big ears "

        #set the hairlength and convert it to two decimal places
        hairlength = "{:.2f}".format(HAIRLENGTH_VAL.item())

            # set the sugar
        if sugarcheck == True: 
            sugar = "(high detailed skin:1.2), 8k uhd, dslr,soft lighting, high quality, film grain"
        else:
            sugar = "by NHK Animation, digital art, trending on artstation, illustration"

        # set the prompts
        prompt = abstr + age + gender + "infront of grey background " + glasses + SKINCOLOR_VAL + " skin color, " + facewidth + facialhair + " face, " +  str(hairlength) + "m long " + hairstructure + HAIRCOLOR_VAL + " hair, " + statur + nose + mouth + eyesize + EYECOLOR_VAL + " eyes "+ "and " + ears + "is looking at the camera with a proud expression on the face, " + sugar


        return prompt


def generate_latentprompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL):

    if calculate_postivevalue(OPEN_VAL) > 0:
        openness= "(Openness:" + calculate_postivevalue(OPEN_VAL) + ")"
    elif calculate_postivevalue(CON_VAL) > 0:   
        conscientiousness= "(Conscientiousness:" + calculate_postivevalue(CON_VAL) + ")"
    elif calculate_postivevalue(EXTRA_VAL) > 0:
        extraversion= "(Extraversion:" + calculate_postivevalue(EXTRA_VAL) + ")"
    elif calculate_postivevalue(AGREE_VAL) > 0:
        agreeableness= "(Agreeableness:" + calculate_postivevalue(AGREE_VAL) + ")"
    elif calculate_postivevalue(NEURO_VAL) > 0:
        neuroticism= "(Neuroticism:" + calculate_postivevalue(NEURO_VAL) + ")"
    elif calculate_postivevalue(ACCEPT_VAL) > 0:
        acceptance= "(Acceptance:" + calculate_postivevalue(ACCEPT_VAL) + ")"
    elif calculate_postivevalue(LIKE_VAL) > 0:
        likeability= "(Likeability:" + calculate_postivevalue(LIKE_VAL) + ")"
    elif calculate_postivevalue(EMP_VAL) > 0:
        empathy= "(Empathy:" + calculate_postivevalue(EMP_VAL) + ")"
    elif calculate_postivevalue(ANTHRO_VAL) > 0:
        anthropomorphism= "(Anthropomorphism:" + calculate_postivevalue(ANTHRO_VAL) + ")"
    elif calculate_postivevalue(TRUST_VAL) > 0:
        trust= "(Trust:" + calculate_postivevalue(TRUST_VAL) + ")"
    prompt = "A portrait captures a person exuding traits of  " + openness + ", " + conscientiousness + ", " + extraversion + ", " + agreeableness + ", " + neuroticism + ", " + acceptance + ", " + likeability + ", " + empathy + ", " + anthropomorphism + ", " + trust + " , looking confidently at the camera with a proud expression, against a blue background"

    return prompt


def calculate_postivevalue(input_parameter):
    if input_parameter > 0.5:
        # Positive Prompt
        ergebnis = (input_parameter - 0.5)/0.5
    else:
        # Neutral & Input-Parameter ist kleiner 0,5
        ergebnis = 0
    
    return ergebnis

def calculate_negativevalue(input_parameter):
    if input_parameter < 0.5:
        # Negative Prompt
        ergebnis = 1- (input_parameter * 2)
    else:
        # Neutral & Input-Parameter ist größer 0,5
        ergebnis = 0
    
    return ergebnis


def generate_latent_negativePrompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL):
     
    if calculate_negativevalue(OPEN_VAL) > 0:
        openness= "(Openness:" + calculate_negativevalue(OPEN_VAL) + ")"
    elif calculate_negativevalue(CON_VAL) > 0:   
        conscientiousness= "(Conscientiousness:" + calculate_negativevalue(CON_VAL) + ")"
    elif calculate_negativevalue(EXTRA_VAL) > 0:
        extraversion= "(Extraversion:" + calculate_negativevalue(EXTRA_VAL) + ")"
    elif calculate_negativevalue(AGREE_VAL) > 0:
        agreeableness= "(Agreeableness:" + calculate_negativevalue(AGREE_VAL) + ")"
    elif calculate_negativevalue(NEURO_VAL) > 0:
        neuroticism= "(Neuroticism:" + calculate_negativevalue(NEURO_VAL) + ")"
    elif calculate_negativevalue(ACCEPT_VAL) > 0:
        acceptance= "(Acceptance:" + calculate_negativevalue(ACCEPT_VAL) + ")"
    elif calculate_negativevalue(LIKE_VAL) > 0:
        likeability= "(Likeability:" + calculate_negativevalue(LIKE_VAL) + ")"
    elif calculate_negativevalue(EMP_VAL) > 0:
        empathy= "(Empathy:" + calculate_negativevalue(EMP_VAL) + ")"
    elif calculate_negativevalue(ANTHRO_VAL) > 0:
        anthropomorphism= "(Anthropomorphism:" + calculate_negativevalue(ANTHRO_VAL) + ")"
    elif calculate_negativevalue(TRUST_VAL) > 0:
        trust= "(Trust:" + calculate_negativevalue(TRUST_VAL) + ")"
    n_prompt=  "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature, black-white, " + openness + ", " + conscientiousness + ", " + extraversion + ", " + agreeableness + ", " + neuroticism + ", " + acceptance + ", " + likeability + ", " + empathy + ", " + anthropomorphism + ", " + trust
    return n_prompt

def generate_defined_negativePrompt():
    
    n_prompt=  "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature, black-white"
    return n_prompt