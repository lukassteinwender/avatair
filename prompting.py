
def generate_definedprompt(ABSTR_VAL, AGE_VAL, ETHN_VAL, GENDER_VAL):

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
        prompt = abstr + age + ethn + gender + "is looking at the camera with a proud expression on the face and a blue background, " + sugar


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


def generate_negativePrompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL):
     
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

def generate_negativePrompt():
    
    n_prompt=  "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature, black-white"
    return n_prompt