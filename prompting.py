import color

def generate_definedprompt(ABSTR_VAL, AGE_VAL, GENDER_VAL, GLASSES_VAL, SKINCOLOR_VAL_R, SKINCOLOR_VAL_G, SKINCOLOR_VAL_B, FACEWIDTH_VAL, FACIALHAIR_VAL,  HAIRLENGTH_VAL, HAIRSTRUCTURE_VAL, HAIRCOLOR_VAL_R, HAIRCOLOR_VAL_G, HAIRCOLOR_VAL_B, STATUR_VAL, NOSE_VAL, MOUTH_VAL, EYECOLOR_VAL_R, EYECOLOR_VAL_G, EYECOLOR_VAL_B, EYESIZE_VAL, EARS_VAL):

        # --- SAFETY DEFAULTS (Prevents UnboundLocalError) ---
        abstr = "A "
        sugarcheck = False
        age = "adult "
        gender = "person "
        glasses = ""
        facewidth = ""
        facialhair = ""
        hairstructure = ""
        statur = ""
        nose = ""
        mouth = ""
        eyesize = ""
        ears = ""
        
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

        
        #set the abstractness level
        if 0.00 <= ABSTR_VAL < 0.20: abstr = "A ultra abstract "; sugarcheck = False
        elif 0.20 <= ABSTR_VAL < 0.40: abstr = "A abstract "; sugarcheck = False
        elif 0.40 <= ABSTR_VAL < 0.60: abstr = "A realistic "; sugarcheck = False
        elif 0.60 <= ABSTR_VAL < 0.80: abstr = "A very realistic " ; sugarcheck = False
        elif 0.80 <= ABSTR_VAL <= 1.00: abstr = "A realistic "; sugarcheck = True

        #set Age
        age = f"{min(100, int(AGE_VAL.item() * 100) + 1)} y.o. "

        #set Gender
        if 0.00 <= GENDER_VAL < 0.34: gender = "female person "
        elif 0.34 <= GENDER_VAL < 0.67: gender = "Androgynous "
        else: gender = "male person " # Fixed logic: catches 0.67 to 1.00
        
        #set Glasses
        if 0.00 <= GLASSES_VAL < 0.50: glasses = "with glasses, "
        else: glasses = "without glasses, "

        #set the face width
        if 0.00 <= FACEWIDTH_VAL < 0.34: facewidth = "thin "
        elif 0.34 <= FACEWIDTH_VAL < 0.67: facewidth = "medium-sized "
        else: facewidth = "wide "

        #set the facial hair
        if 0.00 <= FACIALHAIR_VAL < 0.34: facialhair = "clean shaven "
        elif 0.34 <= FACIALHAIR_VAL < 0.67: facialhair = "little bearded"
        else: facialhair = "bearded "

        #set the hair structure
        if 0.00 <= HAIRSTRUCTURE_VAL < 0.20: hairstructure = "stick straight "
        elif 0.20 <= HAIRSTRUCTURE_VAL < 0.40: hairstructure = "straight "
        elif 0.40 <= HAIRSTRUCTURE_VAL < 0.60: hairstructure = "wavy "
        elif 0.60 <= HAIRSTRUCTURE_VAL < 0.80: hairstructure = "big curls "
        else: hairstructure = "small curls "

        #set the statur
        if 0.00 <= STATUR_VAL < 0.20: statur = "very thin stature, "
        elif 0.20 <= STATUR_VAL < 0.40: statur = "thin stature, "
        elif 0.40 <= STATUR_VAL < 0.60: statur = "regular stature, "
        elif 0.60 <= STATUR_VAL < 0.80: statur = "obese stature, "
        else: statur = "very obese stature, "

        #set the nose, mouth, eyesize and ears
        if 0.00 <= NOSE_VAL < 0.25: nose = "no nose, "
        elif 0.25 <= NOSE_VAL < 0.50: nose = "small nose, "
        elif 0.50 <= NOSE_VAL < 0.75: nose = "medium-sized nose, "
        else: nose = "big nose, "
        
        if 0.00 <= MOUTH_VAL < 0.25: mouth = "no mouth, "
        elif 0.25 <= MOUTH_VAL < 0.50: mouth = "small mouth, "
        elif 0.50 <= MOUTH_VAL < 0.75: mouth = "medium-sized mouth, "
        else: mouth = "big mouth, "

        if 0.00 <= EYESIZE_VAL < 0.25: eyesize = "no "
        elif 0.25 <= EYESIZE_VAL < 0.50: eyesize = "small "
        elif 0.50 <= EYESIZE_VAL < 0.75: eyesize = "medium-sized "
        else: eyesize = "big "

        if 0.00 <= EARS_VAL < 0.25: ears= "no ears "
        elif 0.25 <= EARS_VAL < 0.50: ears= "small ears "
        elif 0.50 <= EARS_VAL < 0.75: ears= "medium-sized ears "
        else: ears= "big ears "

        #set the hairlength and convert it to two decimal places
        hairlength = "{:.2f}".format(HAIRLENGTH_VAL.item())

            # set the sugar
        if sugarcheck == True: 
            sugar = "(high detailed skin:1.2), 8k uhd, dslr,soft lighting, high quality, film grain"
        else:
            sugar = "by NHK Animation, digital art, trending on artstation, illustration"

        # set the prompts
        prompt = abstr + age + gender + "in front of grey background " + glasses + SKINCOLOR_VAL + " skin color, " + facewidth + facialhair + " face, " +  str(hairlength) + "m long " + hairstructure + HAIRCOLOR_VAL + " hair, " + statur + nose + mouth + eyesize + EYECOLOR_VAL + " eyes "+ "and " + ears + "is looking at the camera with a proud expression on the face, " + sugar

        return prompt



def generate_latentprompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL):
    openness= "(Openness:" + str(calculate_postivevalue("{:.2f}".format(OPEN_VAL.item()))) + ")"
    conscientiousness= "(Conscientiousness:" + str(calculate_postivevalue("{:.2f}".format(CON_VAL.item()))) + ")"
    extraversion= "(Extraversion:" + str(calculate_postivevalue("{:.2f}".format(EXTRA_VAL.item()))) + ")"
    agreeableness= "(Agreeableness:" + str(calculate_postivevalue("{:.2f}".format(AGREE_VAL.item()))) + ")"
    neuroticism= "(Neuroticism:" + str(calculate_postivevalue("{:.2f}".format(NEURO_VAL.item()))) + ")"
    acceptance= "(Acceptance:" + str(calculate_postivevalue("{:.2f}".format(ACCEPT_VAL.item()))) + ")"
    likeability= "(Likeability:" + str(calculate_postivevalue("{:.2f}".format(LIKE_VAL.item()))) + ")"
    empathy= "(Empathy:" + str(calculate_postivevalue("{:.2f}".format(EMP_VAL.item()))) + ")"
    anthropomorphism= "(Anthropomorphism:" + str(calculate_postivevalue("{:.2f}".format(ANTHRO_VAL.item()))) + ")"
    trust= "(Trust:" + str(calculate_postivevalue("{:.2f}".format(TRUST_VAL.item()))) + ")"
    
    prompt = "A portrait captures a person exuding traits of " + openness + ", " + conscientiousness + ", " + extraversion + ", " + agreeableness + ", " + neuroticism + ", " + acceptance + ", " + likeability + ", " + empathy + ", " + anthropomorphism + ", " + trust + " , looking confidently at the camera with a proud expression, against a blue background"
    return prompt


def calculate_postivevalue(input_parameter):
    if float(input_parameter) > 0.5:
        result = "{:.2f}".format((float(input_parameter) - 0.5)/0.5)
    else:
        result = 0
    return result

def calculate_negativevalue(input_parameter):
    if float(input_parameter) < 0.5:
        result = "{:.2f}".format(1 - (float(input_parameter) * 2))
    else:
        result = 0
    return result


def generate_latent_negativePrompt(OPEN_VAL, CON_VAL, EXTRA_VAL, AGREE_VAL, NEURO_VAL, ACCEPT_VAL, LIKE_VAL, EMP_VAL, ANTHRO_VAL, TRUST_VAL):
    openness= "(Openness:" + str(calculate_negativevalue("{:.2f}".format(OPEN_VAL.item()))) + ")"
    conscientiousness= "(Conscientiousness:" + str(calculate_negativevalue("{:.2f}".format(CON_VAL.item()))) + ")"
    extraversion= "(Extraversion:" + str(calculate_negativevalue("{:.2f}".format(EXTRA_VAL.item()))) + ")"
    agreeableness= "(Agreeableness:" + str(calculate_negativevalue("{:.2f}".format(AGREE_VAL.item()))) + ")"
    neuroticism= "(Neuroticism:" + str(calculate_negativevalue("{:.2f}".format(NEURO_VAL.item()))) + ")"
    acceptance= "(Acceptance:" + str(calculate_negativevalue("{:.2f}".format(ACCEPT_VAL.item()))) + ")"
    likeability= "(Likeability:" + str(calculate_negativevalue("{:.2f}".format(LIKE_VAL.item()))) + ")"
    empathy= "(Empathy:" + str(calculate_negativevalue("{:.2f}".format(EMP_VAL.item()))) + ")"
    anthropomorphism= "(Anthropomorphism:" + str(calculate_negativevalue("{:.2f}".format(ANTHRO_VAL.item()))) + ")"
    trust= "(Trust:" + str(calculate_negativevalue("{:.2f}".format(TRUST_VAL.item()))) + ")"
    
    n_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature, black-white, " + openness + ", " + conscientiousness + ", " + extraversion + ", " + agreeableness + ", " + neuroticism + ", " + acceptance + ", " + likeability + ", " + empathy + ", " + anthropomorphism + ", " + trust
    return n_prompt

def generate_defined_negativePrompt():
    n_prompt= "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, anime:1.4) text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorlydrawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, badproportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms,missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, nsfw, inclined head, tilted head, two persons, text, symbol, logo, artist signature, black-white"
    return n_prompt