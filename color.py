import webcolors

def colour_to_text(colour_val):
    closest_colour(colour_val)
    ##actual_name, closest_name = get_colour_name(colour_val)
    return closest_colour(colour_val)

## closest colour always equals actual colour, if actual colour exists. 
##In order to always get a colour text, we choose the closest colour.

def closest_colour(requested_colour): 
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]



##print(colour_to_text((230,90,30)))