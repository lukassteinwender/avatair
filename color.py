import webcolors

def colour_to_text(colour_val):
    closest_colour(colour_val)
    ##actual_name, closest_name = get_colour_name(colour_val)
    return closest_colour(colour_val)

## closest colour always equals actual colour, if actual colour exists. 
##In order to always get a colour text, we choose the closest colour.

def closest_colour(requested_colour): 
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name, spec="css3")
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]



##print(colour_to_text((230,90,30)))
