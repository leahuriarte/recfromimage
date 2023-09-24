from fastai.vision.widgets import *
from fastai.vision.all import *
from pathlib import Path
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import color

file_name='style.pkl'
learn_inference = load_learner(Path()/file_name)

#defining wallpaper class for testing
class Wallpaper():
    def __init__(self, id, color, style):
        #path to wallpaper
        self.id = id
        #color of wallpaper
        self.color = color
        #style of wallpaper
        self.style = style



categories = ['texture', 'geometric', 'stripe', 'nature']

#sample list of wallpapers
wallpapers = [Wallpaper("26", "green", "texture"), Wallpaper("24", "darkblue", "stripe"), Wallpaper("30", "gray", "texture"), Wallpaper("7", "darkblue", "geometric"), Wallpaper("31", "darkblue", "nature"), Wallpaper("22", "gray", "nature")]



def give_recommendation(image_path):
    #loading image
    location = image_path
    image = open(location, 'rb')

    #classifying style of image
    pred, pred_idx, probs = learn_inference.predict(PILImage.create(image))
    style = pred

    #classifying color of image
    image_color = color.compare(image_path)

    matches = []

    #checking if there are wallpapers with the same color and style
    for wallpaper in wallpapers:
        if wallpaper.color == image_color and wallpaper.style == style:
            matches.append(wallpaper)
        if len(matches) == 6:
            return matches
        
    #if there are not enough wallpapers with the same color and style, return wallpapers with the same style
    if len(matches) < 6:
        for wallpaper in wallpapers:
            if wallpaper.style == style:
                matches.append(wallpaper)
            if len(matches) == 6:
                return matches
            
    return matches

#test
print(give_recommendation("/Users/Leah/boho/images/test/1.jpg"))



     
        




