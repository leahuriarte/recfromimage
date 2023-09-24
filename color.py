from __future__ import print_function
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

NUM_CLUSTERS = 5

import math

tags = [["darkblue", "white"], ["brown", "pink"], ["yellow", "brown", "white", "orange"], ["blue", "darkblue", "white"], ["blue", "darkblue", "gray"], ["darkblue", "white"], ["yellow", "white", "orange"], ["brown", "white"], ["green", "white"], ["brown", "white"]]

def compare(location):    

    #print('reading image')
    im = Image.open(location)
    im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')

    colors = [ "#aa42f5", "#42a4f5", "#0a497d","#42f554", "#0c7d0a","#FFFF00", "#f59b42", "#FF0000", "#964B00", "#ffffff", "#cfcfcf"]
    color_names = [ "purple", "blue" , "darkblue", "green" ,"darkgreen", "yellow", " orange", "red", "brown", "white", "gray"]

    color_distance_list = []

    input_color = tuple(int(colour.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    for i in range (len(colors)):
        use_color = colors[i]
        my_color = tuple(int(use_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        get_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(my_color, input_color)])) 
        color_distance_list.append(get_distance)

    sorted_color_distance_list = min(color_distance_list)
    closest_hex = color_distance_list.index(sorted_color_distance_list)

    main_color = color_names[closest_hex]

    return main_color





