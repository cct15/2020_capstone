import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def get_path(dir):
  file_list = []
  for file1 in os.listdir(dir):
    if (file1 != 'Additional Fish (Dead)') & (file1 != 'Red and Blue Petco Fish'):
      path1 = os.path.join(dir, file1)
      #print(path1, file1)
      if os.path.isdir(path1):
        for file2 in os.listdir(path1):
          path2 = os.path.join(path1, file2)
          # if file is a dir or an image
          if os.path.isdir(path2):       
            for file3 in os.listdir(path2):
              if file3.endswith('.CR3'):
                file_list.append((file3,file1,file2,os.path.join(path2,file3)))
          else:
              if file2.endswith('.CR3'): file_list.append((file2,file1,'NA',path2))
  file_list = np.array(file_list)
  print('Root:{}'.format(dir))
  print('Number of fish: {}, Number of pictures: {}'.format(len(np.unique(file_list[:,1])),file_list.shape[0]))
  return pd.DataFrame(file_list, columns = ['Image Name', 'Fish Number', 'Tag', 'path'])  
  #[(image name, fish number, if it has a tag like 'LB', image path)]

import rawpy
import imageio

# Read Raw Pictures. Return RGB values (numpy array)
def read_raw(path):
  try:
    with rawpy.imread(path) as raw:
      rgb = raw.postprocess()
    return rgb
  except ValueError:
    return None

from color_checker import colour_checkers_segmentation
import colour

# Extract color card from image. Return color card (numpy array)
def get_color_card(img):
  color_card, patches, color_checker = colour_checkers_segmentation(img)
  return color_card, patches, color_checker