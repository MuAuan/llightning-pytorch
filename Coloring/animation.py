import os
import pickle
import numpy as np
import PIL.Image
import pandas as pd

s=32
images = []
s1=1
s0 =3 #100
sk=0
for j in range(0,s,1):
    try:
        im = PIL.Image.open('./YCC/preds0-31/image_preds{}'.format(j)+'.png')
        #im = PIL.Image.open('./hands/draw_results/data_plot'+str(i)+'.png')
        im =im.resize(size=(640, 478), resample=PIL.Image.NEAREST)
        images.append(im)
    except Exception as e:
        print(sk,e)
        sk += 1
        
print("finish", s, len(images))

images[0].save('./YCC/preds0-31/preds_.gif', save_all=True, append_images=images[1:s], duration=500*1, loop=0)  