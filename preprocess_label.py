import glob
import random
import os
import numpy as np
import cv2
from PIL import Image
root = '../data'
modes = ['train', 'eval'] # train dataset and test dataset
for mode in modes:
    labels = sorted(glob.glob(os.path.join(root, mode, "labels") + "/*.*"))
    # os.path.join(root, mode, "train_label"),labels

    for label_path in labels:
        label_path      
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = np.array(img_B).astype("uint8")
        
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
        out_path = os.path.join(root, mode, "gray_label",photo_id+'.png')
        cv2.imwrite(out_path,img_B)
