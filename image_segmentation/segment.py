import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

parser = argparse.ArgumentParser()
parser.add_argument(
   '--input',
   default='input/image_4.jpg'
)
args = parser.parse_args()

def show_anns(anns):
   if len(anns) == 0:
       return
   sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
   ax = plt.gca()
   ax.set_autoscale_on(False)
   for ann in sorted_anns:
       m = ann['segmentation']
       img = np.ones((m.shape[0], m.shape[1], 3))
       color_mask = np.random.random((1, 3)).tolist()[0]
       for i in range(3):
           img[:,:,i] = color_mask[i]
       np.dstack((img, m*0.35))
       ax.imshow(np.dstack((img, m*0.35)))

sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
 
image_path = args.input
image_name = image_path.split(os.path.sep)[-1]
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

max_size = 1024
h, w = image.shape[:2]
scale = max_size / max(h, w)
image_resized = cv2.resize(image, (int(w*scale), int(h*scale)))

masks = mask_generator.generate(image_resized)
plt.figure(figsize=(12, 9))
plt.imshow(image_resized)
show_anns(masks)
plt.axis('off')
plt.savefig(os.path.join('outputs', image_name), bbox_inches='tight')
