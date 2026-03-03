import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
torch.set_float32_matmul_precision("high")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
 
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

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
device = "mps" if torch.backends.mps.is_available() else "cpu"
sam.to(device)
sam.float()
#sam.cuda()
mask_generator = SamAutomaticMaskGenerator(sam)
 
image_path = args.input
image_name = image_path.split(os.path.sep)[-1]
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = image.astype(np.float32, copy=False)

max_size = 1024  # maximum height or width
h, w = image.shape[:2]
scale = max_size / max(h, w)
if scale < 1:
    new_w, new_h = int(w * scale), int(h * scale)
    new_image = cv2.resize(image, (new_w, new_h))
new_image = new_image.astype(np.float32, copy=False)  
  
masks = mask_generator.generate(new_image)
for mask in new_image:
    seg = mask['segmentation']

    # Convert to NumPy if it's a tensor
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy()
    
    # Convert boolean → uint8 for cv2
    seg = seg.astype(np.uint8)

    # Resize
    seg_resized = cv2.resize(
        seg,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    # Convert back to boolean for visualization
    mask['segmentation'] = seg_resized.astype(bool)
plt.figure(figsize=(12, 9))
plt.imshow(image)
show_anns(new_image)
plt.axis('off')
plt.savefig(os.path.join('outputs', image_name), bbox_inches='tight')