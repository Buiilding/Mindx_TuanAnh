import os
from PIL import Image
import torch #operated by facebook with a lot of library
import torch.nn as nn
import inference_square_jpg
weight_path = 'C:/Users/tuana/SLC_PRO/Source_Code/best.pth'
from Classification_Model import Model
from PIL import Image
import sys
import torchvision.transforms as transforms
model = Model(27)
im_path = str(sys.argv[1])
uploaded_file= Image.open(im_path)
top_k_classes = inference_square_jpg.inference(model, weight_path, uploaded_file, 27)
names = ['0','1','2','3','4','5','6','7','8','9','NULL','a','b','bye','c','d','e','good','good morning','hello','little bit','no','pardon','please','project','whats up','yes']
predicted_class = names[top_k_classes[0]]
print(predicted_class)
