from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join


class HandsDataset(Dataset):
	def __init__(self, image_type, path, transform=None):
		
		self.image_type = image_type
		self.path = path
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		self.size = len(onlyfiles)//2
		self.feat_transform = transform['feature']
		self.targ_transform = transform['target']
		self.feat_prefix = "feature"
		self.targ_prefix = "mask"

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		
		idx = str(int(idx)) # why??
		images_ext = ".png" if self.image_type == "synth" else ".jpg"
		
		feature_img_name = os.path.join(self.path, self.feat_prefix + idx + images_ext)
		target_img_name = os.path.join(self.path, self.targ_prefix + idx + images_ext)
		feature_image = Image.open(feature_img_name)
		target_image = Image.open(target_img_name)
		feature_image = np.array(feature_image)
		target_image = np.array(target_image)
		
		if self.feat_transform:
			feature_image = self.feat_transform(feature_image)
		if self.targ_transform:
			target_image = self.targ_transform(target_image)

		return [feature_image, target_image]
