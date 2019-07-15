from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

class HandsDataset(Dataset):
	def __init__(self, image_type, path):

		self.image_type = image_type
		self.path = path
		onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
		self.size = len(onlyfiles)//2
		self.feat_prefix = "feature"
		self.targ_prefix = "mask"
		self.images_ext = ".png"

	def __len__(self):
		return self.size

	def transform(self, image, mask):
		# Resize
		resize = transforms.Resize(size=(282, 282))
		image = resize(image)
		mask = resize(mask)

		# Random crop
		i, j, h, w = transforms.RandomCrop.get_params(
		image, output_size=(256, 256))
		image = TF.crop(image, i, j, h, w)
		mask = TF.crop(mask, i, j, h, w)

		# Random horizontal flipping
		if random.random() > 0.5:
			image = TF.hflip(image)
			mask = TF.hflip(mask)

		# Random vertical flipping
		if random.random() > 0.5:
			image = TF.vflip(image)
			mask = TF.vflip(mask)

		# Transform to tensor
		image = TF.to_tensor(image)
		mask = TF.to_tensor(mask)
		image = TF.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		return image, mask

	def __getitem__(self, idx):

		idx = str(idx) # why??

		feature_img_name = os.path.join(self.path, self.feat_prefix + idx + self.images_ext)
		target_img_name = os.path.join(self.path, self.targ_prefix + idx + self.images_ext)
		feature_image = Image.open(feature_img_name)
		target_image = Image.open(target_img_name)

		feature_image, target_image = self.transform(feature_image, target_image)

		return [feature_image, target_image]
