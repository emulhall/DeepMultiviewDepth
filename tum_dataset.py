import logging
import os
import pickle

import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import cv2

from read_write_dense import read_array
from colmap_error import colmapError

class TUMDataset(Dataset):
    def __init__(self, usage, dataset_pickle_file, resize = 2):
        super().__init__()

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]
        logging.info('Number of frames for the usage {0} is {1}.'
               .format(usage, len(self.data_info[0])))

        self.image_size = (640 // resize, 480 // resize)
        self.to_tensor = transforms.ToTensor()

    def load_image(self, file, output_normalized_image=False):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(file)
        if image.width != self.image_size[0]:
            image = image.resize(self.image_size)
        if not output_normalized_image:
            img = np.array(image).astype(np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img
        else:
            imgn = self.to_tensor(image)
            img = np.array(image).astype(np.uint8)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            return img, imgn

    def __getitem__(self, index):
        color_file = self.data_info[0][index]
        color, colorn = self.load_image(color_file, output_normalized_image=True)

        # ground truth depth
        depth_gt_file = self.data_info[1][index]
        depth_gt_img = Image.open(depth_gt_file).convert('F')  # Convert to float32
        depth_gt_values = np.array(depth_gt_img.resize((320, 240), resample=Image.NEAREST))
        depth_gt_tensor = torch.Tensor(depth_gt_values) / 5000.0
        depth_gt_tensor = depth_gt_tensor.view(1, depth_tensor.shape[0], depth_tensor.shape[1])

        # normal prediction
        predicted_normal_file = self.data_info[3][index]
        normal_values = read_array(predicted_normal_file)
        if normal_values.shape != self.image_size:
            normal_values = cv2.resize(normal_values, self.image_size,
                                       interpolation=cv2.INTER_NEAREST)
        normal_tensor = self.to_tensor(normal_values)

        # depth prediction
        predicted_depth_file = self.data_info[2][self.idx[index]]
        pred_depth_values = read_array(predicted_depth_file) / 10.0
        if pred_depth_values.shape != self.image_size:
            pred_depth_values = cv2.resize(pred_depth_values, self.image_size,
                                           interpolation=cv2.INTER_NEAREST)
        min_depth, max_depth = np.percentile(pred_depth_values, [5, 95])
        outlier_mask = np.logical_or(pred_depth_values < min_depth,
                                     pred_depth_values > max_depth)
        pred_depth_values[pred_depth_values < min_depth] = min_depth
        pred_depth_values[pred_depth_values > max_depth] = max_depth
        pred_depth_tensor = torch.Tensor(np.array(pred_depth_values))

        # uncertainty prediction
        error = colmapError(pred_depth_values, 2, outlier_mask)
        error = torch.Tensor(error)        
    
        output = {'imagen': colorn, 'image': color,
                  'depth_gt': depth_gt_tensor,
                  'predicted_normal': normal_tensor,
                  'predicted_depth': predicted_depth_tensor,
                  'uncertainty': error}

        return output


    def __len__(self):
        return len(self.idx)
