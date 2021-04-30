import logging
import os
import random
import pickle

import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import skimage.io as sio
import cv2

from read_write_dense import read_array
from colmap_error import colmapError, edgeDetectionError


def generate_image_homogeneous_coordinates(fc, cc, image_width, image_height):
    homogeneous = np.zeros((image_height, image_width, 3))
    homogeneous[:, :, 2] = 1

    xx, yy = np.meshgrid([i for i in range(0, image_width)], [i for i in range(0, image_height)])
    homogeneous[:, :, 0] = (xx - cc[0]) / fc[0]
    homogeneous[:, :, 1] = (yy - cc[1]) / fc[1]

    return torch.from_numpy(homogeneous.astype(np.float32))

class DynamicDataset(Dataset):
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
        #TODO We might need to look at this scale
        depth_gt_tensor = torch.Tensor(depth_gt_values) / 100.0
        depth_gt_tensor = depth_gt_tensor.view(1, depth_gt_tensor.shape[0], depth_gt_tensor.shape[1])

        # normal prediction
        predicted_normal_file = self.data_info[3][index]
        normal_values = read_array(predicted_normal_file)
        if normal_values.shape != self.image_size:
            normal_values = cv2.resize(normal_values, self.image_size,
                                       interpolation=cv2.INTER_NEAREST)
        normal_tensor = self.to_tensor(normal_values)

        # depth prediction
        predicted_depth_file = self.data_info[2][index]
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
        img = np.array(Image.fromarray(color.permute(1,2,0).numpy(), "RGB").convert("L"))
        error = edgeDetectionError(pred_depth_values, 2, outlier_mask, img)
        #error = colmapError(pred_depth_values, 1, outlier_mask)
        error = torch.Tensor(error)
        #error = torch.Tensor(outlier_mask)

        # path to save final test data output
        path_split = depth_gt_file.split("/")
        output_path = os.path.join("results", path_split[-2], path_split[-1])
    
        output = {'imagen': colorn, 'image': color,
                  'depth_gt': depth_gt_tensor,
                  'predicted_normal': normal_tensor,
                  'predicted_depth': pred_depth_tensor[None, :, :],
                  'uncertainty': error[None, :, :],
                  'output_path': output_path}

        return output

    def __len__(self):
        return len(self.data_info[0])


class DemoFlowDataset(Dataset):
    def __init__(self, usage, root, window: list, resize=2):
        super().__init__()

        self.root = root
        self.idx = range(10, 300, 10)
        self.data_len = len(self.idx)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, self.data_len))

        self.fc = np.array([577.87061, 580.25851]) / resize
        self.cc = np.array([319.87654, 239.87603]) / resize
        self.image_size = (640 // resize, 480 // resize)
        self.homogeneous_coords = generate_image_homogeneous_coordinates(
            self.fc, self.cc, *self.image_size).permute(2, 0, 1)

        self.window = np.array(window)

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def color2pose(color_info):
        pose_info = color_info.replace('color/frame', 'pose/frame').replace('color.jpg', 'pose.txt')
        return pose_info

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

    def handle_bad_item(self):
        logging.info("Warning: Bad pose detected. Replace with a random data item.")
        index = random.randrange(len(self))
        return self[index]

    def __getitem__(self, index):
        color_info = os.path.join(self.root, 'color', 'frame-%06d.color.jpg' % self.idx[index])
        frame_index = self.idx[index]

        color_info2 = []
        if self.window[0] != 0:
            window = self.window
        else:
            raise NotImplementedError
        for delta in window:
            delta_info = color_info.replace(color_info[-16:-10], '%06d' % (frame_index + delta))
            # if the next/previous image does not exist, use the previous/next image.
            if not os.path.isfile(delta_info):
                delta_info = delta_info.replace('%06d' % (frame_index + delta), '%06d' % (frame_index - delta))
                if not os.path.isfile(delta_info):
                    return self.handle_bad_item()
            color_info2.append(delta_info)

        poses_info = [self.color2pose(info) for info in [color_info] + color_info2]

        # Open image
        color, colorn = self.load_image(color_info, output_normalized_image=True)
        color2 = torch.stack([self.load_image(info) for info in color_info2])

        depth_info = color_info.replace('color', 'depth').replace('jpg', 'pgm')
        depth_img = sio.imread(depth_info) / 1000.0
        if depth_img.shape[1] != self.image_size[0]:
            depth_img = cv2.resize(depth_img, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.tensor(depth_img).float()
        depth_tensor = depth_tensor.view(1, depth_tensor.shape[0], depth_tensor.shape[1])

        poses = [np.loadtxt(info, dtype=np.float32) for info in poses_info]
        for pose in poses:
            if len(pose[~np.isfinite(pose)]) > 0:
                print(color_info)
                return self.handle_bad_item()
        poses_ref_in_other = [np.matmul(np.linalg.inv(pose), poses[0]) for pose in poses[1:]]
        rots_ref_in_other = np.stack([pose[0:3, 0:3] for pose in poses_ref_in_other])
        ts_ref_in_other = np.stack([pose[0:3, 3] for pose in poses_ref_in_other])

        predicted_normal_file = color_info.replace('color/frame', 'normal_pred/frame').replace('.color.jpg',
                                                                                               '-normal_pred.png')
        normal_img = Image.open(predicted_normal_file)
        assert normal_img.width == 320
        assert normal_img.height == 240
        normal_values = 1 - np.asarray(normal_img).astype(np.float32) / 127.5
        normal_tensor = self.to_tensor(normal_values)

        output = {'imagen': colorn, 'image': color, 'image2': color2,
                  'homo': self.homogeneous_coords, 'depth': depth_tensor,
                  'rots_ref_in_other': rots_ref_in_other, 'ts_ref_in_other': ts_ref_in_other,
                  'predicted_normal': normal_tensor,
                  'scene_path': '/'.join(color_info.split('/')[0:-2]), 'frame_index': frame_index}

        return output

    def __len__(self):
        return self.data_len

