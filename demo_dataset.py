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


def generate_image_homogeneous_coordinates(fc, cc, image_width, image_height):
    homogeneous = np.zeros((image_height, image_width, 3))
    homogeneous[:, :, 2] = 1

    xx, yy = np.meshgrid([i for i in range(0, image_width)], [i for i in range(0, image_height)])
    homogeneous[:, :, 0] = (xx - cc[0]) / fc[0]
    homogeneous[:, :, 1] = (yy - cc[1]) / fc[1]

    return torch.from_numpy(homogeneous.astype(np.float32))

class DynamicDataset(Dataset):
    def __init__(self, usage, root, window: list, resize=2):
        super().__init__()

        self.root = root
        #TODO what is this?
        self.idx = range(10,300,10)
        self.data_len = len(self.idx)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, self.data_len))

        #TODO what is fc?
        self.fc = np.array([577.87061, 580.25851]) / resize
        #TODO what is cc?
        self.cc = np.array([319.87654, 239.87603]) / resize
        #TODO what is our image size? What should this be changed to?
        self.image_size = (640 // resize, 480 // resize)
        #TODO look into this function and what this does
        self.homogeneous_coords = generate_image_homogeneous_coordinates(
            self.fc, self.cc, *self.image_size).permute(2, 0, 1)

        self.window = np.array(window)

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def color2pose(color_info):
        pose_info = color_info.replace('color/frame', 'pose/frame').replace('.png', '.txt')
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
        color_info = os.path.join(self.root, 'color', 'frame_%06d.png' % self.idx[index])
        frame_index = self.idx[index]

        color_info2 = []
        if self.window[0] != 0:
            window = self.window
        else:
            raise NotImplementedError
        for delta in window:
            delta_info = color_info.replace(color_info[-10:-4], '%06d' % (frame_index + delta))
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

        depth_info = color_info.replace('color', 'depth').replace('png', 'pgm')
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

        predicted_normal_file = color_info.replace('color/frame', 'normal_pred/frame')
        normal_img = Image.open(predicted_normal_file)
        normal_values = 1 - np.asarray(normal_img).astype(np.float32) / 127.5

        if normal_values.shape[1] != self.image_size[0]:
            normal_values = cv2.resize(normal_values, self.image_size, interpolation=cv2.INTER_NEAREST)
        normal_tensor = self.to_tensor(normal_values)

        output = {'imagen': colorn, 'image': color, 'image2': color2,
                  'homo': self.homogeneous_coords, 'depth': depth_tensor,
                  'rots_ref_in_other': rots_ref_in_other, 'ts_ref_in_other': ts_ref_in_other,
                  'predicted_normal': normal_tensor,
                  'scene_path': '/'.join(color_info.split('/')[0:-2]), 'frame_index': frame_index}

        return output

    def __len__(self):
        return self.data_len


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


class NyuFlowDataset(Dataset):
    def __init__(self, root, usage, dataset_pickle_file, window: list, skip_every_n_image=1, resize=2):
        super(NyuFlowDataset, self).__init__()

        self.root = root
        self.max_depth = 7
        self.min_depth = 0.1

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]
        self.idx = range(0, len(self.data_info), skip_every_n_image)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, len(self)))

        self.fc = np.array([5.1885790117450188e+02, 5.1946961112127485e+02]) / resize
        # self.cc = np.array([3.2558244941119034e+02, 2.5373616633400465e+02]) / resize
        self.cc = np.array([159.5, 119.5])
        self.image_size = (640 // resize, 480 // resize)
        self.homogeneous_coords = generate_image_homogeneous_coordinates(
            self.fc, self.cc, *self.image_size).permute(2, 0, 1)

        self.window = np.array(window)

        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def color2pose(color_info):
        frame_id = int(color_info[-10:-4])
        pose_info = color_info.replace('color_%06d.png' % frame_id, '%d.txt' % frame_id).replace('color', 'pose')
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
        seq, name = self.data_info[self.idx[index]]
        color_info = os.path.join(self.root, seq, 'color', 'color_%s.png' % name)
        color_info2 = []
        if self.window[0] != 0:
            window = self.window
        else:
            # TODO: adaptive window
            window_file = color_info.replace('color/frame', 'frames_select/frames').replace('color.jpg', 'txt').replace(
                'dgx', 'oitstorage')
            if not os.path.isfile(window_file):
                return self.handle_bad_item()
            window = np.loadtxt(window_file, dtype=np.float32).astype(np.int32)

        frame_index = int(name)
        for delta in window:
            delta_info = os.path.join(self.root, seq, 'color', 'color_%06d.png' % (frame_index + delta))
            # if the next/previous image does not exist, use the previous/next image.
            if not os.path.isfile(delta_info):
                delta_info = os.path.join(self.root, seq, 'color', 'color_%06d.png' % (frame_index - delta))
                if not os.path.isfile(delta_info):
                    print('Not found:', delta_info)
                    return self.handle_bad_item()
            color_info2.append(delta_info)

        # color
        color, colorn = self.load_image(color_info, output_normalized_image=True)
        color2 = torch.stack([self.load_image(info) for info in color_info2])

        # depth
        depth_info = color_info.replace('color', 'depth')
        depth_img = Image.open(depth_info).convert('F')  # Convert to float32
        depth_img = depth_img.resize((320, 240), resample=Image.NEAREST)
        depth_tensor = torch.Tensor(np.array(depth_img)) / 5000.0
        crop = [20 // 2, 459 // 2, 24 // 2, 615 // 2]  # eigen crop
        depth_tensor[:crop[0]] = 0
        depth_tensor[crop[1]:] = 0
        depth_tensor[:, :crop[2]] = 0
        depth_tensor[:, crop[3]:] = 0
        depth_tensor[depth_tensor < self.min_depth] = 0
        depth_tensor[depth_tensor > self.max_depth] = 0
        depth_tensor = depth_tensor.view(1, depth_tensor.shape[0], depth_tensor.shape[1])

        # poses
        poses_info = [self.color2pose(info) for info in [color_info] + color_info2]
        for info in poses_info:
            if not os.path.isfile(info):
                print('Not found', info)
                return self.handle_bad_item()
        poses = [np.loadtxt(info, dtype=np.float32) for info in poses_info]
        for pose in poses:
            if len(pose[~np.isfinite(pose)]) > 0:
                print(color_info)
                return self.handle_bad_item()
        poses_ref_in_other = [np.matmul(np.linalg.inv(pose), poses[0]) for pose in poses[1:]]
        rots_ref_in_other = np.stack([pose[0:3, 0:3] for pose in poses_ref_in_other])
        ts_ref_in_other = np.stack([pose[0:3, 3] for pose in poses_ref_in_other])

        # normal prediction
        predicted_normal_file = color_info.replace('color', 'normal_pred')
        normal_img = Image.open(os.path.join(self.root, seq, 'normal_pred', predicted_normal_file))
        assert normal_img.width == 320
        assert normal_img.height == 240
        normal_values = 1 - np.asarray(normal_img).astype(np.float32) / 127.5
        normal_tensor = -self.to_tensor(normal_values) + 0.5

        output = {'imagen': colorn, 'image': color, 'image2': color2,
                  'depth': depth_tensor, 'homo': self.homogeneous_coords,
                  'rots_ref_in_other': rots_ref_in_other, 'ts_ref_in_other': ts_ref_in_other,
                  'predicted_normal': normal_tensor,
                  'scene_path': '/'.join(color_info.split('/')[0:-2]), 'frame_index': frame_index}

        return output

    def __len__(self):
        return len(self.idx)
