# date: 2026-05-11
# description: Frequency Class for deepfake datasets.

import sys

import lmdb

sys.path.append('.')

import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from .albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']#

def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeFrequencyDataset(data.Dataset):
    """
    Deepfake datasets include frequency information.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            print(f"Training datasets: {dataset_list}")
            # Training data should be collected together for training
            image_list, label_list = [], []
            # Handle both string and list inputs for train_dataset
            if isinstance(dataset_list, str):
                dataset_list = [dataset_list]
            
            for one_data in dataset_list:
                print(f"Processing dataset: {one_data}")
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                
            # Balance the dataset unless the caller needs the complete frame
            # pool for an epoch-level sampler.
            if self.config.get('balance_train_data', True):
                image_list, label_list = self.balance_dataset(
                    image_list, label_list, ratio=self.config['training_data_ratio'])
            else:
                print(f"Training data balancing disabled; using all {len(image_list)} frame references.")
        elif mode == 'test':
            dataset_list = config['test_dataset']
            print(f"Testing datasets: {dataset_list}")
            # Testing data should be collected together for testing
            image_list, label_list = [], []
            
            # Handle both string and list inputs for test_dataset
            if isinstance(dataset_list, str):
                dataset_list = [dataset_list]
                
            for one_data in dataset_list:
                print(f"Processing dataset: {one_data}")
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
                
            # Preserve every evaluation reference when requested. The default
            # remains balanced to keep existing benchmark behavior unchanged.
            if config.get('balance_test_data', True):
                image_list, label_list = self.balance_dataset(image_list, label_list)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        
    def balance_dataset(self, image_list, label_list, ratio=1.0):
        """Balance the dataset to have 50% real samples and equal distribution of fake classes.
        Also applies max_samples_per_class limit if specified in config.
        
        Args:
            image_list: List of image paths
            label_list: List of corresponding labels
            
        Returns:
            balanced_images: Balanced list of image paths
            balanced_labels: Balanced list of labels
        """
        # Convert to numpy arrays for easier manipulation
        images = np.array(image_list)
        labels = np.array(label_list)
        
        # Log pre-balancing distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        dist_str = "\nPre-balancing distribution:"
        for label, count in zip(unique_labels, counts):
            class_name = "Real" if label == 0 else f"Fake_{label}"
            percentage = (count / total) * 100
            dist_str += f"\n  {class_name}: {count} ({percentage:.1f}%)"
        print(dist_str)
        
        # Get unique labels and their counts
        real_mask = labels == 0
        fake_labels = unique_labels[unique_labels != 0]

        # Get max_samples_per_class from config
        max_samples = self.config.get('max_samples_per_class', None)
        
        # For binary classification (just real vs fake_4)
        if len(fake_labels) == 1:
            # No real images
            if len(labels[real_mask])==0:
                # Target equal number of real and fake samples
                target_samples = len(labels[~real_mask])
                
                # Apply max_samples limit if specified
                if max_samples is not None:
                    target_samples = min(target_samples, max_samples)
                    
                # Sample fake images
                fake_indices = np.where(~real_mask)[0]
                if len(fake_indices) > target_samples:
                    fake_indices = np.random.choice(fake_indices, target_samples, replace=False)
                    
                balanced_indices = fake_indices
            
            else:
                # Target equal number of real and fake samples
                target_samples = min(len(labels[real_mask]), len(labels[~real_mask]))
                
                # Apply max_samples limit if specified
                if max_samples is not None:
                    target_samples = min(target_samples, max_samples)
                
                # Sample real images
                real_indices = np.where(real_mask)[0]
                if len(real_indices) > target_samples:
                    real_indices = np.random.choice(real_indices, target_samples, replace=False)
                    
                # Sample fake images
                fake_indices = np.where(~real_mask)[0]
                if len(fake_indices) > target_samples:
                    fake_indices = np.random.choice(fake_indices, target_samples, replace=False)
                    
                balanced_indices = np.concatenate([real_indices, fake_indices])
                
        # For multi-class classification
        else:
            # Calculate target sizes for perfect balance
            num_fake_classes = len(fake_labels)
            real_img_number = len(labels[labels == 0])
            target_per_class = min([
                real_img_number,  # number of real samples
                *[len(labels[labels == i]) for i in fake_labels]  # number of samples per fake class
            ])

            # When real image number is not enough, make sure the ratio is still k:1
            if real_img_number < num_fake_classes*target_per_class:
                target_per_class = int(real_img_number/num_fake_classes)

            # Apply max_samples limit if specified
            if max_samples is not None:
                target_per_class = min(target_per_class, max_samples)
            
            # Adjust the dataset scale
            target_per_class = int(target_per_class * ratio)

            # Sample real images
            real_indices = np.where(real_mask)[0]
            if len(real_indices) > target_per_class:
                real_indices = np.random.choice(real_indices, num_fake_classes*target_per_class, replace=False)
            
            # Sample fake images for each fake class
            fake_indices = []
            for fake_label in fake_labels:
                fake_mask = labels == fake_label
                fake_label_indices = np.where(fake_mask)[0]
                if len(fake_label_indices) > target_per_class:
                    fake_label_indices = np.random.choice(fake_label_indices, target_per_class, replace=False)
                fake_indices.extend(fake_label_indices)
            
            balanced_indices = np.concatenate([real_indices, fake_indices])
        
        # Shuffle the indices
        np.random.shuffle(balanced_indices)
        balanced_images = images[balanced_indices].tolist()
        balanced_labels = labels[balanced_indices]

        # Convert all fake labels (>0) to 1 if backbone is configured for binary classification
        if 'backbone_config' in self.config and self.config['backbone_config'].get('num_classes', 0) == 2:
            # label = 1 if label > 0 else 0
            if self.config['detect_novel']:
                balanced_labels[balanced_labels>=2]=2
            else:
                balanced_labels[balanced_labels>=1]=1
        balanced_labels.tolist()
        
        # Log post-balancing distribution
        unique_labels, counts = np.unique(balanced_labels, return_counts=True)
        total = len(balanced_labels)
        dist_str = "\nPost-balancing distribution:"
        for label, count in zip(unique_labels, counts):
            class_name = "Real" if label == 0 else f"Fake_{label}"
            percentage = (count / total) * 100
            dist_str += f"\n  {class_name}: {count} ({percentage:.1f}%)"
        print(dist_str)
        
        return balanced_images, balanced_labels

    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks


    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.
        
        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize empty lists for image paths and labels
        image_list = []
        label_list = []
        name_list = []
        frame_path_list = []  # Added back the frame path list

        print(f"Loading dataset: {dataset_name}")

        # Handle DF40 datasets differently
        if dataset_name.startswith('DF40_'):
            if ('CDF_train' in dataset_name) or ('CDF_test_seen' in dataset_name):
                # Look up the json_split directory in a few canonical locations.
                split_root_env = os.environ.get('DF40_JSON_SPLIT_DIR')
                possible_paths = [
                    os.path.join(self.config['dataset_json_folder'].replace('json_file', 'json_split'), dataset_name + '.json'),
                    os.path.join(self.config['dataset_json_folder'].replace('json_used', 'json_split'), dataset_name + '.json'),
                ]
                if split_root_env:
                    possible_paths.append(os.path.join(split_root_env, dataset_name + '.json'))
                print(f"Trying paths: {possible_paths}")
                
                dataset_info = None
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                dataset_info = json.load(f)
                                # For CDF datasets, the key might be different from the filename
                                # Try to find a matching key
                                if dataset_name not in dataset_info:
                                    for key in dataset_info.keys():
                                        if ('CDF' in key and 
                                            ('train' in key and 'train' in dataset_name) or 
                                            ('test' in key and 'test' in dataset_name)):
                                            # Create a new dict with the correct key
                                            dataset_info = {dataset_name: dataset_info[key]}
                                            break
                                break
                        except Exception as e:
                            continue
                
                if dataset_info is None:
                    raise ValueError(f'dataset {dataset_name} not found in any of the expected locations: {possible_paths}')
            else:
                try:
                    with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                        dataset_info = json.load(f)
                except Exception as e:
                    print(e)
                    raise ValueError(f'dataset {dataset_name} not exist!')
        else:
            # For non-DF40 datasets, use the original dataset format
            try:
                with open(os.path.join(self.config['dataset_json_folder'], dataset_name + '.json'), 'r') as f:
                    dataset_info = json.load(f)
            except Exception as e:
                print(e)
                raise ValueError(f'dataset {dataset_name} not exist!')
        
        # Get the information for the current dataset
        print(self.config['dataset_json_folder'])
        for label in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label][self.mode]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if dataset_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++','DeepFakeDetection','FaceShifter','NO_real','NO_DF','NO_F2F','NO_FS','NO_NT']:
                sub_dataset_info = sub_dataset_info[self.compression]
            
            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                # Unique video name
                unique_video_name = video_info['label'] + '_' + video_name
                # Get the label and frame paths for the current video
                label = self.config['label_dict'][video_info['label']]
                frame_paths = video_info['frames']
                old_root = os.environ.get('DF40_OLD_ROOT')
                new_root = os.environ.get('DF40_DATA_ROOT')
                if old_root and new_root:
                    frame_paths = [p.replace(old_root, new_root) for p in frame_paths]


                """
                Some frames in training set may be enmpty
                """
                if len(frame_paths)==0:
                    continue

                # sorted video path to the lists
                if "_FF_" in dataset_name:
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                    else:
                        # Handle different file naming patterns
                        if any('seed' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('seed', '').split('.')[0]))
                        elif any('sample-' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('sample-', '').split('.')[0]))
                        elif any('MidJourney' in path for path in frame_paths) or any('CollabDiff' in path for path in frame_paths):
                            frame_paths = frame_paths  # These don't need sorting
                        else:
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                elif "_CDF_" in dataset_name:
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                    else:
                        # Handle different file naming patterns
                        if any(label in path for path in frame_paths for label in ['ddim', 'pixart', 'sd2.1']):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                        elif any('seed' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('seed', '').split('.')[0]))
                        elif any('sample-' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('sample-', '').split('.')[0]))
                        elif any('MidJourney' in path for path in frame_paths) or any('CollabDiff' in path for path in frame_paths):
                            frame_paths = frame_paths  # These don't need sorting
                        else:
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                elif "_mixed_" in dataset_name:
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                    else:
                        # Handle different file naming patterns
                        if any(label in path for path in frame_paths for label in ['ddim', 'pixart', 'sd2.1',]):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                        elif any('seed' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('seed', '').split('.')[0]))
                        elif any('sample-' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('sample-', '').split('.')[0]))
                        elif any('MidJourney' in path for path in frame_paths) or any('CollabDiff' in path for path in frame_paths):
                            frame_paths = frame_paths  # These don't need sorting
                        else:
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
                # elif "DSS" in dataset_name:
                #     frame_paths = frame_paths
                # elif "SBS" in dataset_name:
                #     frame_paths = frame_paths
                else:
                    if '\\' in frame_paths[0]:
                        frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                    else:
                        if any(label in path for path in frame_paths for label in ['ddim', 'pixart', 'sd2.1',]):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                        elif any('seed' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('seed', '').split('.')[0]))
                        elif any('sample-' in path for path in frame_paths):
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].replace('sample-', '').split('.')[0]))
                        elif any('MidJourney' in path for path in frame_paths) or any('CollabDiff' in path for path in frame_paths):
                            frame_paths = frame_paths  # These don't need sorting
                        elif any('OpenDataLab___ForgeryNet' in path for path in frame_paths):
                            frame_paths = frame_paths  # These don't need sorting
                        else:
                            frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))

                # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
                # In this case, we select self.frame_num frames from the original 270 frames
                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    if self.video_level:
                        # Select clip_size continuous frames
                        start_frame = random.randint(0, total_frames - self.frame_num)
                        frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
                    else:
                        # Select self.frame_num frames evenly distributed throughout the video
                        frame_indices = np.linspace(
                            0, total_frames - 1, num=self.frame_num, dtype=int
                        )
                        frame_paths = [frame_paths[i] for i in frame_indices]
                    total_frames = len(frame_paths)
                
                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError('clip_size must be specified when video_level is True.')
                    # Check if the number of total frames is greater than or equal to clip_size
                    if total_frames >= self.clip_size:
                        # Initialize an empty list to store the selected continuous frames
                        selected_clips = []

                        # Calculate the number of clips to select
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            # Calculate the step size between each clip
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)

                            # Select clip_size continuous frames from each part of the video
                            for i in range(num_clips):
                                # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                                start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1))
                                continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                                assert len(continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                                selected_clips.append(continuous_frames)

                        else:
                            start_frame = random.randrange(0, total_frames - self.clip_size + 1)
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(continuous_frames)==self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        # video name save
                        name_list.extend([unique_video_name] * len(selected_clips))

                    else:
                        print(f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")
                
                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    # video name save
                    name_list.extend([unique_video_name] * len(frame_paths))
            
        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, name_list = zip(*shuffled)
        
        return frame_path_list, label_list, name_list

     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution'] # if self.mode == "train" else self.config['resolution']

        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

        spectrum_images = []
        filter_images = []
        image = deepcopy(img)

        for i in range(3):  # R, G, B
            channel = image[:, :, i]

            # FFT magnitude spectrum
            f_shifted = np.fft.fftshift(np.fft.fft2(channel))
            magnitude_spectrum = np.log(np.abs(f_shifted) + 1e-3)
            magnitude_spectrum = cv2.normalize(
                magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
            )
            spectrum_images.append(np.uint8(magnitude_spectrum))

            # High-pass filter (channel - median blur)
            median_blurred = cv2.medianBlur(channel, 5)
            filter_images.append(cv2.subtract(channel, median_blurred))

        combined_spectrum = cv2.merge(spectrum_images)
        combined_filtered = cv2.merge(filter_images)

        return (
            Image.fromarray(np.array(img, dtype=np.uint8)),
            Image.fromarray(np.array(combined_spectrum, dtype=np.uint8)),
            Image.fromarray(np.array(combined_filtered, dtype=np.uint8)),
        )


    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size,3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark=self.rescale_landmarks(np.float32(landmark), original_size=256, new_size=self.config['resolution'])
        return landmark

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        frequency_map_tensors = []
        high_frequency_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image, freq_map, high_map = self.load_rgb(image_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load image for dataset sample index {index}, "
                    f"label {label}, path {image_path!r}: {e}"
                ) from e
            image = np.array(image)  # Convert to numpy array for data augmentation
            freq_map = np.array(freq_map)
            high_map = np.array(high_map)


            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)  

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                freq_map = self.normalize(self.to_tensor(freq_map))
                high_map = self.normalize(self.to_tensor(high_map))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            """
            need to unsqueeze the outside dimension
            """

            image_tensors.append(image_trans)
            frequency_map_tensors.append(freq_map)
            high_frequency_tensors.append(high_map)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            freq_tensors = frequency_map_tensors[0]
            high_tensors = high_frequency_tensors[0]

            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]
        return image_tensors, label, landmark_tensors, mask_tensors, freq_tensors, high_tensors, image_paths
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks, freq_tensors, high_tensors, path = zip(*batch)
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        freq_tensors = torch.stack(freq_tensors, dim=0)
        high_tensors = torch.stack(high_tensors, dim=0)

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        data_dict['freq'] = freq_tensors
        data_dict['high'] = high_tensors
        # data_dict['path'] = path
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


# if __name__ == "__main__":
#     with open('/training/config/detector/video_baseline.yaml', 'r') as f:
#         config = yaml.safe_load(f)
#     train_set = DeepfakeAbstractBaseDataset(
#                 config = config,
#                 mode = 'train', 
#             )
#     train_data_loader = \
#         torch.utils.data.DataLoader(
#             dataset=train_set,
#             batch_size=config['train_batchSize'],
#             shuffle=True, 
#             num_workers=0,
#             collate_fn=train_set.collate_fn,
#         )
#     from tqdm import tqdm
#     for iteration, batch in enumerate(tqdm(train_data_loader)):
#         if iteration > 10:
#             break
