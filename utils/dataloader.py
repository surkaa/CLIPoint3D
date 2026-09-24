import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import os
from pathlib import Path
import random
from PIL import Image
import json
from utils.data_utils import normal_pc, rotation_point_cloud, jitter_point_cloud
import sys
import h5py
import glob
import open3d as o3d

def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def list_class_names(root_dir):
    """Return sorted class directory names, ignoring metadata files."""
    root_dir = Path(root_dir)
    return sorted(path.name for path in root_dir.iterdir() if path.is_dir())


class PointDA(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seed=42):
        """
        Args:
            root_dir (str): Path to ModelNet10 dataset
            split (str): 'train' or 'test'
            transform: Optional transform to be applied on a sample
            seed (int): Seed for random number generation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.classes = list_class_names(self.root_dir)

        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # Get all point cloud files
        self.files = []
        self.labels = []

        for class_name in self.classes:
            class_path = self.root_dir / class_name / split
            if not class_path.exists():
                continue

            for file in class_path.glob('*.npy'):  # Assuming .npy format
                self.files.append(file)
                self.labels.append(self.class_to_idx[class_name])

        # Randomly select 32 samples from each class if split is train
        if self.split == 'train':
            random.seed(seed)
            self.files = []
            self.labels = []
            for class_name in self.classes:
                class_files = list((self.root_dir / class_name / split).glob('*.npy'))
                if len(set([str(file).split('/')[-3] for file in class_files])) != 1:
                    print("Warning: Class names are not the same for all classes.")

                selected_files = random.sample(class_files, min(64, len(class_files)))
                self.files.extend(selected_files)
                self.labels.extend([self.class_to_idx[class_name]] * len(selected_files))

            print(f"Number of files: {len(self.files)}")
            print(f"Number of labels: {len(self.labels)}")

        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

        self.captions = json.load(open('PointDA_data/captions.json', 'r'))
        # print(self.captions)

        self.prefix_caption = 'A point cloud model of a {class} which has {description}' 


    def __len__(self):
        return len(self.files)

    def rotate_point_cloud(self, point_cloud):
        """ Randomly rotate a point cloud to augment the dataset
            rotation is per shape based along up direction
            Input:
            Nx3 tensor, original point cloud
            Return:
            Nx3 tensor, rotated point cloud
        """
        rotated_data = torch.zeros(point_cloud.shape, dtype=torch.float32)
        rotation_angle = torch.rand(1).item() * 2 * np.pi  # Generate a random rotation angle
        cosval = torch.cos(torch.tensor(rotation_angle))
        sinval = torch.sin(torch.tensor(rotation_angle))
        rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
        rotated_data = torch.matmul(point_cloud.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, point_cloud, sigma=0.01, clip=0.05):
        """ Randomly jitter points. Jittering is per point.
            Input:
            Nx3 tensor, original point cloud
            Return:
            Nx3 tensor, jittered point cloud
        """
        assert(clip > 0)
        jittered_data = torch.clamp(sigma * torch.randn(point_cloud.size()), -clip, clip)
        jittered_data += point_cloud
        return jittered_data


    def __getitem__(self, idx):

        file = self.files[idx]

        # Load point cloud
        point_cloud = np.load(file)
        label = self.labels[idx]

        # Convert to tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        point_cloud = point_cloud - point_cloud.mean(dim=0, keepdim=True)  # center
        dist = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=1)))  # calculate max distance
        point_cloud = point_cloud / dist  # scale

        point_cloud = self.jitter_point_cloud(self.rotate_point_cloud(point_cloud))
        label = torch.tensor(label)

        return point_cloud, label
    
        # point_cloud = normal_pc(point_cloud)
        # point_cloud = jitter_point_cloud(rotation_point_cloud(point_cloud))
        # point_cloud = torch.from_numpy(point_cloud).float()

class GraspNet(Dataset):
    def __init__(self, root_dir, split='train', data_type='Real', camera='kinect', transform=None, seed=42):
        """
        Args:
            root_dir (str): Path to GraspNet dataset
            split (str): 'train' or 'test'  
            data_type (str): 'Real' or 'Synthetic'
            camera (str): 'kinect' or 'realsense'
            transform: Optional transform to be applied on a sample
            seed (int): Seed for random number generation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.data_type = data_type
        self.camera = camera
        self.transform = transform
        
        # Get all point cloud files
        self.files = []
        self.labels = []
        self.file_paths = []
        
        # For Synthetic data, combine both kinect and realsense
        if data_type == 'Synthetic':
            cameras = ['kinect', 'realsense']
        else:
            cameras = [camera]
        
        # Collect all class directories across cameras to get class names (following PointDA pattern)
        all_class_names = set()
        for cam in cameras:
            base_path = self.root_dir / split / data_type / cam
            if base_path.exists():
                class_names = [d.name for d in sorted(base_path.glob('*')) if d.is_dir()]
                all_class_names.update(class_names)
        
        # Sort class names alphabetically (same as PointDA)
        self.classes = sorted(list(all_class_names))
        self.num_classes = len(self.classes)
        
        # Create class to index mapping (same pattern as PointDA)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Load data from all specified cameras
        for cam in cameras:
            base_path = self.root_dir / split / data_type / cam
            
            if not base_path.exists():
                print(f'Base path does not exist: {base_path}')
                continue
                
            # Iterate through scene directories
            for scene_dir in sorted(base_path.glob('*')):
                if not scene_dir.is_dir():
                    print('Scene directory is not a directory')
                    continue
                    
                # Extract class label from folder name  
                class_name = scene_dir.name
                class_label = self.class_to_idx[class_name]
                    
                # Get all .npy files in this scene
                for npy_file in sorted(scene_dir.glob('*.xyz')):
                    self.files.append(npy_file)
                    self.file_paths.append(str(npy_file))
                    self.labels.append(class_label)
        
        # Randomly select 32 samples from each class if split is train (same as PointDA)
        if self.split == 'train':
            random.seed(seed)
            
            # Group files by class
            from collections import defaultdict
            class_to_files = defaultdict(list)
            for file_path, label in zip(self.files, self.labels):
                class_name = self.idx_to_class[label]
                class_to_files[class_name].append((file_path, label))
            
            # Sample 32 files per class
            self.files = []
            self.labels = []
            self.file_paths = []
            
            for class_name in self.classes:
                class_files = class_to_files[class_name]
                if len(class_files) == 0:
                    print(f"Warning: No files found for class {class_name}")
                    continue
                    
                # Sample min(64, available_files) for this class
                selected_files = random.sample(class_files, min(64, len(class_files)))
                
                for file_path, label in selected_files:
                    self.files.append(file_path)
                    self.labels.append(label)
                    self.file_paths.append(str(file_path))
                
            
            print(f"Total files: {len(self.files)}")
            print(f"Total labels: {len(self.labels)}")

        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.files)
    
    def rotate_point_cloud(self, point_cloud):
        """ Randomly rotate a point cloud to augment the dataset
            rotation is per shape based along up direction
            Input:
            Nx3 tensor, original point cloud
            Return:
            Nx3 tensor, rotated point cloud
        """
        rotated_data = torch.zeros(point_cloud.shape, dtype=torch.float32)
        rotation_angle = torch.rand(1).item() * 2 * np.pi  # Generate a random rotation angle
        cosval = torch.cos(torch.tensor(rotation_angle))
        sinval = torch.sin(torch.tensor(rotation_angle))
        rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
        rotated_data = torch.matmul(point_cloud.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, point_cloud, sigma=0.01, clip=0.05):
        """ Randomly jitter points. Jittering is per point.
            Input:
            Nx3 tensor, original point cloud
            Return:
            Nx3 tensor, jittered point cloud
        """
        assert(clip > 0)
        jittered_data = torch.clamp(sigma * torch.randn(point_cloud.size()), -clip, clip)
        jittered_data += point_cloud
        return jittered_data

    def __getitem__(self, idx):
        file = self.files[idx]
        
        # Load point cloud from .npy file
        try:
            point_cloud = np.load(file)
        except Exception as e:
            # Fallback to .xyz file if .npy fails
            xyz_file = o3d.io.read_point_cloud(str(file))
            point_cloud = np.asarray(xyz_file.points)
            
        label = self.labels[idx]
        
        # Convert to tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        
            
        point_cloud = point_cloud - point_cloud.mean(dim=0, keepdim=True)  # center
        dist = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=1)))  # calculate max distance
        point_cloud = point_cloud / dist  # scale

        # Apply augmentations
        point_cloud = self.jitter_point_cloud(self.rotate_point_cloud(point_cloud))
        label = torch.tensor(label)
        
        return point_cloud, label



class Scannet(Dataset):
    def __init__(self, root_dir, split='train', transform=None, seed=42):
        """
        Args:
            root_dir (str): Path to ModelNet10 dataset
            split (str): 'train' or 'test'
            transform: Optional transform to be applied on a sample
            seed (int): Seed for random number generation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.classes = list_class_names(self.root_dir.parent / 'modelnet')
        
        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        # Get all point cloud files

        files = []
        labels = []
        if self.split == 'train':
            data_pth = load_dir(root_dir, name='train_files.txt')
        else:
            data_pth = load_dir(root_dir, name='test_files.txt')
        
        for pth in data_pth:
            data_file = h5py.File(pth, 'r')
            point = data_file['data'][:]
            label = data_file['label'][:]
            
            # idx = [index for index, value in enumerate(list(label)) if value in self.label_map]
            # point_new = point[idx]
            # label_new = np.array([self.label_map.index(value) for value in label[idx]])
            
            files.append(point)
            labels.append(label)

        self.files = np.concatenate(files, axis=0)
        self.labels = np.concatenate(labels, axis=0)

        from collections import defaultdict
        label_to_files = defaultdict(list)

        for file, label in zip(self.files, self.labels):
            label_to_files[self.idx_to_class[int(label)]].append(file)

        # Randomly select 32 samples from each class if split is train
        if self.split == 'train':
            random.seed(seed)
            self.files = []
            self.labels = []
            for class_name in self.classes:
                class_files = label_to_files[class_name]
                selected_files = random.sample(class_files, min(64, len(class_files)))
                self.files.extend(selected_files)
                self.labels.extend([self.class_to_idx[class_name]] * len(selected_files))

            print(f"Number of files: {len(self.files)}")
            print(f"Number of labels: {len(self.labels)}")

        # Default transforms if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        
        self.captions = json.load(open('PointDA_data/captions.json', 'r'))
        # print(self.captions)

        self.prefix_caption = 'A point cloud model of a {class} which has {description}' 


    def __len__(self):
        return len(self.files)
    
    def rotate_point_cloud(self, point_cloud):
        """ Randomly rotate a point cloud to augment the dataset
            rotation is per shape based along up direction
            Input:
            Nx3 tensor, original point cloud
            Return:
            Nx3 tensor, rotated point cloud
        """
        rotated_data = torch.zeros(point_cloud.shape, dtype=torch.float32)
        rotation_angle = torch.rand(1).item() * 2 * np.pi  # Generate a random rotation angle
        cosval = torch.cos(torch.tensor(rotation_angle))
        sinval = torch.sin(torch.tensor(rotation_angle))
        rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                        [0, 1, 0],
                                        [-sinval, 0, cosval]])
        rotated_data = torch.matmul(point_cloud.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, point_cloud, sigma=0.01, clip=0.05):
        """ Randomly jitter points. Jittering is per point.
            Input:
            Nx3 tensor, original point cloud
            Return:
            Nx3 tensor, jittered point cloud
        """
        assert(clip > 0)
        jittered_data = torch.clamp(sigma * torch.randn(point_cloud.size()), -clip, clip)
        jittered_data += point_cloud
        return jittered_data

    def __getitem__(self, idx):
        
        # Load point cloud
        point_cloud = self.files[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        point_cloud = torch.from_numpy(point_cloud).float()
        point_cloud = point_cloud - point_cloud.mean(dim=0, keepdim=True)  # center
        dist = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=1)))  # calculate max distance
        point_cloud = point_cloud / dist  # scale

        point_cloud = self.jitter_point_cloud(self.rotate_point_cloud(point_cloud))
        label = torch.tensor(label)
        
        return point_cloud, label

