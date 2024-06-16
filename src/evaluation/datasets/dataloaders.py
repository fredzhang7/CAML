# Adapted from https://github.com/Tsingularity/FRN.

import sys
import torch
from torch.utils.data import Sampler
from PIL import Image

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from src.evaluation.datasets import samplers, transform_manager


from torchvision.datasets import ImageFolder

class TransformedImageFolder(ImageFolder):
    def __init__(self, root, is_training, transform_type, pre, **kwargs):
        self.is_training = is_training
        self.transform_type = transform_type
        self.pre = pre
        super().__init__(root, is_training, transform_type, pre, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = image_loader(path, self.is_training, self.transform_type, self.pre)
        return sample, target


def get_dataset(data_path, is_training, transform_type, pre):
    dataset = TransformedImageFolder(root=data_path, is_training=is_training, transform_type=transform_type, pre=pre)
    return dataset

def meta_test_dataloader(data_path, way, shot, pre, transform_type=None, query_shot=16, trial=1000):
    dataset = get_dataset(data_path=data_path, is_training=False, transform_type=transform_type, pre=pre)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=samplers.random_sampler(data_source=dataset, way=way, shot=shot, query_shot=query_shot, trial=trial),
        num_workers=3,
        pin_memory=False)

    return loader


def image_loader(path, is_training, transform_type, pre):
  p = Image.open(path)
  p = p.convert('RGB')

  if isinstance(transform_type, int):
    final_transform = transform_manager.get_transform(is_training=is_training, transform_type=transform_type, pre=pre)
  else:
    final_transform = transform_type

  p = final_transform(p)

  return p

