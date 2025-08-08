import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    CLASSES = ['background', 'sick']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # Extract specific categories from the labels.
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def save_images(i, image_vis, pr_mask, save_dir, file_prefix):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the visualization image
    vis_path = os.path.join(save_dir, f"{file_prefix}_image_vis_{i}.png")
    plt.imsave(vis_path, image_vis)

    # Save the ground truth mask image
    #gt_path = os.path.join(save_dir, f"{file_prefix}_gt_mask.png")
    #plt.imsave(gt_path, gt_mask, cmap='gray')

    # Save the predicted mask image
    pr_path = os.path.join(save_dir, f"{file_prefix}_pr_mask_{i}.png")
    plt.imsave(pr_path, pr_mask, cmap='gray')

    print(f"Images {i} saved at {save_dir}")

def segment_sick():
    DATA_DIR = './result'

    x_test_dir = DATA_DIR
    y_test_dir = DATA_DIR

    ENCODER = 'efficientnet-b7'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['sick']
    DEVICE = 'cpu'

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Load the best model.
    best_model = torch.load('./sick.pth', map_location=torch.device('cpu'))

    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )
    test_dataset_vis = Dataset(
        x_test_dir, y_test_dir,
        classes=CLASSES,
    )
    results = []
    for i in range(len(test_dataset)):
        image_vis = test_dataset_vis[i][0].astype('uint8')
        image, _ = test_dataset[i]

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        num_ones = np.count_nonzero(pr_mask == 1)
        results.append(num_ones)
        save_images(i, image_vis, pr_mask, save_dir='output/sick', file_prefix='sick')
    return results


if __name__ == '__main__':
    results = segment_sick()