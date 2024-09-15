import os
import numpy as np
from PIL import Image
import cv2

class PatchExtractor:
    def __init__(self, hr_dir, hr_patch_size, stride):
        self.hr_dir = hr_dir
        self.hr_patch_size = hr_patch_size
        self.lr2_patch_size = hr_patch_size // 2
        self.lr4_patch_size = hr_patch_size // 4
        self.stride = stride
        self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png') or f.endswith('.jpg')])

    def bicubic_downsample(self, image, scale):
        # Downsample the image using bicubic interpolation
        width, height = image.size
        new_size = (width // scale, height // scale)
        return image.resize(new_size, Image.BICUBIC)

    def extract_patches(self, hr_image):
        hr_patches = []
        lr2_patches = []
        lr4_patches = []

        hr_np = np.array(hr_image)
        lr2_np = np.array(self.bicubic_downsample(hr_image, 2))
        lr4_np = np.array(self.bicubic_downsample(hr_image, 4))

        h, w = hr_np.shape[:2]

        for i in range(0, h - self.hr_patch_size + 1, self.stride):
            for j in range(0, w - self.hr_patch_size + 1, self.stride):
                hr_patch = hr_np[i:i + self.hr_patch_size, j:j + self.hr_patch_size]

                lr2_i = i // 2
                lr2_j = j // 2
                lr2_patch = lr2_np[lr2_i:lr2_i + self.lr2_patch_size, lr2_j:lr2_j + self.lr2_patch_size]

                lr4_i = i // 4
                lr4_j = j // 4
                lr4_patch = lr4_np[lr4_i:lr4_i + self.lr4_patch_size, lr4_j:lr4_j + self.lr4_patch_size]

                if hr_patch.shape == (self.hr_patch_size, self.hr_patch_size, 3) and \
                   lr2_patch.shape == (self.lr2_patch_size, self.lr2_patch_size, 3) and \
                   lr4_patch.shape == (self.lr4_patch_size, self.lr4_patch_size, 3):
                    hr_patches.append(hr_patch)
                    lr2_patches.append(lr2_patch)
                    lr4_patches.append(lr4_patch)

        return hr_patches, lr2_patches, lr4_patches

    def save_patches(self, hr_patch_dir, lr2_patch_dir, lr4_patch_dir):
        if not os.path.exists(hr_patch_dir):
            os.makedirs(hr_patch_dir)
        if not os.path.exists(lr2_patch_dir):
            os.makedirs(lr2_patch_dir)
        if not os.path.exists(lr4_patch_dir):
            os.makedirs(lr4_patch_dir)

        for idx in range(len(self.hr_files)):
            hr_name = os.path.join(self.hr_dir, self.hr_files[idx])
            hr_image = Image.open(hr_name).convert("RGB")
            hr_patches, lr2_patches, lr4_patches = self.extract_patches(hr_image)

            base_name = os.path.splitext(self.hr_files[idx])[0]
            for patch_idx, (hr_patch, lr2_patch, lr4_patch) in enumerate(zip(hr_patches, lr2_patches, lr4_patches)):
                hr_patch_image = Image.fromarray(hr_patch)
                lr2_patch_image = Image.fromarray(lr2_patch)
                lr4_patch_image = Image.fromarray(lr4_patch)

                hr_patch_path = os.path.join(hr_patch_dir, f"{base_name}_hr_{patch_idx}.png")
                lr2_patch_path = os.path.join(lr2_patch_dir, f"{base_name}_lr2_{patch_idx}.png")
                lr4_patch_path = os.path.join(lr4_patch_dir, f"{base_name}_lr4_{patch_idx}.png")

                hr_patch_image.save(hr_patch_path)
                lr2_patch_image.save(lr2_patch_path)
                lr4_patch_image.save(lr4_patch_path)

# Example usage:
# hr_dir = 'path_to_hr_images'
# hr_patch_size = 256
# stride = 128
# hr_patch_dir = 'path_to_save_hr_patches'
# lr2_patch_dir = 'path_to_save_lr2_patches'
# lr4_patch_dir = 'path_to_save_lr4_patches'
# extractor = PatchExtractor(hr_dir, hr_patch_size, stride)
# extractor.save_patches(hr_patch_dir, lr2_patch_dir, lr4_patch_dir)