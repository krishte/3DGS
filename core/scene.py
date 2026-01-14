import json
import os

import numpy as np
import torch
from PIL import Image

from .camera import Camera
from .gaussian import GaussianSplats


class Scene(torch.nn.Module):
    def __init__(self, object):
        super().__init__()
        self.gaussian_splats = GaussianSplats(10000)
        self.cameras = torch.nn.ModuleList()

        scene_path = os.path.join("data", "nerf_synthetic", "nerf_synthetic", object)

        with open(os.path.join(scene_path, "transforms_train.json")) as f:
            meta = json.load(f)

        for frame in meta["frames"]:
            # Load image
            img_path = f"{scene_path}/{frame['file_path']}.png"
            image = torch.tensor(np.array(Image.open(img_path).convert("RGB"))) / 255.0

            # Create camera
            camera = Camera.from_nerf_json(
                width=800,
                height=800,
                fov_x=meta["camera_angle_x"],
                transform_matrix=frame["transform_matrix"],
                gt_image=image,
            )
            self.cameras.append(camera)
