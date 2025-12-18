from typing import Hashable, Mapping, Sequence

import torch
from monai.transforms import Flip, MapTransform


class CustomOrientationd(MapTransform):
    """Custom transform to handle image orientation and flipping."""
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.flipper = Flip(spatial_axis=1)  # Flip along the horizontal axis

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            if "00200020" in img.meta and "F" in img.meta["00200020"]["Value"][1]:
                # Flip the image horizontally
                img = self.flipper(img)
            d[key] = img
        return d


class CustomLoadd(MapTransform):
    """Custom transform to load PyTorch tensors."""
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, str]) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img_path = d[key]
            d[key] = torch.load(img_path, weights_only=True)
        return d


class CustomWindowd(MapTransform):
    """Custom transform for window-level adjustment of medical images."""
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def _get_window_center(self, img) -> torch.Tensor:
        return torch.tensor(
            img.meta["52009229"]["Value"][0]["00289132"]["Value"][0]["00281050"][
                "Value"
            ][0],
            dtype=torch.float32,
        )

    def _get_window_width(self, img) -> torch.Tensor:
        return torch.tensor(
            img.meta["52009229"]["Value"][0]["00289132"]["Value"][0]["00281051"][
                "Value"
            ][0],
            dtype=torch.float32,
        )

    def rescale_window(self, img, center, width):
        window_min = center - (width / 2)
        window_max = center + (width / 2)

        # Apply windowing
        windowed_image = torch.clamp(img, window_min, window_max)

        # Rescale between 0 and 1
        windowed_image = (windowed_image - window_min) / (window_max - window_min)

        # Clip values outside the range [0, 1] (just in case)
        windowed_image = torch.clamp(windowed_image, 0, 1)

        return windowed_image

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            try:
                center = self._get_window_center(img)
                width = self._get_window_width(img)
            except Exception as e:
                print(f"Error in windowing: {e}")
                center = 512
                width = 512
                print(f"Setting center and width to {center} and {width}")

            d[key] = self.rescale_window(img, center, width)
        return d


class RepeatChanneld(MapTransform):
    """Custom transform to repeat single channel to create pseudo RGB image."""
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            # Expand the single channel to create a pseudo RGB image
            if len(img.shape) == 3 and img.shape[0] == 1:
                img = img.expand(3, -1, -1)
            elif len(img.shape) == 4 and img.shape[0] == 1:
                img = img.expand(3, -1, -1, -1)
            else:
                raise ValueError(f"Unexpected shape: {img.shape}")
            d[key] = img
        return d


class AddChanneld(MapTransform):
    """Custom transform to add a channel dimension to the input."""
    def __init__(self, keys: Sequence[str], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key].unsqueeze(0)
        return d 