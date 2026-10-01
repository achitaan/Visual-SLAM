"""A bounded image cache for full KITTI sequences."""
from collections import OrderedDict
from collections.abc import Sequence

import cv2 as cv
from kitti import image_paths


class ImageSequence(Sequence):
    def __init__(self, folder, max_frames=None, cache_size=4):
        self.paths = image_paths(folder, max_frames)
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._cache:
            image = cv.imread(str(self.paths[index]), cv.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read image: {self.paths[index]}")
            self._cache[index] = image
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]
