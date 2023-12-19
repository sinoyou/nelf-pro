import torch
import numpy as np 
from typing import List, Tuple, Union
from torchtyping import TensorType
from sklearn.cluster import KMeans

class FactorPoseGenerator:
    def __init__(self, strategy, return_type='index'):
        self.strategy = strategy
        self.return_type = return_type

    def sample(self, positions: Union[TensorType['num_cam', 3, 4], TensorType['num_cam', 3]], num_samples: int, **kwargs):
        assert positions.shape[0] >= num_samples, f'Number of cameras ({positions.shape[0]}) must be >= number of samples ({num_samples})'

        if positions.shape[-2:] == (3, 4):
            positions = positions[:, :3, 3]

        if self.return_type == 'index':
            if self.strategy == 'random':
                return self._sample_random(positions, num_samples, **kwargs)
            elif self.strategy == 'fps':
                return self._sample_fps(positions, num_samples, **kwargs)
            else:
                raise NotImplementedError(f'Camera sampling strategy {self.strategy} not implemented for return type {self.return_type}')
        elif self.return_type == 'position':
            if self.strategy == 'kmeans':
                return self._sample_kmeans(positions, num_samples, **kwargs)
            else:
                raise NotImplementedError(f'Camera sampling strategy {self.strategy} not implemented for return type {self.return_type}')
        else:
            raise NotImplementedError(f'Unknown return type {self.return_type}')
        
    @staticmethod
    def get_random_offset(shape: torch.Size, scale: float=0.1, seed: int=1737) -> torch.Tensor:
        torch.manual_seed(seed=seed)
        return torch.randn(shape) * scale
    
    def _sample_random(self, positions, num_samples):
        np.random.seed(1737)
        indices = np.random.choice(range(positions.shape[0]), size=num_samples, replace=False)
        return indices

    def _sample_fps(self, positions, num_samples, return_order=False):
        """Iteratively remove points (views) with the minimum distance to its closest neighbor."""
        mink = 1
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions)

        points = positions
        n = len(points)
        if n < num_samples:
            print("INVALID. n_points_to_sample must be smaller than points length!")
            return None
        else:
            current_subset = len(points)

        # compute pairwise distances
        A = points.unsqueeze(0)  # 1 x n x 3
        B = points.unsqueeze(1)  # n x 1 x 3
        pairwise_distances_grid = torch.norm(A - B, dim=-1)  # n x n
        max_distance = pairwise_distances_grid.max()

        # distances on the diagonal are zero, set them to the maximum
        pairwise_distances_grid[pairwise_distances_grid == 0.0] = max_distance

        removal_order = []
        while current_subset != num_samples:
            # flat_index = torch.argmin(pairwise_distances_grid, keepdim=True)
            # min_y = torch.div(flat_index, n, rounding_mode="trunc")
            partitionk = mink if mink > 1 else 2
            mink_vals, mink_idx = torch.topk(pairwise_distances_grid, partitionk, largest=False, dim=0)
            minavg_vals = mink_vals.mean(dim=0) if mink > 1 else mink_vals
            if (minavg_vals == np.inf).all():
                minavg_vals, mink_idx = torch.topk(pairwise_distances_grid, 1, largest=False, dim=0)
            min_y = torch.argmin(minavg_vals, keepdim=True)

            # check for a better order between A=(x,min_y) and B=(min_y,x) and their second closest points
            if mink == 1:
                x = mink_idx[0, min_y]
                A = mink_vals[0, min_y]
                B = mink_vals[0, x]
                assert A == B

                if mink_vals[1, min_y] > mink_vals[1, x]:
                    min_y = x

            pairwise_distances_grid[:, min_y] = np.inf
            pairwise_distances_grid[min_y, :] = np.inf
            removal_order.append(min_y.item())
            current_subset -= 1

        mask = pairwise_distances_grid != np.inf

        select_index = torch.nonzero(torch.sum(mask, dim=0)).squeeze().numpy()

        if not return_order:
            return select_index
        else:
            return select_index, removal_order
    
    def _sample_kmeans(self, positions, num_samples):
        kmeans = KMeans(n_clusters=num_samples, random_state=1737, init='k-means++').fit(positions)
        return torch.from_numpy(kmeans.cluster_centers_)