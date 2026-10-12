from typing import Iterator, Dict, Any, List

class BaseDataset:
    """Base class for dataset iterators."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def get_w2c_gps_list(self) -> List[Any]:
        """Returns w2c_gps for all frames. Needed for crop_lod_for_batch."""
        raise NotImplementedError

    def get_first_intrinsics(self) -> Dict[str, Any]:
        """Returns the intrinsic information for the first frame. Needed for warmup_gpu_rasterizer."""
        raise NotImplementedError
