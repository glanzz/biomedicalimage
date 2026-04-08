"""
Data preprocessing utilities for PMC-OA dataset
Supports both sequential and parallel processing with Dask
"""

import os
import time
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

class ImagePreprocessor:
    """Sequential image preprocessing"""

    def __init__(
        self,
        image_size: int = 224,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.image_size = image_size
        self.mean = np.array(normalize_mean)
        self.std = np.array(normalize_std)

    def preprocess_single_image(self, image) -> np.ndarray:
        """
        Preprocess a single image

        Args:
            image: Either a PIL Image object or a file path (str)

        Returns:
            Preprocessed image as numpy array
        """
        # Handle both PIL Image objects and file paths
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize (using LANCZOS for high quality)
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.mean) / self.std

        return image_array

    def preprocess_batch_sequential(
        self,
        samples: List[Dict],
        output_dir: str,
        split_name: str
    ) -> Dict[str, any]:
        """Process batch sequentially and save"""

        os.makedirs(output_dir, exist_ok=True)

        processed_data = {
            'images': [],
            'captions': [],
            'image_ids': []
        }

        start_time = time.time()

        for i, sample in enumerate(tqdm(samples, desc=f"Processing {split_name} (sequential)")):
            try:
                # Preprocess image
                image = sample['image']
                processed_img = self.preprocess_single_image(image)

                processed_data['images'].append(processed_img)
                processed_data['captions'].append(sample['caption'])
                processed_data['image_ids'].append(sample.get('image_id', i))

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        elapsed_time = time.time() - start_time
        throughput = len(processed_data['images']) / elapsed_time if elapsed_time > 0 else 0

        # Convert to numpy arrays and save
        output_file = f"{output_dir}/{split_name}_processed.npz"
        np.savez_compressed(
            output_file,
            images=np.array(processed_data['images']),
            captions=np.array(processed_data['captions']),
            image_ids=np.array(processed_data['image_ids'])
        )

        metrics = {
            'split': split_name,
            'num_samples': len(processed_data['images']),
            'elapsed_time': elapsed_time,
            'throughput': throughput,
            'images_per_second': throughput,
            'output_file': output_file
        }

        return metrics


# Dask parallel preprocessing
import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar


class ParallelImagePreprocessor(ImagePreprocessor):
    """Dask-based parallel image preprocessing"""

    def __init__(self, n_workers: int = 8, threads_per_worker: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self.client = None

    def setup_cluster(self):
        """Initialize Dask cluster"""
        cluster = LocalCluster(
            n_workers=self.n_workers,
            threads_per_worker=self.threads_per_worker,
            memory_limit='4GB',
            dashboard_address=':8787',
            silence_logs=True
        )
        self.client = Client(cluster)
        print(f"Dask cluster initialized with {self.n_workers} workers")
        print(f"Dashboard: {self.client.dashboard_link}")

    def close_cluster(self):
        """Close Dask cluster"""
        if self.client:
            self.client.close()
            self.client = None

    def process_partition(self, partition: List[Dict]) -> List[Dict]:
        """Process a partition of samples"""
        processed = []

        for sample in partition:
            try:
                image = sample['image']
                processed_img = self.preprocess_single_image(image)

                processed.append({
                    'image': processed_img,
                    'caption': sample['caption'],
                    'image_id': sample.get('image_id', 0)
                })
            except Exception as e:
                # Skip failed samples
                continue

        return processed

    def preprocess_batch_parallel(
        self,
        samples: List[Dict],
        output_dir: str,
        split_name: str,
        partition_size: int = 100
    ) -> Dict[str, any]:
        """Process batch in parallel using Dask"""

        if not self.client:
            self.setup_cluster()

        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()

        # Create Dask bag from samples
        bag = db.from_sequence(samples, partition_size=partition_size)

        # Process partitions in parallel
        processed_bag = bag.map_partitions(self.process_partition)

        # Compute with progress bar
        print(f"\nProcessing {len(samples)} images with {self.n_workers} workers...")
        with ProgressBar():
            processed_results = processed_bag.compute()

        # Flatten results
        all_images = []
        all_captions = []
        all_ids = []

        for partition_result in processed_results:
            for item in partition_result:
                all_images.append(item['image'])
                all_captions.append(item['caption'])
                all_ids.append(item['image_id'])

        elapsed_time = time.time() - start_time
        throughput = len(all_images) / elapsed_time if elapsed_time > 0 else 0

        # Save processed data
        output_file = f"{output_dir}/{split_name}_processed_parallel.npz"
        np.savez_compressed(
            output_file,
            images=np.array(all_images),
            captions=np.array(all_captions),
            image_ids=np.array(all_ids)
        )

        metrics = {
            'split': split_name,
            'num_samples': len(all_images),
            'elapsed_time': elapsed_time,
            'throughput': throughput,
            'n_workers': self.n_workers,
            'images_per_second': throughput,
            'output_file': output_file
        }

        return metrics


# Test functionality
if __name__ == "__main__":
    print("Testing preprocessing module...")

    # Create dummy image for testing
    from PIL import Image
    import numpy as np

    # Test sequential preprocessor
    print("\n1. Testing Sequential Preprocessor:")
    preprocessor = ImagePreprocessor()

    dummy_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    processed = preprocessor.preprocess_single_image(dummy_img)

    print(f"   Input shape: (512, 512, 3)")
    print(f"   Output shape: {processed.shape}")
    print(f"   Expected: (224, 224, 3)")
    assert processed.shape == (224, 224, 3), "Shape mismatch!"
    print("   ✓ Sequential preprocessor test passed!")

    # Test parallel preprocessor
    print("\n2. Testing Parallel Preprocessor:")
    parallel_preprocessor = ParallelImagePreprocessor(n_workers=2, threads_per_worker=1)

    # Create test samples
    test_samples = [
        {'image': Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)),
         'caption': f'Test caption {i}',
         'image_id': i}
        for i in range(10)
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = parallel_preprocessor.preprocess_batch_parallel(
            test_samples,
            tmpdir,
            'test',
            partition_size=5
        )
        print(f"   Processed {metrics['num_samples']} samples")
        print(f"   Throughput: {metrics['throughput']:.2f} images/sec")
        print(f"   ✓ Parallel preprocessor test passed!")

    parallel_preprocessor.close_cluster()

    print("\n✅ All preprocessing tests passed!")
