"""
CPU-based baseline for biomedical image analysis

Implements parallel feature extraction + XGBoost classification baseline.
This provides a CPU-only comparison point against GPU-based VLM training.

Task: Image modality classification (X-ray, CT, MRI, microscopy, etc.)
Approach:
1. BiomedCLIP feature extraction (on CPU)
2. Parallel processing with joblib
3. XGBoost classification
4. Benchmark sequential vs parallel performance

Performance target: >10x speedup with 16+ cores
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib
from joblib import Parallel, delayed
import torch
from tqdm import tqdm
import json
import time
import os
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.vision_encoder import BiomedCLIPEncoder

logger = logging.getLogger(__name__)


class CPUBaseline:
    """CPU baseline with parallel feature extraction and XGBoost classification

    This class implements a CPU-only baseline for image analysis tasks.
    It demonstrates multi-core parallelism using joblib and compares
    sequential vs parallel feature extraction.

    Architecture:
    1. BiomedCLIP encoder (on CPU) extracts 512-dim features
    2. Features processed in parallel using joblib
    3. XGBoost classifier trained on features
    4. Classification task: image modality detection

    Args:
        n_jobs: Number of parallel workers for feature extraction and XGBoost
        device: Device to use ('cpu' or 'cuda')
    """

    def __init__(self, n_jobs: int = 16, device: str = 'cpu'):
        """Initialize CPU baseline

        Args:
            n_jobs: Number of parallel jobs (default: 16)
            device: Device for feature extraction (default: 'cpu')
        """
        self.n_jobs = n_jobs
        self.device = device

        logger.info(f"Initializing CPU baseline with {n_jobs} parallel jobs")

        # Initialize vision encoder on CPU
        logger.info("Loading BiomedCLIP encoder...")
        self.vision_encoder = BiomedCLIPEncoder(freeze=True)
        self.vision_encoder.model = self.vision_encoder.model.to(device)
        self.vision_encoder.model.eval()

        logger.info("Vision encoder loaded on CPU")

        # XGBoost classifier with parallel tree construction
        self.classifier = XGBClassifier(
            n_estimators=1000,      # 1000 trees
            max_depth=8,            # Max tree depth
            learning_rate=0.1,      # Learning rate
            tree_method='hist',     # Histogram-based algorithm (faster)
            n_jobs=n_jobs,          # Parallel tree construction
            random_state=42,
            subsample=0.8,          # Row sampling
            colsample_bytree=0.8,   # Column sampling
            eval_metric='mlogloss'
        )

        self.label_encoder = LabelEncoder()

        logger.info("CPU baseline initialized successfully")

    def extract_single_feature(self, image):
        """Extract features for a single image

        Args:
            image: PIL Image

        Returns:
            Feature vector (512-dim)
        """
        with torch.no_grad():
            # Preprocess image
            pixel_values = self.vision_encoder.processor(
                images=image,
                return_tensors="pt"
            )['pixel_values'].to(self.device)

            # Extract features
            features = self.vision_encoder(pixel_values)

        return features.cpu().numpy().flatten()

    def extract_features_sequential(self, images, desc="Extracting features"):
        """Extract features sequentially (single-core baseline)

        Args:
            images: List of PIL images
            desc: Progress bar description

        Returns:
            Tuple of (feature_matrix, metrics_dict)
        """
        logger.info(f"Sequential feature extraction: {len(images)} images")

        features = []
        start_time = time.time()

        for image in tqdm(images, desc=desc, disable=False):
            feat = self.extract_single_feature(image)
            features.append(feat)

        elapsed_time = time.time() - start_time
        throughput = len(images) / elapsed_time

        metrics = {
            'elapsed_time': elapsed_time,
            'throughput': throughput,
            'method': 'sequential',
            'n_jobs': 1,
            'num_images': len(images)
        }

        logger.info(f"Sequential: {elapsed_time:.2f}s, {throughput:.2f} images/sec")

        return np.array(features), metrics

    def extract_features_parallel(self, images, desc="Extracting features (parallel)"):
        """Extract features in parallel using joblib

        Uses joblib's Parallel with loky backend for true multi-process parallelism.
        This bypasses Python's GIL for CPU-bound tasks.

        Args:
            images: List of PIL images
            desc: Progress bar description

        Returns:
            Tuple of (feature_matrix, metrics_dict)
        """
        logger.info(f"Parallel feature extraction: {len(images)} images with {self.n_jobs} workers")

        start_time = time.time()

        # Use joblib for parallel processing
        # loky backend provides true multi-processing (not threads)
        with joblib.parallel_backend('loky', n_jobs=self.n_jobs):
            features = Parallel(verbose=10)(
                delayed(self.extract_single_feature)(img)
                for img in tqdm(images, desc=desc)
            )

        elapsed_time = time.time() - start_time
        throughput = len(images) / elapsed_time

        metrics = {
            'elapsed_time': elapsed_time,
            'throughput': throughput,
            'method': 'parallel',
            'n_jobs': self.n_jobs,
            'num_images': len(images)
        }

        logger.info(f"Parallel ({self.n_jobs} jobs): {elapsed_time:.2f}s, {throughput:.2f} images/sec")

        return np.array(features), metrics

    def infer_modality_labels(self, captions):
        """Infer image modality from captions using keyword matching

        Simple heuristic-based approach to assign modality labels
        based on caption content. This is used for CPU baseline
        classification task.

        Args:
            captions: List of image captions

        Returns:
            List of modality labels
        """
        labels = []

        modality_keywords = {
            'xray': ['x-ray', 'radiograph', 'chest', 'skeletal'],
            'ct': ['ct scan', 'computed tomography', 'ct image', 'axial'],
            'mri': ['mri', 'magnetic resonance', 't1', 't2', 'flair'],
            'microscopy': ['microscop', 'histolog', 'pathology', 'cell', 'tissue'],
            'ultrasound': ['ultrasound', 'sonograph', 'doppler'],
            'pet': ['pet scan', 'positron emission'],
            'endoscopy': ['endoscop', 'colonoscop'],
        }

        for caption in captions:
            caption_lower = caption.lower()

            assigned = False
            for modality, keywords in modality_keywords.items():
                if any(word in caption_lower for word in keywords):
                    labels.append(modality)
                    assigned = True
                    break

            if not assigned:
                labels.append('other')

        return labels

    def train(self, X_train, y_train):
        """Train XGBoost classifier

        Trains multi-class XGBoost classifier on extracted features.
        XGBoost parallelizes tree construction across n_jobs workers.

        Args:
            X_train: Training features (N x 512)
            y_train: Training labels

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training XGBoost with {self.n_jobs} parallel jobs...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Feature dimensions: {X_train.shape[1]}")

        start_time = time.time()

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        logger.info(f"Classes: {self.label_encoder.classes_}")
        logger.info(f"Class distribution: {np.bincount(y_train_encoded)}")

        # Train classifier
        self.classifier.fit(
            X_train,
            y_train_encoded,
            verbose=True
        )

        training_time = time.time() - start_time

        logger.info(f"Training completed in {training_time:.2f}s ({training_time/60:.2f}m)")

        return {
            'training_time': training_time,
            'num_samples': len(X_train),
            'num_features': X_train.shape[1],
            'num_classes': len(self.label_encoder.classes_),
            'classes': self.label_encoder.classes_.tolist()
        }

    def evaluate(self, X_test, y_test):
        """Evaluate classifier on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics dictionary
        """
        logger.info("Evaluating model...")

        y_test_encoded = self.label_encoder.transform(y_test)

        # Predict
        start_time = time.time()
        y_pred = self.classifier.predict(X_test)
        inference_time = time.time() - start_time

        # Metrics
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
        f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')

        report = classification_report(
            y_test_encoded,
            y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        logger.info(f"\nAccuracy: {accuracy:.4f}")
        logger.info(f"F1 (macro): {f1_macro:.4f}")
        logger.info(f"F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"Inference time: {inference_time:.3f}s")
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            y_test_encoded,
            y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'inference_time': inference_time,
            'inference_throughput': len(X_test) / inference_time,
            'classification_report': report
        }

    def save(self, path):
        """Save trained model

        Args:
            path: Path to save model
        """
        logger.info(f"Saving model to {path}...")

        joblib.dump({
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'n_jobs': self.n_jobs
        }, path)

        logger.info("Model saved successfully")

    def load(self, path):
        """Load trained model

        Args:
            path: Path to load model from
        """
        logger.info(f"Loading model from {path}...")

        data = joblib.load(path)
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        if 'n_jobs' in data:
            self.n_jobs = data['n_jobs']

        logger.info("Model loaded successfully")


def benchmark_cpu_baseline(
    dataset_path="data/raw/pmc_oa_10k",
    output_dir="outputs/baseline",
    n_jobs=16,
    train_samples=5000,
    test_samples=1000
):
    """Run CPU baseline experiments and benchmarks

    Comprehensive benchmarking of CPU baseline:
    1. Sequential vs parallel feature extraction
    2. XGBoost training
    3. Evaluation on test set
    4. Performance metrics collection

    Args:
        dataset_path: Path to dataset
        output_dir: Output directory
        n_jobs: Number of parallel workers
        train_samples: Number of training samples
        test_samples: Number of test samples

    Returns:
        Results dictionary
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    os.makedirs(output_dir, exist_ok=True)

    from datasets import load_from_disk

    print("="*80)
    print("CPU BASELINE BENCHMARK")
    print("="*80)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Train samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print("="*80)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(dataset_path)

    # Use subset for faster testing
    train_data = dataset['train'].select(range(min(train_samples, len(dataset['train']))))
    test_data = dataset['test'].select(range(min(test_samples, len(dataset['test']))))

    train_images = [s['image'] for s in train_data]
    train_captions = [s['caption'] for s in train_data]

    test_images = [s['image'] for s in test_data]
    test_captions = [s['caption'] for s in test_data]

    logger.info(f"Loaded {len(train_images)} training images, {len(test_images)} test images")

    # Create baseline
    baseline = CPUBaseline(n_jobs=n_jobs)

    # Infer labels from captions
    logger.info("Inferring modality labels from captions...")
    train_labels = baseline.infer_modality_labels(train_captions)
    test_labels = baseline.infer_modality_labels(test_captions)

    unique_labels, counts = np.unique(train_labels, return_counts=True)
    logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")

    # Results storage
    results = {}

    # Benchmark 1: Sequential feature extraction (1000 images)
    print("\n" + "="*80)
    print("BENCHMARK 1: Sequential Feature Extraction")
    print("="*80)

    baseline_seq = CPUBaseline(n_jobs=1)
    X_sample_seq, metrics_seq = baseline_seq.extract_features_sequential(
        train_images[:min(1000, len(train_images))]
    )
    results['sequential'] = metrics_seq

    print(f"Time: {metrics_seq['elapsed_time']:.2f}s")
    print(f"Throughput: {metrics_seq['throughput']:.2f} images/sec")

    # Benchmark 2: Parallel feature extraction (full dataset)
    print("\n" + "="*80)
    print(f"BENCHMARK 2: Parallel Feature Extraction ({n_jobs} workers)")
    print("="*80)

    X_train, metrics_parallel_train = baseline.extract_features_parallel(train_images)
    X_test, metrics_parallel_test = baseline.extract_features_parallel(test_images)

    results['parallel_train'] = metrics_parallel_train
    results['parallel_test'] = metrics_parallel_test

    print(f"Train time: {metrics_parallel_train['elapsed_time']:.2f}s")
    print(f"Train throughput: {metrics_parallel_train['throughput']:.2f} images/sec")
    print(f"Test time: {metrics_parallel_test['elapsed_time']:.2f}s")
    print(f"Test throughput: {metrics_parallel_test['throughput']:.2f} images/sec")

    # Calculate speedup
    # Extrapolate sequential time for full dataset
    seq_time_per_image = metrics_seq['elapsed_time'] / metrics_seq['num_images']
    estimated_seq_time_full = seq_time_per_image * len(train_images)
    actual_parallel_time = metrics_parallel_train['elapsed_time']

    speedup = estimated_seq_time_full / actual_parallel_time
    efficiency = (speedup / n_jobs) * 100

    results['speedup'] = speedup
    results['efficiency'] = efficiency

    print(f"\nEstimated sequential time (full): {estimated_seq_time_full:.2f}s")
    print(f"Actual parallel time (full): {actual_parallel_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")

    # Benchmark 3: XGBoost training
    print("\n" + "="*80)
    print("BENCHMARK 3: XGBoost Training")
    print("="*80)

    train_metrics = baseline.train(X_train, train_labels)
    results['training'] = train_metrics

    # Benchmark 4: Evaluation
    print("\n" + "="*80)
    print("BENCHMARK 4: Model Evaluation")
    print("="*80)

    eval_metrics = baseline.evaluate(X_test, test_labels)
    results['evaluation'] = eval_metrics

    # Save results
    results_file = f"{output_dir}/cpu_baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Save model
    model_file = f"{output_dir}/cpu_baseline_model.joblib"
    baseline.save(model_file)

    logger.info(f"Model saved to {model_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Sequential throughput: {metrics_seq['throughput']:.2f} images/sec")
    print(f"Parallel throughput: {metrics_parallel_train['throughput']:.2f} images/sec")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Efficiency: {efficiency:.1f}%")
    print(f"Training time: {train_metrics['training_time']:.2f}s")
    print(f"Test accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {eval_metrics['f1_macro']:.4f}")
    print("="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CPU Baseline Benchmark')
    parser.add_argument('--dataset_path', type=str, default='data/raw/pmc_oa_10k',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='outputs/baseline',
                       help='Output directory')
    parser.add_argument('--n_jobs', type=int, default=16,
                       help='Number of parallel jobs')
    parser.add_argument('--train_samples', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=1000,
                       help='Number of test samples')

    args = parser.parse_args()

    benchmark_cpu_baseline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        train_samples=args.train_samples,
        test_samples=args.test_samples
    )
