"""
Evaluation metrics for image captioning

Implements standard NLG metrics:
- BLEU (1-4): N-gram overlap with reference
- ROUGE-L: Longest common subsequence
- CIDEr: Consensus-based similarity (optional - requires pycocoevalcap)

These metrics evaluate generated captions against ground truth references.
"""

from typing import List, Dict, Tuple
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class CaptionMetrics:
    """Compute captioning evaluation metrics

    Supports BLEU, ROUGE-L, and optionally CIDEr metrics.
    Used to evaluate quality of generated captions against references.

    Example:
        >>> metrics = CaptionMetrics()
        >>> refs = ["A chest X-ray showing normal lungs"]
        >>> preds = ["Chest X-ray with normal lung fields"]
        >>> scores = metrics.compute_all(refs, preds)
        >>> print(scores['BLEU-4'])
    """

    def __init__(self, use_cider: bool = False):
        """Initialize metrics calculator

        Args:
            use_cider: Whether to compute CIDEr (requires pycocoevalcap)
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        self.use_cider = use_cider

        if use_cider:
            try:
                from pycocoevalcap.cider.cider import Cider
                self.cider_scorer = Cider()
                logger.info("CIDEr metric enabled")
            except ImportError:
                logger.warning("pycocoevalcap not installed, CIDEr disabled")
                self.use_cider = False
                self.cider_scorer = None
        else:
            self.cider_scorer = None

    def compute_bleu(
        self,
        references: List[List[str]],
        predictions: List[str]
    ) -> Dict[str, float]:
        """Compute BLEU scores (BLEU-1 through BLEU-4)

        BLEU measures n-gram overlap between predictions and references.
        Higher scores indicate better match with reference captions.

        Args:
            references: List of reference caption lists (each sample can have multiple refs)
            predictions: List of predicted captions

        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        # Tokenize
        refs_tokenized = [[ref.split() for ref in refs] for refs in references]
        preds_tokenized = [pred.split() for pred in predictions]

        # Compute BLEU-N scores
        bleu_scores = {}

        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            try:
                score = corpus_bleu(
                    refs_tokenized,
                    preds_tokenized,
                    weights=weights,
                    smoothing_function=self.smoothing
                )
                bleu_scores[f'BLEU-{n}'] = score
            except Exception as e:
                logger.warning(f"Error computing BLEU-{n}: {e}")
                bleu_scores[f'BLEU-{n}'] = 0.0

        return bleu_scores

    def compute_rouge(
        self,
        references: List[str],
        predictions: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE-L scores

        ROUGE-L measures longest common subsequence between
        predictions and references.

        Args:
            references: List of reference captions
            predictions: List of predicted captions

        Returns:
            Dictionary with ROUGE-L precision, recall, F1
        """
        scores = {
            'precision': [],
            'recall': [],
            'fmeasure': []
        }

        for ref, pred in zip(references, predictions):
            try:
                score = self.rouge_scorer.score(ref, pred)['rougeL']
                scores['precision'].append(score.precision)
                scores['recall'].append(score.recall)
                scores['fmeasure'].append(score.fmeasure)
            except Exception as e:
                logger.warning(f"Error computing ROUGE for sample: {e}")
                scores['precision'].append(0.0)
                scores['recall'].append(0.0)
                scores['fmeasure'].append(0.0)

        return {
            'ROUGE-L-P': np.mean(scores['precision']),
            'ROUGE-L-R': np.mean(scores['recall']),
            'ROUGE-L-F': np.mean(scores['fmeasure'])
        }

    def compute_cider(
        self,
        references: Dict[int, List[str]],
        predictions: Dict[int, List[str]]
    ) -> float:
        """Compute CIDEr score

        CIDEr (Consensus-based Image Description Evaluation) measures
        similarity using TF-IDF weighted n-gram matching.

        Args:
            references: Dict mapping sample ID to list of references
            predictions: Dict mapping sample ID to list of predictions

        Returns:
            CIDEr score (float)
        """
        if not self.use_cider or self.cider_scorer is None:
            logger.warning("CIDEr not available")
            return 0.0

        try:
            score, _ = self.cider_scorer.compute_score(references, predictions)
            return float(score)
        except Exception as e:
            logger.error(f"Error computing CIDEr: {e}")
            return 0.0

    def compute_all(
        self,
        references: List[str],
        predictions: List[str]
    ) -> Dict[str, float]:
        """Compute all available metrics

        Args:
            references: List of reference captions
            predictions: List of predicted captions

        Returns:
            Dictionary with all metric scores
        """
        logger.info(f"Computing metrics for {len(predictions)} samples...")

        metrics = {}

        # BLEU (with single reference per sample)
        try:
            bleu_scores = self.compute_bleu(
                [[ref] for ref in references],
                predictions
            )
            metrics.update(bleu_scores)
            logger.info(f"BLEU-4: {bleu_scores['BLEU-4']:.4f}")
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")

        # ROUGE
        try:
            rouge_scores = self.compute_rouge(references, predictions)
            metrics.update(rouge_scores)
            logger.info(f"ROUGE-L-F: {rouge_scores['ROUGE-L-F']:.4f}")
        except Exception as e:
            logger.error(f"Error computing ROUGE: {e}")

        # CIDEr (optional)
        if self.use_cider:
            try:
                refs_dict = {i: [ref] for i, ref in enumerate(references)}
                preds_dict = {i: [pred] for i, pred in enumerate(predictions)}
                cider_score = self.compute_cider(refs_dict, preds_dict)
                metrics['CIDEr'] = cider_score
                logger.info(f"CIDEr: {cider_score:.4f}")
            except Exception as e:
                logger.error(f"Error computing CIDEr: {e}")

        return metrics


def evaluate_model(
    model,
    dataloader,
    device: str = "cuda",
    max_samples: int = 1000,
    num_beams: int = 3,
    max_new_tokens: int = 256
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Evaluate trained model on dataset

    Generates captions for test images and computes evaluation metrics.

    Args:
        model: Trained VLM model with generate_caption method
        dataloader: Test dataloader
        device: Device to run on ('cuda' or 'cpu')
        max_samples: Maximum number of samples to evaluate
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum tokens to generate

    Returns:
        Tuple of (metrics_dict, references_list, predictions_list)
    """
    logger.info("="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    logger.info(f"Device: {device}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Num beams: {num_beams}")
    logger.info(f"Max tokens: {max_new_tokens}")
    logger.info("="*80)

    model.eval()
    model = model.to(device)

    references = []
    predictions = []
    num_evaluated = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating captions"):
            if num_evaluated >= max_samples:
                break

            try:
                pixel_values = batch['pixel_values'].to(device)

                # Get ground truth captions
                if 'caption' in batch:
                    ground_truth = batch['caption']
                elif 'captions' in batch:
                    ground_truth = batch['captions']
                else:
                    logger.warning("No caption field found in batch")
                    continue

                # Generate captions
                generated = model.generate_caption(
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    temperature=0.7
                )

                references.extend(ground_truth)
                predictions.extend(generated)
                num_evaluated += len(ground_truth)

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

    logger.info(f"Evaluated {len(predictions)} samples")

    # Compute metrics
    metrics_calculator = CaptionMetrics(use_cider=True)
    metrics = metrics_calculator.compute_all(references, predictions)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("="*80)

    return metrics, references, predictions


def print_sample_predictions(
    references: List[str],
    predictions: List[str],
    num_samples: int = 10
):
    """Print sample predictions for qualitative evaluation

    Args:
        references: Reference captions
        predictions: Predicted captions
        num_samples: Number of samples to print
    """
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    for i in range(min(num_samples, len(references))):
        print(f"\nSample {i+1}:")
        print(f"Reference:  {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-"*80)


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy data
    print("Testing CaptionMetrics...")

    references = [
        "This is a chest X-ray showing normal lung fields and clear heart borders.",
        "CT scan of the brain demonstrating no acute intracranial abnormalities.",
        "MRI of the spine showing degenerative disc disease at L4-L5.",
        "Microscopy image of tissue showing cellular infiltration.",
        "Ultrasound of the abdomen with normal liver echogenicity."
    ]

    predictions = [
        "Chest X-ray with normal lungs and heart.",
        "Brain CT scan showing no abnormalities.",
        "Spine MRI with disc degeneration at L4-L5.",
        "Tissue microscopy showing cell infiltration.",
        "Abdominal ultrasound with normal liver."
    ]

    # Test metrics
    calculator = CaptionMetrics(use_cider=False)
    metrics = calculator.compute_all(references, predictions)

    print("\nMetrics:")
    print("-"*80)
    for k, v in metrics.items():
        print(f"{k:.<40} {v:.4f}")
    print("-"*80)

    print("\nSample predictions:")
    print_sample_predictions(references, predictions, num_samples=3)

    print("\n✅ CaptionMetrics test complete!")
