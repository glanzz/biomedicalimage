"""
BiomedCLIP Vision Encoder
Pretrained on biomedical image-text pairs from PubMed Central

This encoder extracts visual features from medical images using the BiomedCLIP model,
which is specifically trained on biomedical imaging data.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoProcessor
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BiomedCLIPEncoder(nn.Module):
    """
    BiomedCLIP vision encoder wrapper for biomedical image feature extraction.

    Architecture:
        - Vision Transformer (ViT) base model with patch size 16
        - Input: 224x224 RGB images
        - Output: 768-dimensional image features
        - Pretrained on PubMed Central image-text pairs

    Args:
        model_name: HuggingFace model identifier
        freeze: Whether to freeze encoder weights (recommended for fine-tuning)
        device: Device to load model on
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        freeze: bool = True,
        device: Optional[str] = None
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading BiomedCLIP vision encoder: {model_name}")
        logger.info(f"Device: {self.device}")

        try:
            # Load vision model (CLIP vision component only)
            self.vision_model = CLIPVisionModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )

            # Load processor for image preprocessing
            self.processor = AutoProcessor.from_pretrained(model_name)

            # Get model configuration
            self.config = self.vision_model.config
            self.hidden_size = self.config.hidden_size  # 768 for ViT-Base
            self.image_size = self.config.image_size    # 224
            self.patch_size = self.config.patch_size    # 16

            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Hidden size: {self.hidden_size}")
            logger.info(f"  Image size: {self.image_size}x{self.image_size}")
            logger.info(f"  Patch size: {self.patch_size}x{self.patch_size}")

        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP model: {e}")
            raise

        # Move to device
        self.vision_model = self.vision_model.to(self.device)

        # Freeze parameters if requested
        if freeze:
            self._freeze_encoder()

        # Set to eval mode by default (important for BatchNorm/Dropout)
        self.vision_model.eval()

    def _freeze_encoder(self):
        """Freeze all vision encoder parameters"""
        for param in self.vision_model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in self.vision_model.parameters())
        trainable_params = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)

        logger.info(f"✓ Vision encoder frozen")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

    def unfreeze_encoder(self):
        """Unfreeze encoder for full fine-tuning"""
        for param in self.vision_model.parameters():
            param.requires_grad = True

        logger.info("✓ Vision encoder unfrozen for fine-tuning")

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_dict: bool = True
    ) -> torch.Tensor:
        """
        Extract visual features from images.

        Args:
            pixel_values: Preprocessed images [batch_size, 3, 224, 224]
            return_dict: Whether to return detailed output (not used, for compatibility)

        Returns:
            image_features: Pooled image features [batch_size, 768]
        """
        # Ensure correct device
        pixel_values = pixel_values.to(self.device)

        # Forward through vision model
        # Use no_grad if frozen to save memory
        with torch.set_grad_enabled(self.training and not self.freeze):
            outputs = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True
            )

            # Get pooled output (CLS token representation)
            # Shape: [batch_size, hidden_size]
            image_features = outputs.pooler_output

        return image_features

    def get_last_hidden_state(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get full sequence of patch embeddings (useful for advanced architectures).

        Args:
            pixel_values: Preprocessed images [batch_size, 3, 224, 224]

        Returns:
            hidden_states: Patch embeddings [batch_size, num_patches+1, 768]
        """
        pixel_values = pixel_values.to(self.device)

        with torch.set_grad_enabled(self.training and not self.freeze):
            outputs = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True
            )

            # Last hidden state includes all patch embeddings + CLS token
            hidden_states = outputs.last_hidden_state

        return hidden_states

    def preprocess_images(self, images, return_tensors="pt"):
        """
        Preprocess PIL images for the model.

        Args:
            images: List of PIL Images or single PIL Image
            return_tensors: Format to return ("pt" for PyTorch)

        Returns:
            Preprocessed pixel values ready for forward pass
        """
        processed = self.processor(
            images=images,
            return_tensors=return_tensors
        )
        return processed['pixel_values']

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and parameter information"""
        total_params = sum(p.numel() for p in self.vision_model.parameters())
        trainable_params = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)

        return {
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen': self.freeze,
            'device': str(self.device)
        }


# Unit tests
if __name__ == "__main__":
    import sys
    from PIL import Image
    import numpy as np

    print("="*70)
    print("TESTING BiomedCLIP Vision Encoder")
    print("="*70)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    try:
        # Test 1: Model initialization
        print("\n[Test 1] Initializing BiomedCLIP encoder...")
        encoder = BiomedCLIPEncoder(freeze=True)
        print("✓ Encoder initialized successfully")

        # Test 2: Model info
        print("\n[Test 2] Getting model info...")
        info = encoder.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("✓ Model info retrieved")

        # Test 3: Forward pass with dummy data
        print("\n[Test 3] Testing forward pass with dummy data...")
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        features = encoder(dummy_input)

        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {features.shape}")
        print(f"  Expected output shape: [{batch_size}, 768]")

        assert features.shape == (batch_size, 768), f"Shape mismatch! Got {features.shape}"
        assert not features.isnan().any(), "Output contains NaN values"
        print("✓ Forward pass successful")

        # Test 4: Get last hidden state
        print("\n[Test 4] Testing last hidden state extraction...")
        hidden_states = encoder.get_last_hidden_state(dummy_input)
        num_patches = (224 // 16) ** 2 + 1  # 14*14 + 1 (CLS token) = 197

        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Expected shape: [{batch_size}, {num_patches}, 768]")

        assert hidden_states.shape == (batch_size, num_patches, 768), "Hidden state shape mismatch"
        print("✓ Hidden state extraction successful")

        # Test 5: Image preprocessing
        print("\n[Test 5] Testing image preprocessing...")
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )

        processed = encoder.preprocess_images(dummy_image)
        print(f"  Processed shape: {processed.shape}")
        print(f"  Expected shape: [1, 3, 224, 224]")

        assert processed.shape == (1, 3, 224, 224), "Preprocessing shape mismatch"
        print("✓ Image preprocessing successful")

        # Test 6: Batch preprocessing
        print("\n[Test 6] Testing batch preprocessing...")
        dummy_images = [
            Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
            for _ in range(3)
        ]

        processed_batch = encoder.preprocess_images(dummy_images)
        print(f"  Processed batch shape: {processed_batch.shape}")
        print(f"  Expected shape: [3, 3, 224, 224]")

        assert processed_batch.shape == (3, 3, 224, 224), "Batch preprocessing shape mismatch"
        print("✓ Batch preprocessing successful")

        # Test 7: Gradient computation (when unfrozen)
        print("\n[Test 7] Testing gradient computation...")
        encoder.unfreeze_encoder()
        encoder.train()

        dummy_input = torch.randn(2, 3, 224, 224, requires_grad=True)
        features = encoder(dummy_input)
        loss = features.sum()
        loss.backward()

        has_grads = any(p.grad is not None for p in encoder.vision_model.parameters())
        print(f"  Gradients computed: {has_grads}")
        assert has_grads, "Gradients not computed when unfrozen"
        print("✓ Gradient computation successful")

        # Test 8: Memory efficiency when frozen
        print("\n[Test 8] Testing memory efficiency with frozen encoder...")
        encoder_frozen = BiomedCLIPEncoder(freeze=True)
        encoder_frozen.eval()

        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        dummy_input = torch.randn(2, 3, 224, 224)
        features = encoder_frozen(dummy_input)

        print(f"  Features shape: {features.shape}")
        print("✓ Frozen encoder memory test passed")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nBiomedCLIP Vision Encoder is ready for use in VLM pipeline.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
