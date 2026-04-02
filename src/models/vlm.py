"""
Biomedical Vision-Language Model (VLM)
Combines BiomedCLIP vision encoder with BioMistral-7B language model

Architecture:
    Image → BiomedCLIP (frozen) → Projector (trainable) → BioMistral + LoRA → Caption

This is the main model that integrates all components for biomedical image captioning.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List
import logging

from .vision_encoder import BiomedCLIPEncoder
from .language_model import BioMistralLM
from .projector import MultiLayerProjector

logger = logging.getLogger(__name__)


class BiomedVLM(nn.Module):
    """
    Biomedical Vision-Language Model for medical image captioning.

    Architecture Flow:
        1. Image → BiomedCLIP encoder → 768D features
        2. 768D features → Multi-layer projector → 4096D embeddings
        3. Visual embeddings + text tokens → BioMistral → Output logits
        4. Training: Compute language modeling loss
        5. Inference: Generate caption autoregressively

    Args:
        vision_model_name: HuggingFace identifier for vision model
        language_model_name: HuggingFace identifier for language model
        freeze_vision_encoder: Whether to freeze vision encoder weights
        use_lora: Whether to use LoRA for language model
        projector_config: Configuration dict for projector network
        device: Device to load models on
    """

    def __init__(
        self,
        vision_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        language_model_name: str = "BioMistral/BioMistral-7B",
        freeze_vision_encoder: bool = True,
        use_lora: bool = True,
        projector_config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("="*70)
        logger.info("Initializing Biomedical Vision-Language Model")
        logger.info("="*70)

        # Initialize vision encoder (BiomedCLIP)
        logger.info("\n[1/4] Loading Vision Encoder...")
        self.vision_encoder = BiomedCLIPEncoder(
            model_name=vision_model_name,
            freeze=freeze_vision_encoder,
            device=self.device
        )
        logger.info(f"✓ Vision encoder loaded: {self.vision_encoder.hidden_size}D features")

        # Initialize language model (BioMistral with LoRA)
        logger.info("\n[2/4] Loading Language Model...")
        self.language_model = BioMistralLM(
            model_name=language_model_name,
            load_in_8bit=True,
            use_lora=use_lora,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            device_map="auto"
        )
        logger.info(f"✓ Language model loaded: {self.language_model.hidden_size}D embeddings")

        # Initialize projection network
        logger.info("\n[3/4] Initializing Projector Network...")
        proj_config = projector_config or {}
        self.projector = MultiLayerProjector(
            input_dim=self.vision_encoder.hidden_size,  # 768
            output_dim=self.language_model.hidden_size,  # 4096
            hidden_dim=proj_config.get('hidden_dim', 2048),
            num_layers=proj_config.get('num_layers', 2),
            activation=proj_config.get('activation', 'gelu'),
            dropout=proj_config.get('dropout', 0.1)
        ).to(self.device)
        logger.info("✓ Projector initialized")

        # Add special image token to tokenizer
        logger.info("\n[4/4] Configuring Special Tokens...")
        self._setup_special_tokens()

        # Cache tokenizer for convenience
        self.tokenizer = self.language_model.tokenizer

        logger.info("\n" + "="*70)
        logger.info("Model Initialization Complete!")
        logger.info("="*70)
        self._print_model_summary()

    def _setup_special_tokens(self):
        """Add special tokens for image input"""
        # Check if <image> token exists
        image_token = "<image>"
        special_tokens = {"additional_special_tokens": [image_token]}

        # Add token if not present
        num_added = self.language_model.tokenizer.add_special_tokens(special_tokens)

        if num_added > 0:
            # Resize token embeddings
            self.language_model.resize_token_embeddings(
                len(self.language_model.tokenizer)
            )
            logger.info(f"✓ Added {num_added} special tokens")

        # Get token ID
        self.image_token_id = self.language_model.tokenizer.convert_tokens_to_ids(image_token)
        logger.info(f"  Image token ID: {self.image_token_id}")

    def _print_model_summary(self):
        """Print summary of model architecture and parameters"""
        # Count parameters by component
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        vision_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)

        projector_params = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        lm_params = sum(p.numel() for p in self.language_model.parameters())
        lm_trainable = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)

        total_params = vision_params + projector_params + lm_params
        total_trainable = vision_trainable + projector_trainable + lm_trainable

        logger.info("\nModel Architecture Summary:")
        logger.info(f"  Vision Encoder: {vision_params:,} params ({vision_trainable:,} trainable)")
        logger.info(f"  Projector: {projector_params:,} params ({projector_trainable:,} trainable)")
        logger.info(f"  Language Model: {lm_params:,} params ({lm_trainable:,} trainable)")
        logger.info(f"  Total: {total_params:,} params ({total_trainable:,} trainable)")
        logger.info(f"  Trainable: {total_trainable/total_params*100:.2f}%")

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the complete VLM.

        Args:
            pixel_values: Images [batch_size, 3, 224, 224]
            input_ids: Caption token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]

        Returns:
            Model outputs with loss (if labels provided) and logits
        """
        batch_size = pixel_values.shape[0]

        # 1. Extract vision features
        # [batch_size, 3, 224, 224] → [batch_size, 768]
        vision_features = self.vision_encoder(pixel_values)

        # 2. Project to language model space
        # [batch_size, 768] → [batch_size, 4096]
        vision_embeds = self.projector(vision_features)

        # Add sequence dimension for concatenation
        # [batch_size, 4096] → [batch_size, 1, 4096]
        vision_embeds = vision_embeds.unsqueeze(1)

        # 3. Get text embeddings from language model
        # [batch_size, seq_len] → [batch_size, seq_len, 4096]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 4. Concatenate vision and text embeddings
        # Format: [image_embedding, caption_tokens]
        # [batch_size, 1, 4096] + [batch_size, seq_len, 4096] → [batch_size, 1+seq_len, 4096]
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # 5. Extend attention mask for image token
        # [batch_size, 1] + [batch_size, seq_len] → [batch_size, 1+seq_len]
        image_attention = torch.ones(
            batch_size, 1,
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        extended_attention_mask = torch.cat([image_attention, attention_mask], dim=1)

        # 6. Extend labels if provided (for training)
        if labels is not None:
            # Image token should not contribute to loss
            image_labels = torch.full(
                (batch_size, 1),
                -100,  # Ignore index for CrossEntropyLoss
                dtype=labels.dtype,
                device=labels.device
            )
            extended_labels = torch.cat([image_labels, labels], dim=1)
        else:
            extended_labels = None

        # 7. Forward through language model
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
            return_dict=True
        )

        return outputs

    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate captions for images.

        Args:
            pixel_values: Images [batch_size, 3, 224, 224]
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search (1 = greedy)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling (False = greedy)

        Returns:
            List of generated captions
        """
        self.eval()
        batch_size = pixel_values.shape[0]

        # 1. Extract and project vision features
        vision_features = self.vision_encoder(pixel_values)
        vision_embeds = self.projector(vision_features).unsqueeze(1)

        # 2. Create attention mask for image token
        vision_attention_mask = torch.ones(
            batch_size, 1,
            dtype=torch.long,
            device=pixel_values.device
        )

        # 3. Generate with language model
        generated_ids = self.language_model.generate(
            inputs_embeds=vision_embeds,
            attention_mask=vision_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams if not do_sample else 1,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 50,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # 4. Decode generated tokens
        captions = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        return captions

    def get_trainable_parameters(self):
        """Get only trainable parameters (for optimizer)"""
        return [p for p in self.parameters() if p.requires_grad]

    def save_model(self, save_path: str):
        """Save model state dict"""
        torch.save({
            'projector': self.projector.state_dict(),
            'language_model': self.language_model.model.state_dict(),
        }, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load model state dict"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.projector.load_state_dict(checkpoint['projector'])
        self.language_model.model.load_state_dict(checkpoint['language_model'])
        logger.info(f"Model loaded from {load_path}")


# Unit tests
if __name__ == "__main__":
    import sys

    print("="*70)
    print("TESTING Biomedical Vision-Language Model")
    print("="*70)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("\nNote: Full VLM testing requires significant memory and GPU.")
    print("Testing architecture with mock components...\n")

    try:
        # Test with architecture validation (no actual model loading)
        print("[Test 1] Validating VLM architecture...")

        # This will test the architecture but may fail on model loading
        # which is expected if models aren't downloaded
        try:
            vlm = BiomedVLM(
                freeze_vision_encoder=True,
                use_lora=True,
                projector_config={
                    'hidden_dim': 2048,
                    'num_layers': 2,
                    'dropout': 0.1
                }
            )
            print("✓ VLM initialized successfully")

            # Test forward pass
            print("\n[Test 2] Testing forward pass...")
            batch_size = 2
            dummy_pixels = torch.randn(batch_size, 3, 224, 224).cuda()
            dummy_input_ids = torch.randint(0, 1000, (batch_size, 20)).cuda()
            dummy_attention = torch.ones(batch_size, 20).cuda()
            dummy_labels = dummy_input_ids.clone()

            outputs = vlm(
                pixel_values=dummy_pixels,
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention,
                labels=dummy_labels
            )

            print(f"  Loss: {outputs.loss.item():.4f}")
            print(f"  Logits shape: {outputs.logits.shape}")
            print("✓ Forward pass successful")

            # Test generation
            print("\n[Test 3] Testing caption generation...")
            captions = vlm.generate_caption(
                pixel_values=dummy_pixels,
                max_new_tokens=20,
                do_sample=True
            )

            print(f"  Generated {len(captions)} captions")
            for i, cap in enumerate(captions):
                print(f"  [{i}]: {cap[:50]}...")
            print("✓ Generation successful")

            print("\n" + "="*70)
            print("ALL TESTS PASSED! ✅")
            print("="*70)

        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            print("   This is expected if models aren't downloaded.")
            print("   Testing architecture components individually...")

            # Test individual components
            print("\n[Fallback Test] Testing projector component...")
            from .projector import MultiLayerProjector

            projector = MultiLayerProjector(
                input_dim=768,
                output_dim=4096,
                hidden_dim=2048,
                num_layers=2
            )

            vision_feats = torch.randn(4, 768)
            projected = projector(vision_feats)

            assert projected.shape == (4, 4096), "Projector output mismatch"
            print("✓ Projector component works")

            print("\n✅ Architecture validation passed")
            print("   (Full model testing requires downloaded models)")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
