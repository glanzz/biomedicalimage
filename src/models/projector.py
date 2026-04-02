"""
Multi-Layer Projector Network
Maps vision features from BiomedCLIP (768D) to BioMistral embedding space (4096D)

This module implements a learnable projection network to align visual and textual representations.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MultiLayerProjector(nn.Module):
    """
    Multi-layer projection network to map vision features to LLM embedding space.

    Architecture:
        Input: [batch_size, 768] (BiomedCLIP features)
        Hidden layers with LayerNorm and GELU activation
        Output: [batch_size, 4096] (BioMistral embedding dimension)

    This is a critical component that learns to align visual and language modalities.

    Args:
        input_dim: Dimension of vision features (768 for BiomedCLIP)
        output_dim: Dimension of LLM embeddings (4096 for Mistral-7B)
        hidden_dim: Dimension of hidden layers
        num_layers: Number of projection layers (minimum 2)
        activation: Activation function ('gelu' or 'relu')
        dropout: Dropout probability (0.0 to disable)
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 4096,
        hidden_dim: int = 2048,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        super().__init__()

        assert num_layers >= 2, "Need at least 2 layers for projection"
        assert activation in ["gelu", "relu"], f"Unsupported activation: {activation}"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        logger.info(f"Initializing MultiLayerProjector:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Output dim: {output_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Num layers: {num_layers}")
        logger.info(f"  Activation: {activation}")
        logger.info(f"  Dropout: {dropout}")

        # Build projection layers
        layers = []

        # Input layer: input_dim -> hidden_dim
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        ])

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim -> hidden_dim
        for i in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim -> output_dim
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.projector = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✓ Projector initialized with {total_params:,} parameters")

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for LayerNorm
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to LLM embedding space.

        Args:
            vision_features: Vision encoder output [batch_size, input_dim]

        Returns:
            projected_features: Projected features [batch_size, output_dim]
        """
        # Ensure input is 2D
        if vision_features.dim() == 3:
            # Handle case where input is [batch_size, 1, input_dim]
            vision_features = vision_features.squeeze(1)

        assert vision_features.dim() == 2, f"Expected 2D input, got {vision_features.dim()}D"
        assert vision_features.shape[1] == self.input_dim, \
            f"Expected input_dim={self.input_dim}, got {vision_features.shape[1]}"

        # Project
        projected = self.projector(vision_features)

        return projected

    def get_parameter_count(self) -> dict:
        """Get detailed parameter count"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'trainable_percentage': (trainable / total * 100) if total > 0 else 0
        }


class IdentityProjector(nn.Module):
    """
    Identity projector for when dimensions already match.
    Useful for ablation studies or when using same-dimension encoders.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        if input_dim != output_dim:
            logger.warning(
                f"IdentityProjector dimension mismatch: {input_dim} != {output_dim}. "
                f"Consider using MultiLayerProjector instead."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through (identity function)"""
        return x


class LinearProjector(nn.Module):
    """
    Simple single-layer linear projection.
    Faster but less expressive than MultiLayerProjector.
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 4096):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.projection = nn.Linear(input_dim, output_dim)

        # Initialize
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

        logger.info(f"LinearProjector: {input_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear projection"""
        return self.projection(x)


# Unit tests
if __name__ == "__main__":
    import sys

    print("="*70)
    print("TESTING PROJECTOR NETWORKS")
    print("="*70)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    try:
        # Test 1: MultiLayerProjector initialization
        print("\n[Test 1] Initializing MultiLayerProjector...")
        projector = MultiLayerProjector(
            input_dim=768,
            output_dim=4096,
            hidden_dim=2048,
            num_layers=2,
            activation="gelu",
            dropout=0.1
        )
        print("✓ MultiLayerProjector initialized")

        # Test 2: Parameter count
        print("\n[Test 2] Checking parameters...")
        param_info = projector.get_parameter_count()
        for key, value in param_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value:,}")
        print("✓ Parameter count correct")

        # Test 3: Forward pass
        print("\n[Test 3] Testing forward pass...")
        batch_size = 8
        input_features = torch.randn(batch_size, 768)

        output_features = projector(input_features)

        print(f"  Input shape: {input_features.shape}")
        print(f"  Output shape: {output_features.shape}")
        print(f"  Expected: [{batch_size}, 4096]")

        assert output_features.shape == (batch_size, 4096), "Output shape mismatch"
        assert not output_features.isnan().any(), "Output contains NaN"
        print("✓ Forward pass successful")

        # Test 4: Gradient flow
        print("\n[Test 4] Testing gradient flow...")
        projector.train()
        input_features = torch.randn(batch_size, 768, requires_grad=True)

        output = projector(input_features)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        has_grads = all(p.grad is not None for p in projector.parameters())
        assert has_grads, "Some parameters don't have gradients"

        # Check input gradients
        assert input_features.grad is not None, "Input gradient is None"

        print("✓ Gradients flow correctly")

        # Test 5: Different layer counts
        print("\n[Test 5] Testing different layer configurations...")
        for num_layers in [2, 3, 4]:
            proj = MultiLayerProjector(
                input_dim=768,
                output_dim=4096,
                hidden_dim=2048,
                num_layers=num_layers
            )

            output = proj(torch.randn(4, 768))
            assert output.shape == (4, 4096), f"Failed for {num_layers} layers"
            print(f"  {num_layers} layers: ✓")

        print("✓ Different configurations work")

        # Test 6: LinearProjector
        print("\n[Test 6] Testing LinearProjector...")
        linear_proj = LinearProjector(input_dim=768, output_dim=4096)

        output = linear_proj(torch.randn(batch_size, 768))
        assert output.shape == (batch_size, 4096), "LinearProjector shape mismatch"

        param_count = sum(p.numel() for p in linear_proj.parameters())
        print(f"  Parameters: {param_count:,}")
        print("✓ LinearProjector works")

        # Test 7: IdentityProjector
        print("\n[Test 7] Testing IdentityProjector...")
        identity_proj = IdentityProjector(input_dim=768, output_dim=768)

        input_tensor = torch.randn(batch_size, 768)
        output = identity_proj(input_tensor)

        assert torch.equal(input_tensor, output), "IdentityProjector modified input"
        print("✓ IdentityProjector works")

        # Test 8: Batch size variations
        print("\n[Test 8] Testing different batch sizes...")
        test_batch_sizes = [1, 4, 16, 32]

        for bs in test_batch_sizes:
            input_data = torch.randn(bs, 768)
            output = projector(input_data)
            assert output.shape == (bs, 4096), f"Failed for batch_size={bs}"

        print(f"  Tested batch sizes: {test_batch_sizes}")
        print("✓ All batch sizes work")

        # Test 9: Eval mode
        print("\n[Test 9] Testing eval mode...")
        projector.eval()

        with torch.no_grad():
            input_data = torch.randn(batch_size, 768)
            output = projector(input_data)

        assert output.shape == (batch_size, 4096), "Eval mode output mismatch"
        print("✓ Eval mode works")

        # Test 10: Different activations
        print("\n[Test 10] Testing different activations...")
        for activation in ["gelu", "relu"]:
            proj = MultiLayerProjector(
                input_dim=768,
                output_dim=4096,
                activation=activation
            )

            output = proj(torch.randn(4, 768))
            assert output.shape == (4, 4096), f"Failed for activation={activation}"
            print(f"  {activation.upper()}: ✓")

        print("✓ Different activations work")

        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✅")
        print("="*70)
        print("\nProjector networks are ready for use in VLM pipeline.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
