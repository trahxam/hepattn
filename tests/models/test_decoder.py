import pytest
import torch
from torch import nn

from hepattn.models.decoder import MaskformerDecoder, MaskformerDecoderLayer


# Mock the required dependencies
class MockAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.projection = nn.Linear(dim, dim)

    def forward(self, q, kv=None, kv_mask=None, attn_mask=None, q_mask=None):  # noqa: ARG002
        if kv is None:
            kv = q
        return self.projection(q)


class MockDense(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return self.dense(x)


class MockLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class MockResidual(nn.Module):
    def __init__(self, dim, norm):
        super().__init__()
        self.dim = dim
        self.norm = norm(dim)
        self.layer = None

    def __call__(self, layer):
        self.layer = layer
        return self

    def forward(self, x, *args, **kwargs):
        return x + self.layer(self.norm(x), *args, **kwargs)


# Now let's write the tests
class TestMaskformerDecoderLayer:
    @pytest.fixture
    def decoder_layer(self):
        dim = 64
        return MaskformerDecoderLayer(dim=dim, mask_attention=True, bidirectional_ca=True)

    @pytest.fixture
    def sample_data(self):
        batch_size = 2
        num_queries = 5
        seq_len = 10
        dim = 64

        q = torch.randn(batch_size, num_queries, dim)
        kv = torch.randn(batch_size, seq_len, dim)
        attn_mask = torch.zeros(batch_size, num_queries, seq_len, dtype=torch.bool)
        kv_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        return q, kv, attn_mask, kv_mask

    def test_initialization(self, decoder_layer):
        """Test that the decoder layer initializes correctly"""
        assert decoder_layer.mask_attention is True
        assert decoder_layer.bidirectional_ca is True
        assert hasattr(decoder_layer, "q_ca")
        assert hasattr(decoder_layer, "q_sa")
        assert hasattr(decoder_layer, "q_dense")
        assert hasattr(decoder_layer, "kv_ca")
        assert hasattr(decoder_layer, "kv_dense")

    def test_initialization_no_bidirectional(self):
        """Test initialization with bidirectional_ca=False"""
        dim = 64
        layer = MaskformerDecoderLayer(dim=dim, mask_attention=True, bidirectional_ca=False)
        assert not hasattr(layer, "kv_ca")
        assert not hasattr(layer, "kv_dense")

    def test_forward_with_mask_attention(self, decoder_layer, sample_data):
        """Test forward pass with mask_attention=True"""
        q, kv, attn_mask, kv_mask = sample_data
        new_q, new_kv = decoder_layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        assert new_kv.shape == kv.shape

    def test_forward_mask_attention_no_mask(self, decoder_layer, sample_data):
        """Test that mask_attention=True requires an attn_mask"""
        q, kv, _, kv_mask = sample_data
        with pytest.raises(AssertionError, match="attn mask must be provided for mask attention"):
            decoder_layer(q, kv, attn_mask=None, kv_mask=kv_mask)

    def test_forward_no_mask_attention(self, sample_data):
        """Test forward pass with mask_attention=False"""
        q, kv, _, kv_mask = sample_data
        dim = q.shape[-1]
        layer = MaskformerDecoderLayer(dim=dim, mask_attention=False, bidirectional_ca=True)

        # Should not raise an assertion error even with no attn_mask
        new_q, new_kv = layer(q, kv, attn_mask=None, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        assert new_kv.shape == kv.shape

    def test_forward_no_bidirectional(self, sample_data):
        """Test forward pass with bidirectional_ca=False"""
        q, kv, attn_mask, kv_mask = sample_data
        dim = q.shape[-1]
        layer = MaskformerDecoderLayer(dim=dim, mask_attention=True, bidirectional_ca=False)

        new_q, new_kv = layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # Check output shapes
        assert new_q.shape == q.shape
        # Without bidirectional, kv should remain unchanged
        assert new_kv is kv

    def test_attn_mask_all_invalid(self, decoder_layer, sample_data):
        """Test behavior when attn_mask is all True for a query"""
        q, kv, attn_mask, kv_mask = sample_data
        # Set one query's mask to all True (invalid)
        attn_mask[0, 0, :] = True

        # Should not raise an error because the code handles this case
        _, _ = decoder_layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask)

        # We'd need to check that the mask was modified correctly
        # In real testing, you might want to verify this, but we'll skip for simplicity


class TestMaskformerDecoder:
    @pytest.fixture
    def mock_mask_net(self):
        class MockMaskNet(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.projection = nn.Linear(dim, dim)

            def forward(self, x):
                return self.projection(x)

        return MockMaskNet(64)

    @pytest.fixture
    def mock_class_net_binary(self):
        class MockClassNet(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.projection = nn.Linear(dim, 1)

            def forward(self, x):
                return self.projection(x)

        return MockClassNet(64)

    @pytest.fixture
    def mock_class_net_multiclass(self):
        class MockClassNet(nn.Module):
            def __init__(self, dim, num_classes):
                super().__init__()
                self.projection = nn.Linear(dim, num_classes)

            def forward(self, x):
                return self.projection(x)

        return MockClassNet(64, 10)

    @pytest.fixture
    def decoder(self, mock_mask_net, mock_class_net_binary):
        dim = 64
        num_layers = 2
        num_objects = 5
        md_config = {"mask_attention": True, "bidirectional_ca": True}

        return MaskformerDecoder(
            dim=dim,
            num_layers=num_layers,
            num_objects=num_objects,
            md_config=md_config,
            mask_net=mock_mask_net,
            class_net=mock_class_net_binary,
            intermediate_loss=False,
            mask_threshold=0.1,
        )

    @pytest.fixture
    def sample_data(self):
        batch_size = 2
        seq_len = 10
        dim = 64

        x = torch.randn(batch_size, seq_len, dim)
        input_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        # Set some padding
        input_pad_mask[0, -2:] = True

        return x, input_pad_mask

    def test_initialization(self, decoder, mock_mask_net, mock_class_net_binary):
        """Test that the decoder initializes correctly"""
        assert decoder.mask_threshold == 0.1
        assert decoder.intermediate_loss is False
        assert decoder.mask_net is mock_mask_net
        assert decoder.class_net is mock_class_net_binary
        assert len(decoder.layers) == 2
        assert decoder.initial_q.shape == (5, 64)  # num_objects x dim

    def test_forward(self, decoder, sample_data):
        """Test the forward pass"""
        x, input_pad_mask = sample_data
        preds = decoder(x, input_pad_mask)

        # Check output dictionary contains expected keys
        assert "q" in preds
        assert "x" in preds
        assert "class_logits" in preds
        assert "class_probs" in preds
        assert "mask_logits" in preds

        # Check shapes
        batch_size, seq_len, dim = x.shape
        num_objects = decoder.initial_q.shape[0]

        assert preds["q"].shape == (batch_size, num_objects, dim)
        assert preds["x"].shape == (batch_size, seq_len, dim)
        assert preds["class_logits"].shape == (batch_size, num_objects, 1)  # Binary classification
        assert preds["class_probs"].shape == (batch_size, num_objects, 2)  # [1-p, p] format
        assert preds["mask_logits"].shape == (batch_size, num_objects, seq_len)

    def test_forward_with_intermediate_loss(self, mock_mask_net, mock_class_net_binary, sample_data):
        """Test forward with intermediate_loss=True"""
        dim = 64
        num_layers = 2
        num_objects = 5
        md_config = {"mask_attention": True, "bidirectional_ca": True}

        decoder = MaskformerDecoder(
            dim=dim,
            num_layers=num_layers,
            num_objects=num_objects,
            md_config=md_config,
            mask_net=mock_mask_net,
            class_net=mock_class_net_binary,
            intermediate_loss=True,
            mask_threshold=0.1,
        )

        x, input_pad_mask = sample_data
        preds = decoder(x, input_pad_mask)

        # Check intermediate outputs
        assert "intermediate_outputs" in preds
        assert len(preds["intermediate_outputs"]) == num_layers

        # Check intermediate output format
        for inter_out in preds["intermediate_outputs"]:
            assert "q" in inter_out
            assert "class_logits" in inter_out
            assert "class_probs" in inter_out
            assert "mask_logits" in inter_out

    def test_forward_without_pad_mask(self, decoder, sample_data):
        """Test forward without padding mask"""
        x, _ = sample_data
        preds = decoder(x)

        # Check basic output structure
        assert "q" in preds
        assert "x" in preds
        assert "mask_logits" in preds

    def test_class_pred_binary(self, decoder, sample_data):
        """Test class_pred with binary classification"""
        x, _ = sample_data
        batch_size = x.shape[0]
        num_objects = decoder.initial_q.shape[0]
        dim = decoder.initial_q.shape[1]

        q = torch.randn(batch_size, num_objects, dim)
        out = decoder.class_pred(q)

        assert "class_logits" in out
        assert "class_probs" in out
        assert out["class_logits"].shape == (batch_size, num_objects, 1)  # Binary output
        assert out["class_probs"].shape == (batch_size, num_objects, 2)  # [1-p, p]

    def test_class_pred_multiclass(self, mock_mask_net, mock_class_net_multiclass, sample_data):
        """Test class_pred with multi-class classification"""
        dim = 64
        num_layers = 2
        num_objects = 5
        md_config = {"mask_attention": True, "bidirectional_ca": True}

        decoder = MaskformerDecoder(
            dim=dim,
            num_layers=num_layers,
            num_objects=num_objects,
            md_config=md_config,
            mask_net=mock_mask_net,
            class_net=mock_class_net_multiclass,
            intermediate_loss=False,
            mask_threshold=0.1,
        )

        x, _ = sample_data
        batch_size = x.shape[0]
        q = torch.randn(batch_size, num_objects, dim)
        out = decoder.class_pred(q)

        assert "class_logits" in out
        assert "class_probs" in out
        assert out["class_logits"].shape == (batch_size, num_objects, 10)  # 10 classes
        assert out["class_probs"].shape == (batch_size, num_objects, 10)  # Probabilities for 10 classes

    def test_class_pred_no_class_net(self, mock_mask_net, sample_data):
        """Test class_pred with no class_net"""
        dim = 64
        num_layers = 2
        num_objects = 5
        md_config = {"mask_attention": True, "bidirectional_ca": True}

        decoder = MaskformerDecoder(
            dim=dim,
            num_layers=num_layers,
            num_objects=num_objects,
            md_config=md_config,
            mask_net=mock_mask_net,
            class_net=None,
            intermediate_loss=False,
            mask_threshold=0.1,
        )

        x, _ = sample_data
        batch_size = x.shape[0]
        q = torch.randn(batch_size, num_objects, dim)
        out = decoder.class_pred(q)

        # Should return empty dict
        assert out == {}

    def test_mask_pred(self, decoder, sample_data):
        """Test mask_pred function"""
        x, input_pad_mask = sample_data
        batch_size, seq_len, dim = x.shape
        num_objects = decoder.initial_q.shape[0]

        q = torch.randn(batch_size, num_objects, dim)
        mask_logits = decoder.mask_pred(q, x, input_pad_mask)

        # Check shape
        assert mask_logits.shape == (batch_size, num_objects, seq_len)

        # Check padding applied correctly
        # Where input is padded, mask should be set to minimum value
        padded_positions = input_pad_mask[0, -2:].nonzero().squeeze()
        for pos in padded_positions:
            assert torch.all(mask_logits[0, :, pos] == torch.finfo(mask_logits.dtype).min)

    def test_mask_pred_no_pad_mask(self, decoder, sample_data):
        """Test mask_pred without padding mask"""
        x, _ = sample_data
        batch_size, seq_len, dim = x.shape
        num_objects = decoder.initial_q.shape[0]

        q = torch.randn(batch_size, num_objects, dim)
        mask_logits = decoder.mask_pred(q, x)

        # Check shape
        assert mask_logits.shape == (batch_size, num_objects, seq_len)

    def test_mask_pred_wrong_dimensions(self, decoder, sample_data):
        """Test mask_pred with wrong tensor dimensions"""
        x, _ = sample_data
        batch_size, seq_len, dim = x.shape
        num_objects = decoder.initial_q.shape[0]

        # 2D q (missing batch dimension)
        q_2d = torch.randn(num_objects, dim)
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            decoder.mask_pred(q_2d, x)

        # 2D x (missing batch dimension)
        q_3d = torch.randn(batch_size, num_objects, dim)
        x_2d = torch.randn(seq_len, dim)
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            decoder.mask_pred(q_3d, x_2d)

    def test_batch_size_mismatch(self, decoder, sample_data):
        """Test forward with batch size mismatch"""
        x, input_pad_mask = sample_data
        # Change batch size of x
        x_wrong_batch = torch.randn(x.shape[0] + 1, x.shape[1], x.shape[2])

        with pytest.raises(AssertionError, match="Batch size mismatch"):
            decoder(x_wrong_batch, input_pad_mask)

    def test_feature_dimension_mismatch(self, decoder, sample_data):
        """Test forward with feature dimension mismatch"""
        x, input_pad_mask = sample_data
        # Change feature dimension of x
        x_wrong_dim = torch.randn(x.shape[0], x.shape[1], x.shape[2] + 10)

        with pytest.raises(AssertionError, match="Feature dimension mismatch"):
            decoder(x_wrong_dim, input_pad_mask)


# Additional integration tests
class TestMaskformerIntegration:
    def test_full_forward_pass(self):
        """Integration test with real components"""
        batch_size = 2
        seq_len = 10
        dim = 64
        num_objects = 5
        num_layers = 2

        # Create inputs
        x = torch.randn(batch_size, seq_len, dim)
        input_pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        input_pad_mask[0, -2:] = True

        # Create networks
        mask_net = nn.Linear(dim, dim)
        class_net = nn.Linear(dim, 10)  # multi-class (10 classes)

        # Create decoder
        md_config = {"mask_attention": True, "bidirectional_ca": True}

        decoder = MaskformerDecoder(
            dim=dim,
            num_layers=num_layers,
            num_objects=num_objects,
            md_config=md_config,
            mask_net=mask_net,
            class_net=class_net,
            intermediate_loss=True,
            mask_threshold=0.1,
        )

        # Forward pass
        preds = decoder(x, input_pad_mask)

        # Validate outputs
        assert preds["q"].shape == (batch_size, num_objects, dim)
        assert preds["x"].shape == (batch_size, seq_len, dim)
        assert preds["class_logits"].shape == (batch_size, num_objects, 10)
        assert preds["class_probs"].shape == (batch_size, num_objects, 10)
        assert preds["mask_logits"].shape == (batch_size, num_objects, seq_len)
        assert len(preds["intermediate_outputs"]) == num_layers
