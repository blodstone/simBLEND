import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Assuming your project structure allows this import.
# If tests/ is a package, you might use relative imports or adjust PYTHONPATH.
from modules.llama_decoder import LlamaDecoderForNextArticle

@pytest.fixture
def model_params_for_test():
    """
    Provides minimal and LLaMA-compatible parameters for the model.
    - constructor_hidden_size = 7
    - llama_config_hidden_size = constructor_hidden_size + 5 = 12
    - num_attention_heads = 2
    - head_dim = 12 / 2 = 6 (even, valid for RoPE)
    """
    return {
        "codebook_size": 5,  # Vocabulary size for lm_head and embeddings
        "hidden_size": 7,    # Input to LlamaDecoderForNextArticle constructor
        "intermediate_size": 24, # Typical LLaMA intermediate size (e.g., 2*llama_config_hidden_size)
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 10, # Max sequence length model can handle
        "learning_rate": 1e-4, # Not used by common_step directly
        "warmup_epochs": 1     # Not used by common_step directly
    }

@pytest.fixture
def model_instance(model_params_for_test):
    """
    Creates an instance of LlamaDecoderForNextArticle with the test parameters.
    """
    model = LlamaDecoderForNextArticle(**model_params_for_test)
    # Put model in eval mode if dropout or other training-specific layers exist,
    # though for common_step with mocks, it might not be strictly necessary.
    model.eval()
    return model

def test_behavior_mask_in_loss_calculation(model_instance):
    """
    Tests that the behavior_mask correctly influences the loss calculation
    in the common_step method.
    """
    model = model_instance
    torch.manual_seed(42) # For reproducible random tensors

    batch_size = 2
    seq_len = 4 # Must be <= model_params_for_test["max_position_embeddings"]

    # model.config.vocab_size is codebook_size from constructor (5)
    # model.config.hidden_size is (constructor hidden_size + 5) = (7 + 5) = 12

    # --- Prepare Batch Data ---
    # `indices` are the target labels for the next token prediction, with padding
    indices = torch.tensor([
        [1, 2, 3, 0],  # Sample 1: sequence of length 3, padded with 0
        [4, 1, 5, 2]   # Sample 2: sequence of length 2, padded with 0
    ], dtype=torch.long)

    # `behaviors` tensor for `behavior_embedding(1, 5)` must contain only 0s
    behaviors_input = torch.zeros((batch_size, seq_len), dtype=torch.long)
    
    # `behavior_masks_input` is crucial for masking loss contributions:
    # 0: don't include in loss
    # 1: include in loss
    # -1: padding, effectively not included (CrossEntropyLoss ignore_index handles labels,
    #     and (shift_behavior == 1) handles the mask)
    behavior_masks_input = torch.tensor([
        [0, 1, 0, 1],  # Sample 1: predict for pos 1 (mask=1), pos 3 (mask=1)
        [1, 0, -1, 1]  # Sample 2: predict for pos 0 (mask=1), pos 3 (mask=1), pos 2 is padding-like
    ], dtype=torch.long)
    
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    batch = (indices, behaviors_input, behavior_masks_input, attention_mask)

    # --- Mock LlamaModel and LMHead ---
    # Define the object that model.llama(...) is expected to return.
    # This object should have a 'last_hidden_state' attribute.
    mock_llama_call_result = MagicMock()
    # `last_hidden_state` shape: (batch_size, seq_len, LlamaConfig.hidden_size)
    # LlamaConfig.hidden_size is `model.config.hidden_size` (which is 12 for this test)
    mock_llama_call_result.last_hidden_state = torch.randn(batch_size, seq_len, model.config.hidden_size)

    # Define the tensor that model.lm_head(...) is expected to return.
    # These are the raw logits for all sequence positions.
    # LMHead input dim: LlamaConfig.hidden_size (12)
    # LMHead output dim: LlamaConfig.vocab_size (5)
    mock_lm_head_call_result = torch.randn(batch_size, seq_len, model.config.vocab_size)

    # Patch model.llama to be a MagicMock that returns mock_llama_call_result when called.
    # Patch model.lm_head to be a MagicMock that returns mock_lm_head_call_result when called.
    with patch.object(model, 'llama', MagicMock(return_value=mock_llama_call_result)) as patched_llama_attr, \
         patch.object(model, 'lm_head', MagicMock(return_value=mock_lm_head_call_result)) as patched_lm_head_attr:

        # --- Call common_step ---
        # batch_idx=0 is a placeholder
        calculated_loss, returned_shift_logits, returned_shift_labels, returned_loss_mask = \
            model.common_step(batch, 0)

    # --- Manually Calculate Expected Loss ---
    # These are the logits for tokens t=0 to t=SeqLen-2, predicting t=1 to t=SeqLen-1
    expected_shift_logits = mock_lm_head_call_result[:, :-1, :].contiguous()
    # These are the target labels for tokens t=1 to t=SeqLen-1
    expected_shift_labels = indices[:, 1:].contiguous()
    # These are the behavior masks for tokens t=1 to t=SeqLen-1
    expected_shift_behaviors = behavior_masks_input[:, 1:].contiguous()

    # Sanity check: ensure the shifted tensors returned by common_step match our expectations
    assert torch.equal(returned_shift_logits, expected_shift_logits), "Shifted logits mismatch"
    assert torch.equal(returned_shift_labels, expected_shift_labels), "Shifted labels mismatch"

    # This is the crucial mask derived inside common_step for loss calculation
    # behavior_mask = (shift_behaviors == 1).float()
    expected_internal_loss_mask = (expected_shift_behaviors == 1).float()
    assert torch.equal(returned_loss_mask, expected_internal_loss_mask), "Internal loss mask mismatch"

    # Calculate loss manually using the same CrossEntropyLoss settings
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    
    # Per-token losses for all shifted positions, before behavior masking
    # Use the exact same logits and labels that common_step used internally
    per_token_loss_manual = loss_fct(
        expected_shift_logits.reshape(-1, model.config.vocab_size), # Flatten to (B*(SeqLen-1), VocabSize)
        expected_shift_labels.reshape(-1)                            # Flatten to (B*(SeqLen-1),)
    )
    
    # Apply the behavior-derived mask
    # expected_internal_loss_mask is (B, SeqLen-1)
    masked_loss_contributions = per_token_loss_manual * expected_internal_loss_mask.reshape(-1)
    
    # Calculate the final average loss
    sum_of_active_mask_elements = expected_internal_loss_mask.sum()
    
    # Handle case where no tokens contribute to loss
    if sum_of_active_mask_elements == 0:
        expected_loss = torch.tensor(0.0, device=calculated_loss.device, dtype=calculated_loss.dtype)
    else:
        # common_step uses .clamp(min=1) for the denominator
        expected_loss = masked_loss_contributions.sum() / sum_of_active_mask_elements.clamp(min=1)

    assert torch.isclose(calculated_loss, expected_loss), \
        f"Calculated loss {calculated_loss} does not match expected loss {expected_loss}"

        # Assert that the mocked attributes were called as expected
    patched_llama_attr.assert_called_once()
    patched_lm_head_attr.assert_called_once()

    # --- Test Case: All relevant tokens masked out ---
    # Reset mocks for the next call if necessary, or ensure they handle multiple calls if designed for it.
    # For this specific test, common_step is called again, so mocks will be called again.
    # If the mocks should only be called once per setup, re-patch or re-configure.
    # Here, we assume the mocks are fresh or correctly configured for a new call.
    patched_llama_attr.reset_mock()
    patched_lm_head_attr.reset_mock()

    behavior_masks_all_masked_shifted = torch.tensor([
        [0, 0, 0, 0],  # After shifting, shift_behaviors will be [0,0,0] -> mask [0,0,0]
        [-1, 0, -1, 0] # After shifting, shift_behaviors will be [0,-1,0] -> mask [0,0,0]
    ], dtype=torch.long)
    batch_all_masked = (indices, behaviors_input, behavior_masks_all_masked_shifted, attention_mask)
    
    loss_all_masked, _, _, internal_mask_all_masked = model.common_step(batch_all_masked, 0)
    assert internal_mask_all_masked.sum() == 0, \
        "Internal loss mask sum should be 0 when all relevant tokens are masked"
    assert torch.isclose(loss_all_masked, torch.tensor(0.0, device=loss_all_masked.device, dtype=loss_all_masked.dtype)), \
        f"Loss with all relevant tokens masked should be 0.0, got {loss_all_masked}"
