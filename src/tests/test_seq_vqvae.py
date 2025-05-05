import pytest
import torch
import torch.nn as nn

# Assume your classes are defined in 'your_module.py'
# If they are in the same file, you can remove the 'your_module.' prefix
from train_seq_vqvae import (
    CodeEncoder, 
    BahdanauAttention, 
    CodeDecoder, 
    LlamaDecoderForNextArticle,
    # BeamSearchNode # Not testing this helper class directly
) 
# Make sure transformers is installed: pip install transformers
from transformers import LlamaConfig, LlamaModel 

# --- Test Configuration ---
BATCH_SIZE = 4
SEQ_LEN = 10
CODE_ITER = 3 # num_vqvae_iter
EMBEDDING_SIZE = 64
GRU_HIDDEN_SIZE = 128 # Hidden size for GRU components
GRU_NUM_LAYERS = 2
LLAMA_HIDDEN_SIZE = 384 # Must match CodeEncoder output_size and Attention/Decoder expectations
VOCAB_SIZE = 50 # Example vocabulary size for VQVAE codes
DROPOUT = 0.1

# Use CPU for testing unless GPU is necessary and available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Fixtures ---

@pytest.fixture(scope="module")
def embedding_layer():
    """Provides a shared embedding layer."""
    return nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE).to(DEVICE)

@pytest.fixture(scope="module")
def code_encoder(embedding_layer):
    """Provides an instance of CodeEncoder."""
    # Output size must match LLAMA_HIDDEN_SIZE for LlamaDecoderForNextArticle
    encoder = CodeEncoder(
        embedding_size=EMBEDDING_SIZE,
        hidden_size=GRU_HIDDEN_SIZE,
        embedding_layer=embedding_layer,
        output_size=LLAMA_HIDDEN_SIZE, # Output matches Llama input dimension
        num_layers=GRU_NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    encoder.eval() # Set to evaluation mode
    return encoder

@pytest.fixture(scope="module")
def bahdanau_attention():
    """Provides an instance of BahdanauAttention."""
    # Decoder hidden dim often matches GRU hidden size or Llama hidden size depending on usage
    # Encoder output dim matches the output of the encoder (CodeEncoder output / Llama output)
    attention = BahdanauAttention(
        decoder_hidden_dim=GRU_HIDDEN_SIZE, # Assuming query comes from Llama/Decoder GRU state
        encoder_output_dim=LLAMA_HIDDEN_SIZE  # Assuming keys/values come from Llama/Encoder output
    ).to(DEVICE)
    attention.eval()
    return attention

@pytest.fixture(scope="module")
def code_decoder(embedding_layer):
    """Provides an instance of CodeDecoder."""
    # Hidden size should match the hidden state it receives (e.g., from Llama or another GRU)
    # Encoder output dim matches the context it receives (e.g., from Llama or CodeEncoder)
    decoder = CodeDecoder(
        embedding_size=EMBEDDING_SIZE,
        gru_hidden_size=GRU_HIDDEN_SIZE, # Matching the state it will likely receive in the combined model
        vocab_size=VOCAB_SIZE,
        embedding_layer=embedding_layer,
        encoder_output_dim=LLAMA_HIDDEN_SIZE, # Context dimension
        dropout=DROPOUT
    ).to(DEVICE)
    decoder.eval()
    return decoder

@pytest.fixture(scope="module")
def llama_decoder_model():
    """Provides an instance of LlamaDecoderForNextArticle."""
    # Ensure consistency: CodeEncoder output_size == LLAMA_HIDDEN_SIZE
    # Ensure consistency: CodeDecoder hidden_size and encoder_output_dim match LLAMA_HIDDEN_SIZE
    model = LlamaDecoderForNextArticle(
        num_vqvae_iter=CODE_ITER,
        code_embedding_size=EMBEDDING_SIZE,
        teacher_forcing_ratio=0.5, # Value doesn't matter much for shape testing
        gru_hidden_size=GRU_HIDDEN_SIZE, # GRU's internal hidden size
        hidden_size=LLAMA_HIDDEN_SIZE, # Llama hidden size (d_model)
        num_hidden_layers=2, # Keep Llama small for testing
        num_attention_heads=4, # Keep Llama small
        intermediate_size=256, # Keep Llama small
        max_position_embeddings=SEQ_LEN + 5, # Needs to be >= SEQ_LEN
        vocab_size=VOCAB_SIZE, # VQVAE vocab size
        dropout_prob=DROPOUT
    ).to(DEVICE)
    model.eval()
    return model

# --- Test Functions ---

def test_code_encoder_shapes(code_encoder):
    """Tests input and output shapes of CodeEncoder."""
    # Input: (batch_size, seq_len, code_iter) - Long type for embedding indices
    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN, CODE_ITER), dtype=torch.long).to(DEVICE)
    
    # Forward pass
    with torch.no_grad():
        output = code_encoder(dummy_input)

    # Expected output shape: (batch_size, seq_len, output_size)
    expected_shape = (BATCH_SIZE, SEQ_LEN, LLAMA_HIDDEN_SIZE) # Output size matches Llama hidden dim
    
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
    assert output.dtype == torch.float32, f"Expected dtype torch.float32, but got {output.dtype}"
    print(f"\nCodeEncoder shapes OK: Input {dummy_input.shape} -> Output {output.shape}")

def test_bahdanau_attention_shapes(bahdanau_attention):
    """Tests input and output shapes of BahdanauAttention."""
    # Decoder hidden state (query): (batch_size, decoder_hidden_dim)
    dummy_decoder_hidden = torch.randn(BATCH_SIZE, GRU_HIDDEN_SIZE).to(DEVICE)
    
    # Encoder outputs (keys/values): (batch_size, seq_len, encoder_output_dim)
    dummy_encoder_outputs = torch.randn(BATCH_SIZE, SEQ_LEN, LLAMA_HIDDEN_SIZE).to(DEVICE)
    
    # Encoder mask (optional): (batch_size, seq_len) - True/1 for valid tokens
    dummy_encoder_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool).to(DEVICE)
    # Mask out the last few elements for testing
    if SEQ_LEN > 2:
        dummy_encoder_mask[:, -2:] = False 

    # --- Test with mask ---
    with torch.no_grad():
        context_vector, attention_weights = bahdanau_attention(
            dummy_decoder_hidden, dummy_encoder_outputs, dummy_encoder_mask
        )

    # Expected shapes
    expected_context_shape = (BATCH_SIZE, 1, LLAMA_HIDDEN_SIZE)
    expected_weights_shape = (BATCH_SIZE, 1, SEQ_LEN)

    assert context_vector.shape == expected_context_shape, \
        f"Context (with mask): Expected {expected_context_shape}, got {context_vector.shape}"
    assert attention_weights.shape == expected_weights_shape, \
        f"Weights (with mask): Expected {expected_weights_shape}, got {attention_weights.shape}"
    assert context_vector.dtype == torch.float32
    assert attention_weights.dtype == torch.float32
    
    # Check if weights sum to 1 (approximately) where mask allows
    # Calculate sum only over non-masked elements implicitly via softmax
    sums = attention_weights.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        f"Attention weights (with mask) do not sum to 1. Sums: {sums}"
    print(f"\nBahdanauAttention shapes (with mask) OK: Context {context_vector.shape}, Weights {attention_weights.shape}")

    # --- Test without mask ---
    with torch.no_grad():
        context_vector_no_mask, attention_weights_no_mask = bahdanau_attention(
            dummy_decoder_hidden, dummy_encoder_outputs, encoder_mask=None
        )
        
    assert context_vector_no_mask.shape == expected_context_shape, \
        f"Context (no mask): Expected {expected_context_shape}, got {context_vector_no_mask.shape}"
    assert attention_weights_no_mask.shape == expected_weights_shape, \
        f"Weights (no mask): Expected {expected_weights_shape}, got {attention_weights_no_mask.shape}"
    
    sums_no_mask = attention_weights_no_mask.sum(dim=2)
    assert torch.allclose(sums_no_mask, torch.ones_like(sums_no_mask), atol=1e-6), \
        f"Attention weights (no mask) do not sum to 1. Sums: {sums_no_mask}"
    print(f"BahdanauAttention shapes (no mask) OK: Context {context_vector_no_mask.shape}, Weights {attention_weights_no_mask.shape}")


def test_code_decoder_shapes(code_decoder):
    """Tests input and output shapes of CodeDecoder (single step)."""
    # Decoder input token: (batch_size,) or (batch_size, 1) - Long type
    dummy_decoder_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE,), dtype=torch.long).to(DEVICE)
    
    # Decoder hidden state: (batch_size, hidden_size)
    dummy_decoder_hidden = torch.randn(BATCH_SIZE, GRU_HIDDEN_SIZE).to(DEVICE)
    
    # Encoder outputs (for attention): (batch_size, seq_len, encoder_output_dim)
    dummy_encoder_outputs = torch.randn(BATCH_SIZE, SEQ_LEN, LLAMA_HIDDEN_SIZE).to(DEVICE)
    
    # Encoder mask (for attention): (batch_size, seq_len) - True/1 for valid
    dummy_encoder_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool).to(DEVICE)

    # Forward pass (single step)
    with torch.no_grad():
        output_logits, updated_decoder_hidden = code_decoder(
            dummy_decoder_input, dummy_decoder_hidden, dummy_encoder_outputs, dummy_encoder_mask
        )
    output_logprobs = torch.log_softmax(output_logits, dim=-1)
    # Expected shapes
    expected_logprobs_shape = (BATCH_SIZE, VOCAB_SIZE)
    expected_hidden_shape = (BATCH_SIZE, GRU_HIDDEN_SIZE)

    assert output_logprobs.shape == expected_logprobs_shape, \
        f"LogProbs: Expected {expected_logprobs_shape}, got {output_logprobs.shape}"
    assert updated_decoder_hidden.shape == expected_hidden_shape, \
        f"Hidden: Expected {expected_hidden_shape}, got {updated_decoder_hidden.shape}"
        
    assert output_logprobs.dtype == torch.float32
    assert updated_decoder_hidden.dtype == torch.float32
    
    # Check if log probabilities are valid (sum of exp(logprobs) should be ~1)
    probs = torch.exp(output_logprobs)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6), \
        f"Log probabilities do not sum to 1 after exp(). Sums: {sums}"
        
    print(f"\nCodeDecoder shapes OK: LogProbs {output_logprobs.shape}, Hidden {updated_decoder_hidden.shape}")


def test_llama_decoder_model_shapes(llama_decoder_model):
    """Tests input and output shapes of LlamaDecoderForNextArticle."""
    # Input sequence: (batch_size, seq_len, code_iter) - Long type
    dummy_src = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN, CODE_ITER), dtype=torch.long).to(DEVICE)
    
    # Padding mask (optional): (batch_size, seq_len) - True/1 for valid tokens
    dummy_src_padding_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool).to(DEVICE)
    if SEQ_LEN > 1:
        dummy_src_padding_mask[:, -1] = False # Mask last element for testing

    # --- Test with mask ---
    with torch.no_grad():
        output_with_mask = llama_decoder_model(dummy_src, dummy_src_padding_mask)

    # Expected output shape: (batch_size, seq_len, num_vqvae_iter, vocab_size)
    expected_shape = (BATCH_SIZE, SEQ_LEN, CODE_ITER, VOCAB_SIZE)

    assert output_with_mask.shape == expected_shape, \
        f"Output (with mask): Expected {expected_shape}, got {output_with_mask.shape}"
    assert output_with_mask.dtype == torch.float32, \
        f"Output (with mask): Expected dtype torch.float32, got {output_with_mask.dtype}"
    print(f"\nLlamaDecoderForNextArticle shapes (with mask) OK: Input {dummy_src.shape} -> Output {output_with_mask.shape}")

    # --- Test without mask ---
    with torch.no_grad():
        output_no_mask = llama_decoder_model(dummy_src, src_padding_mask=None)
        
    assert output_no_mask.shape == expected_shape, \
        f"Output (no mask): Expected {expected_shape}, got {output_no_mask.shape}"
    assert output_no_mask.dtype == torch.float32, \
        f"Output (no mask): Expected dtype torch.float32, got {output_no_mask.dtype}"
    print(f"LlamaDecoderForNextArticle shapes (no mask) OK: Input {dummy_src.shape} -> Output {output_no_mask.shape}")

    # Optional: Check if outputs are valid log-probabilities per step
    # Note: The model outputs logprobs directly because CodeDecoder uses LogSoftmax
    last_step_logprobs = output_with_mask[:, :, -1, :] # Logprobs for the last code iteration
    probs = torch.exp(last_step_logprobs)
    sums = probs.sum(dim=-1) # Sum over vocab dimension
    # We expect sums to be 1 everywhere, potentially except for masked sequence positions
    # This check might be too strict if masking affects intermediate steps in complex ways
    # assert torch.allclose(sums[dummy_src_padding_mask], torch.ones_like(sums[dummy_src_padding_mask]), atol=1e-5), \
    #     "Log probabilities for last step (masked) do not sum to 1 after exp()"

# --- How to Run ---
# 1. Save your PyTorch classes into a file named `your_module.py`.
# 2. Save this test code into a file named `test_shapes.py` in the same directory.
# 3. Make sure you have pytest installed (`pip install pytest torch transformers`).
# 4. Run pytest from your terminal in that directory: `pytest -v test_shapes.py` 
#    (The `-v` flag provides verbose output).
