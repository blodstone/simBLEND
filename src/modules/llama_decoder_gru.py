import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import random
from transformers.models.llama import LlamaConfig, LlamaModel

class CodeEncoder(nn.Module):
    """
    Encodes sequences of codes using a Bidirectional GRU.
    Processes input with shape (batch_size, seq_len, code_iter) into
    a sequence of context vectors (batch_size, seq_len, output_size).
    """

    def __init__(self, embedding_size, hidden_size, embedding_layer, output_size, num_layers=2, dropout=0.1):
        """
        Args:
            embedding_size (int): Dimension of the code embeddings.
            hidden_size (int): Dimension of the GRU hidden states (per direction).
                                The output linear layer will map back to this size.
            embedding_layer (nn.Embedding): Pre-initialized embedding layer.
            num_layers (int): Number of GRU layers. Default is 2.
            dropout (float): Dropout probability. Default is 0.1.
        """
        super(CodeEncoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        # Bidirectional GRU 
        self.gru = nn.GRU(input_size=self.embedding_size, 
                          hidden_size=self.hidden_size, 
                          num_layers=self.num_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          dropout=dropout)
        # Input: D * num_layers * hidden_size = 2 * num_layers * hidden_size
        # Output: output_size
        self.out = nn.Linear(self.num_layers*self.hidden_size * 2, output_size)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of code IDs.
                              Shape: (batch_size, seq_len, code_iter)

        Returns:
            torch.Tensor: Encoded sequence representation.
                          Shape: (batch_size, seq_len, output_size)
        """
        # x shape: (batch_size, seq_len, code_iter)
        embedded = self.dropout(self.embedding_layer(x))
        # embedded shape: (batch_size, seq_len, code_iter, embedding_size)
        batch, seq_len, code_iter, embedding_dim = embedded.size()

        # Reshape to process each sequence element's codes independently
        # Shape becomes: (batch * seq_len, code_iter, embedding_size)
        embedded_reshape = embedded.view(batch * seq_len, code_iter, embedding_dim)  

        # gru_output shape: (batch * seq_len, code_iter, D * hidden_size) = (batch * seq_len, code_iter, 2 * hidden_size)
        # final_hidden shape: (D * num_layers, batch * seq_len, hidden_size) = (2 * num_layers, batch * seq_len, hidden_size)
        gru_output, final_hidden = self.gru(embedded_reshape)
        # Reshape final hidden state to combine layers and directions for each original sequence position
        # Before permute/reshape: (2 * num_layers, batch * seq_len, hidden_size)
        # Permute to: (batch * seq_len, 2 * num_layers, hidden_size)
        # Reshape to: (batch * seq_len, 2 * num_layers * hidden_size)
        hidden_combined = final_hidden.permute(1, 0, 2).reshape(batch*seq_len, -1)
        # hidden_combined shape: (batch * seq_len, -1)

        # Reshape to (batch, seq_len, D * num_layers * hidden_size) = (batch, seq_len, 2 * num_layers * hidden_size)
        hidden_reshaped_for_linear = hidden_combined.reshape(batch, seq_len, -1)
        # Apply linear layer to get final output per sequence position
        # Input: (batch, seq_len, 2 * num_layers * hidden_size)
        # Output: (batch, seq_len, output_size)
        output = self.out(hidden_reshaped_for_linear)
        return output

class BahdanauAttention(nn.Module):
    """
    Implements Bahdanau (additive) attention mechanism.
    """
    def __init__(self, decoder_hidden_dim: int, encoder_output_dim: int):
        """
        Args:
            decoder_hidden_dim (int): Dimension of the decoder's hidden state (query).
            encoder_output_dim (int): Dimension of the encoder's outputs (keys/values).
        """
        super(BahdanauAttention, self).__init__()
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_output_dim = encoder_output_dim
        self.Wa = nn.Linear(self.encoder_output_dim, self.decoder_hidden_dim, bias=False) # For encoder outputs
        self.Ua = nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim, bias=False)     # For decoder hidden state
        self.Va = nn.Linear(self.decoder_hidden_dim, 1, bias=False)             # To compute the score

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor, 
                encoder_mask: torch.Tensor | None =None):
        """
        Args:
            decoder_hidden (Tensor): Previous decoder hidden state (query).
                                     Shape: (batch_size, decoder_hidden_dim)
            encoder_outputs (Tensor): Encoder outputs (keys and values).
                                      Shape: (batch_size, seq_len, encoder_output_dim)
            encoder_mask (Tensor, optional): Mask indicating padding in encoder outputs.
                                             `True` or `1` for valid tokens, `False` or `0` for padding.
                                             Shape: (batch_size, seq_len). Defaults to None (no mask).

        Returns:
            context_vector (Tensor): Weighted sum of encoder outputs.
                                     Shape: (batch_size, encoder_output_dim)
            attention_weights (Tensor): Calculated attention weights.
                                        Shape: (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # Project decoder hidden state and expand for addition
        # decoder_hidden shape: (batch_size, decoder_hidden_dim)
        # projected_decoder shape: (batch_size, decoder_hidden_dim)
        projected_decoder = self.Ua(decoder_hidden)
        # expanded_decoder shape: (batch_size, 1, decoder_hidden_dim) -> (batch_size, seq_len, decoder_hidden_dim)
        expanded_decoder = projected_decoder.unsqueeze(1).repeat(1, seq_len, 1)

        # Project encoder outputs
        # encoder_outputs shape: (batch_size, seq_len, encoder_output_dim)
        # projected_encoder shape: (batch_size, seq_len, decoder_hidden_dim)

        projected_encoder = self.Wa(encoder_outputs)

        # Calculate alignment scores (energy)
        # score shape: (batch_size, seq_len, 1)
        score = self.Va(torch.tanh(projected_encoder + expanded_decoder))

        # Apply mask *before* softmax
        if encoder_mask is not None:
            # Ensure mask is boolean, shape (batch_size, seq_len)
            if encoder_mask.dtype != torch.bool:
                 encoder_mask = encoder_mask.bool()
            # Expand mask for broadcasting: (batch_size, seq_len, 1)
            mask_expanded = encoder_mask.unsqueeze(-1)
            # Fill padding scores with a large negative number for softmax
            score = score.masked_fill(~mask_expanded, -1e9) # Use inverse mask (mask is True for VALID tokens)

        # Calculate attention weights (probabilities)
        # attention_weights shape after softmax: (batch_size, seq_len, 1)
        # after transpose: (batch_size, 1, seq_len)
        attention_weights = F.softmax(score, dim=1).transpose(1, 2)

        # Calculate context vector (weighted sum of encoder outputs)
        # bmm input 1: (batch_size, 1, seq_len)
        # bmm input 2: (batch_size, seq_len, encoder_output_dim)
        # context_vector shape: (batch_size, 1, encoder_output_dim)
        context_vector = torch.bmm(attention_weights, encoder_outputs)

        # Squeeze context vector and return
        # context_vector shape: (batch_size, 1, encoder_output_dim)
        # attention_weights shape: (batch_size, 1, seq_len)
        return context_vector, attention_weights

class CodeDecoder(nn.Module):
    """
    Decodes sequences using a GRU with Bahdanau Attention.
    Predicts the next code token based on previous token, hidden state, and encoder context.
    """
    def __init__(self, embedding_size: int, 
                 gru_hidden_size: int, 
                 vocab_size: int, 
                 embedding_layer: nn.Embedding, 
                 encoder_output_dim: int,
                 dropout: float=0.1):
        """
        Args:
            embedding_size (int): Dimension of the code embeddings (decoder input).
            hidden_size (int): Dimension of the GRU hidden states.
            vocab_size (int): Size of the output vocabulary (number of possible codes).
            embedding_layer (nn.Embedding): Pre-initialized embedding layer.
            encoder_output_dim (int): Dimension of the encoder outputs (for attention context).
                                      Often 2 * hidden_size if encoder is bidirectional GRU.
            dropout (float): Dropout probability. Default is 0.1.
        """
        super(CodeDecoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.gru_hidden_size = gru_hidden_size
        self.vocab_size = vocab_size
        self.encoder_output_dim = encoder_output_dim
        self.attention = BahdanauAttention(decoder_hidden_dim=self.gru_hidden_size,
                                           encoder_output_dim=self.encoder_output_dim)
        self.gru = nn.GRU(input_size=embedding_size + encoder_output_dim, 
                          hidden_size=gru_hidden_size, 
                          num_layers=1, 
                          batch_first=True, 
                          dropout=dropout)
        self.out = nn.Linear(gru_hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self._init_output_weights()

    def _init_output_weights(self) -> None:
        """Initializes weights for the final linear layer."""
        initrange = 0.1
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange) 

    def forward(self, decoder_input, decoder_hidden, encoder_outputs, encoder_masks):
        """
        Performs one decoding step.

        Args:
            decoder_input (torch.Tensor): Input token(s) for this time step.
                                          Shape: (batch_size,) or (batch_size, 1) - expected single token per batch item.
            decoder_hidden (torch.Tensor): Previous decoder hidden state.
                                           Shape: (batch_size, hidden_size)
            encoder_outputs (torch.Tensor): Outputs from the encoder.
                                            Shape: (batch_size, seq_len, encoder_output_dim)
            encoder_mask (torch.Tensor, optional): Mask for encoder outputs (for attention).
                                                   Shape: (batch_size, seq_len). Defaults to None.

        Returns:
            output_logits (torch.Tensor): Logits over the vocabulary for the next token.
                                          Shape: (batch_size, vocab_size)
            decoder_hidden (torch.Tensor): Updated decoder hidden state.
                                           Shape: (batch_size, hidden_size)
            # Note: Attention weights could also be returned if needed
        """
        if decoder_input.dim() == 1:
            decoder_input = decoder_input.unsqueeze(1)
        
        # embedded shape: (batch_size, 1, embedding_size)
        embedded = self.dropout(self.embedding_layer(decoder_input))

        # decoder_hidden shape: (batch_size, hidden_size)
        # context_vector shape: (batch_size, 1, encoder_output_dim)
        # attention_weights shape: (batch_size, seq_len)
        context_vector, attention_weights = self.attention(decoder_hidden, encoder_outputs, encoder_masks)

        # rnn_input shape: (batch_size, 1, embedding_size + encoder_output_dim)
        rnn_input = torch.cat((embedded, context_vector), dim=2)

        # gru_output shape: (batch_size, 1, hidden_size)
        # updated_decoder_hidden shape: (1, batch_size, hidden_size)
        gru_output, updated_decoder_hidden = self.gru(rnn_input, decoder_hidden.unsqueeze(0))
        # output_squeezed shape: (batch_size, hidden_size)
        output_squeezed = gru_output.squeeze(1)
        # output_logits shape: (batch_size, vocab_size)
        output_logits = self.out(output_squeezed)
        # updated_decoder_hidden shape: (batch_size, hidden_size)
        updated_decoder_hidden = updated_decoder_hidden.squeeze(0)
        return output_logits, updated_decoder_hidden

class LlamaDecoderForNextArticleGRU(L.LightningModule):
    """
    A Decoder-Only Transformer model for sequence prediction, using Hugging Face's LlamaModel as the core.
    Takes a sequence of VQVAE indices and predicts the next indices.
    """
    def __init__(self,
                 num_vqvae_iter: int,
                 # Llama specific configurations (example values)
                 code_embedding_size: int = 384,
                 teacher_forcing_ratio: float = 2,
                 gru_hidden_size: int = 512, # Hidden size for GRU
                 # Llama specific configurations
                 hidden_size: int = 512,       # Corresponds to d_model
                 num_hidden_layers: int = 6,   # Corresponds to num_decoder_layers
                 num_attention_heads: int = 8, # Corresponds to nhead
                 intermediate_size: int = 2048,# Corresponds to d_hid
                 max_position_embeddings: int = 512, # Max sequence length Llama can handle
                 vocab_size: int = 512,      # Llama's internal vocab size (can be different from num_articles)
                 dropout_prob: float = 0.1):
        """
        Args:
            num_vqvae_iter (int): The number of code iterations (length of the code sequence per item).
            code_embedding_size (int): Dimension for the VQVAE code embeddings.
            gru_hidden_size (int): Hidden dimension for the GRU CodeEncoder/CodeDecoder.
                                   *** MUST match llama_hidden_size in this specific (flawed) implementation ***
            gru_num_layers (int): Number of layers for GRU Encoder/Decoder.
            teacher_forcing_ratio (float): Probability of using ground truth for next decoder input.
            llama_hidden_size (int): Dimensionality of the Llama model layers.
            num_hidden_layers (int): Number of hidden layers in the Llama model.
            num_attention_heads (int): Number of attention heads for Llama.
            intermediate_size (int): Dimensionality of the intermediate feed-forward layer in Llama.
            max_position_embeddings (int): Maximum sequence length for Llama.
            vocab_size (int): Vocabulary size (number of unique VQVAE codes). Used for embedding and final prediction.
            dropout_prob (float): Dropout probability (used in GRU, Llama applies its own).
        """
        super().__init__()
        self.model_type = 'LlamaDecoderForNextArticle'


        self.config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
        )

        # 2. Instantiate the Llama Model
        # We use LlamaModel which outputs hidden states, not LlamaForCausalLM which includes a head
        self.llama = LlamaModel(self.config)
        self.num_vqvae_iter = num_vqvae_iter
        self.vocab_size = vocab_size
        self.code_embedding_size = code_embedding_size
        self.embedding_layer = nn.Embedding(vocab_size, self.code_embedding_size)
        self.gru_hidden_size = gru_hidden_size
        self.decoder_num_layers = 2
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.code_encoder = CodeEncoder(embedding_size=self.code_embedding_size,
                                        hidden_size=self.gru_hidden_size,
                                        embedding_layer=self.embedding_layer, 
                                        output_size=self.config.hidden_size)
        self.code_decoder = CodeDecoder(embedding_size=self.code_embedding_size, 
                                        gru_hidden_size=self.gru_hidden_size,
                                        vocab_size=self.vocab_size,
                                        embedding_layer=self.embedding_layer,
                                        encoder_output_dim=self.config.hidden_size,
                                        dropout=dropout_prob
                                        )
        self.llama_out = nn.Linear(self.config.hidden_size, self.gru_hidden_size)
        
    def forward(self,
                src: torch.Tensor,
                src_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass using the LlamaModel.

        Args:
            src (torch.Tensor): Input sequence tensor of article IDs.
                                Assumed to be within VQVAE vocab range [0, vocab_size-1].
                                Shape: (batch_size, seq_len, code_iter).
            src_padding_mask (torch.Tensor, optional): Mask indicating padding tokens in the input.
                                                       Shape: (batch_size, seq_len).
                                                       *IMPORTANT*: This mask should use `True` or `1` for
                                                       non-padded tokens and `False` or `0` for padded tokens,
                                                       which is the opposite of the previous custom implementation.
                                                       If None, assumes no padding. Defaults to None.

        Returns:
            torch.Tensor: Output tensor representing the logits for the next article prediction
                          for each position in the sequence. Shape: (batch_size, seq_len, num_articles).
        """
        # --- Prepare Attention Mask for Hugging Face ---
        # Hugging Face models typically expect attention mask with:
        # 1 for tokens to attend to, 0 for padding tokens.
        # If src_padding_mask is provided in the old format (True for padding), invert it.
        # If src_padding_mask is None, create a mask of all ones.
        if src_padding_mask is not None:
             # Assuming input mask uses True for padding (like previous example)
             # Invert it: True -> 0 (pad), False -> 1 (attend)
             attention_mask = (~src_padding_mask).long()
        else:
            batch_size, seq_len, _ = src.size()
            # No padding mask provided, assume all tokens should be attended to
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=src.device)
        batch_size = src.size(0)
        # encoded_code shape = (batch_size, seq_len, hidden_size)
        encoded_code = self.code_encoder(src)
        device = src.device
        # --- Pass through LlamaModel ---
        # We pass `input_ids` (our src). Llama handles embeddings and positional encodings.
        # `attention_mask` tells Llama which tokens are padding.
        # LlamaModel automatically applies causal masking for decoder-only behavior.
        llama_outputs = self.llama(
            inputs_embeds=encoded_code,
            attention_mask=attention_mask,
            # `return_dict=True` is recommended for clarity
            return_dict=True
        )
        
        # Get the hidden states from the last layer
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = encoded_code.size()

        # encoder_outputs shape: (batch_size, seg_len, hidden_size) 
        encoder_outputs = llama_outputs.last_hidden_state

        # llama_hidden_states shape: (batch_size, seg_len, hidden_size) -> (batch_size, seq_len, gru_hidden_size)
        decoder_first_hiddens = self.llama_out(llama_outputs.last_hidden_state)

        # outputs shape: (batch_size, seq_len, code_iter, vocab) 
        outputs = torch.zeros(batch_size, seq_len, self.num_vqvae_iter, self.vocab_size)
        for l in range(0, seq_len):
            causal_indices = torch.arange(seq_len, device=device)
            causal_mask_for_l = (causal_indices <= l).long()
            causal_mask_for_l_expanded = causal_mask_for_l.unsqueeze(0).expand(batch_size, -1)
            # combined_decoder_mask_for_l shape: (batch_size, seq_len)
            combined_decoder_mask_for_l = attention_mask & causal_mask_for_l_expanded 
            # decoder input shape: (batch, )
            decoder_input = src[:, l, 0]
            # decoder hidden shape: (batch, gru_hidden)
            decoder_hidden = decoder_first_hiddens[:, l, :]
            for t in range(1, self.num_vqvae_iter):
                decoder_hidden = decoder_hidden.contiguous()
                # decoder_output shape: (batch_size, vocab_size)
                # decoder_hidden shape: (batch_size, gru_hidden)
                decoder_output, decoder_hidden = self.code_decoder(decoder_input, decoder_hidden, encoder_outputs, combined_decoder_mask_for_l)
                decoder_output_logprobs = F.log_softmax(decoder_output, dim=1)
                outputs[:, l, t, :] = decoder_output
                teacher_force = random.random() < self.teacher_forcing_ratio
                if t < self.num_vqvae_iter-1:
                    if teacher_force:
                        decoder_input = src[:, l, t+1]
                    else:
                        _, topi = decoder_output_logprobs.topk(1)
                        decoder_input = topi.squeeze(1).detach()
        return outputs

    def compute_loss(self, src):
        """
        Computes the loss for a batch of data.

        Args:
            src: A tensor of the input sequence (batch_size, seq_len, code_iter)

        Returns:
            torch.Tensor: The computed loss.
        """
        # Create a mask for padding tokens in the source sequence
        # Assuming padding is represented by 0 in the input sequence
        src_padding_mask = (src == 0).any(dim=-1)
        # Forward pass through the model
        outputs = self(src, src_padding_mask)
        # Reshape outputs and targets for cross-entropy loss calculation
        # outputs shape: (batch_size, seq_len, code_iter, vocab_size)
        # tgt shape: (batch_size, seq_len, code_iter)
        outputs_reshaped = outputs.view(-1, self.vocab_size)
        src_reshaped = src.view(-1)
        # Calculate cross-entropy loss
        loss = F.cross_entropy(outputs_reshaped, src_reshaped, ignore_index=0)
        return loss