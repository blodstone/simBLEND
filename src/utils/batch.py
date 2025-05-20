import torch
from typing import TypedDict


def pad_sequence(sequence, batch_size, max_len, pad):
        masks = [[1 for _ in sequence[i]] for i in range(batch_size)]
        for i in range(batch_size):
            if len(sequence[i]) < max_len:
                sequence[i] += [pad] * (max_len - len(sequence[i]))
                masks[i] += [0] * (max_len - len(masks[i]))
        return masks, sequence

def pad_sequence_token(padded_sequence, batch_size, max_token_len, pad):
        # masks = [[[0 if token == '<pad' else 1 for token in padded_sequence[i][j]] for j in range(len(padded_sequence[i]))] for i in range(batch_size)]
        masks = []
        new_padded_sequence = []
        for i in range(batch_size):
            masks_item = []
            padded_seq_item = []
            for j in range(len(padded_sequence[i])):
                mask = [1 if token != pad else 0 for token in padded_sequence[i][j]]
                padding_needed = max(0, max_token_len - len(padded_sequence[i][j]))
                padded_tokens = padded_sequence[i][j] + [pad] * padding_needed
                mask += [0] * padding_needed
                masks_item.append(mask)
                padded_seq_item.append(padded_tokens)
            masks.append(masks_item)
            new_padded_sequence.append(padded_seq_item)
        return masks, new_padded_sequence
