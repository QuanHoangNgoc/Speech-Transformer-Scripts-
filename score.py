import numpy as np
import torch


def calculate_wer(reference, hypothesis):
    """Calculate Word Error Rate (WER)"""
    reference = reference.split()
    hypothesis = hypothesis.split()
    # print(reference, hypothesis)

    # Create a distance matrix
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=int)

    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,      # Deletion
                          d[i][j - 1] + 1,      # Insertion
                          d[i - 1][j - 1] + cost)  # Substitution

    return d[len(reference)][len(hypothesis)] / len(reference) if len(reference) > 0 else float('inf')


def norm_ids_string(ids_string, sos_id, eos_id, pad_id):
    text = " "
    for s in ids_string.split():
        if int(s) in [sos_id, pad_id]:
            continue
        if int(s) == eos_id:
            break
        text = text + s + " "
    return text


def calculate_batch_wer(tgt, gnt, sos_id, eos_id, pad_id):
    """Calculate WER for a batch of target and predicted sequences."""
    total_wer = 0
    num_samples = tgt.size(0)

    for i in range(num_samples):
        # Convert tensors to strings
        reference = " ".join(map(str, tgt[i].cpu().numpy()))
        hypothesis = " ".join(map(str, gnt[i].cpu().numpy()))
        reference = norm_ids_string(reference, sos_id, eos_id, pad_id)
        hypothesis = norm_ids_string(hypothesis, sos_id, eos_id, pad_id)

        # Calculate WER for the current sample
        total_wer += calculate_wer(reference, hypothesis)

    avg_wer = total_wer / num_samples if num_samples > 0 else float('inf')
    return avg_wer


# Example usage
if __name__ == "__main__":
    # Assuming tgt and pred are your target and predicted tensors
    tgt = torch.tensor([[1, 4, 5, 6, 2], [1, 5, 7, 2, 0]]
                       )  # Example target tensor
    gnt = torch.tensor([[1, 4, 5, 6, 2], [1, 5, 7, 0, 2]]
                       )  # Example gnt tensor

    wer = calculate_batch_wer(tgt, gnt, 1, 2, 0)
    print(f"Batch WER: {wer:.4f}")
