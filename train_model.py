import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import enter
import preprocessor
import score
import speech_transformer

LOG_STEP = 100
sos_id, eos_id, pad_id = 1, 2, 0


def set_all_seed(seed=42):
    random.seed(seed)  # Set seed for random
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs


def get_learning_rate(step, warmup_steps, d):
    step += 1
    # Follow attention is all you need
    lr = d**(-0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    # if step < warmup_steps:
    #     lr *= 10
    # else:
    #     lr *= 10.0 / (step / 10e3)
    return lr


# ! Tune hypers for training
def train(model: speech_transformer.SpeechTransformer, dataloader, val_dataloader, criterion, optimizer, device, num_epochs=30,
          warmup_steps=4000, max_steps=100*1000, d_model=256, weights_folder="points"):
    import os
    os.makedirs(weights_folder, exist_ok=True)
    set_all_seed()

    for epoch in range(num_epochs):
        if epoch * len(dataloader) > max_steps:
            break
        model.train()
        total_loss = 0

        for step, _data in enumerate(dataloader):
            src, tgt, src_lens, tgt_lens = [x.to(device) for x in _data]

            # Update learning rate
            lr = get_learning_rate(
                step + epoch * len(dataloader), warmup_steps, d_model)  # ! Not same in the paper
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            optimizer.zero_grad()
            # Adjust lengths as needed
            output = model(src, tgt[:, :-1], src_lens, tgt_lens)
            loss = criterion(output.view(-1, output.size(-1)),
                             tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            # ! Clip gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % LOG_STEP == 0:  # Log every 10 batches
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step}, {step + epoch * len(dataloader)}], Acc.Loss: {total_loss / (step+1):.4f}, lr: {lr:.6f}")

        # Sumerize epoch
        print(
            f"\nSummary Epoch [{epoch+1}/{num_epochs}], Epoch Loss: {total_loss / len(dataloader):.4f}")
        # Save checkpoint every epoch
        path = os.path.join(weights_folder, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), path)
        print(f"- Saved model to {path}")
        # Evaluate the model
        tmp = validate(model, val_dataloader, criterion, device)
        print(f"- Val Loss: {tmp[0]:.4f}")
        print(f"- WER: {tmp[1]:.4f}")


def validate(model: speech_transformer.SpeechTransformer, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_wer = 0

    for step, _data in enumerate(dataloader):
        src, tgt, src_lens, tgt_lens = [x.to(device) for x in _data]

        # Forward pass
        with torch.no_grad():
            output = model(src, tgt[:, :-1], src_lens, tgt_lens)
            loss = criterion(output.view(-1, output.size(-1)),
                             tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
            gnt = speech_transformer.greedy_search_decode(
                model, src, src_lens, sos_id=sos_id, eos_id=eos_id, max_len=tgt_lens.max().item() + 2)

        wer = score.calculate_batch_wer(
            tgt=tgt, gnt=gnt, sos_id=sos_id, eos_id=eos_id, pad_id=pad_id)
        total_wer += wer

    avg_loss = total_loss / len(dataloader)
    avg_wer = total_wer / len(dataloader)
    return avg_loss, avg_wer


def get_model(device):
    set_all_seed()
    model = speech_transformer.SpeechTransformer(
        input_dim=128, tgt_vocab_size=5172, pos_len_encoder=1528, pos_len_decoder=340)
    return model.to(device=device)


def get_criterion_adam(model: speech_transformer.SpeechTransformer):
    set_all_seed()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ! Not same in the paper
    optimizer = optim.Adam(model.parameters(), lr=1e-5,
                           betas=(0.9, 0.98), eps=1e-9)  # Reduced learning rate
    return criterion, optimizer


if __name__ == "__main__":
    # Create dataloader
    # train_dataloader, val_dataloader, test_dataloader = preprocessor.get_dataloader(
    #     enter.SORT_TRAIN)
    set_all_seed()
    train_dataloader, val_dataloader, test_dataloader = preprocessor.get_dataloader(
        use_sort_mode=False)

    # Initialize model, criterion, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nInfo: Use device is {device}")

    model = get_model(device=device)
    criterion, optimizer = get_criterion_adam(model=model)
    print(model)

    warmup_steps = 4*1000  # !!!
    print(f"\nWarmup_step = {warmup_steps}")
    lr_max, lr_list = 0, []
    for step in range(0, 10**5, 500):
        lr = get_learning_rate(step, warmup_steps, 256)
        lr_max = max(lr_max, lr)
        lr_list.append(lr)
        print(f"{lr:.6f}", end=" ")
    print(f"\n{1e-3}, {lr_max:.6f}")

    ### Train ###
    if device == torch.device('cuda'):
        train(model, train_dataloader, test_dataloader,
              criterion, optimizer, device, warmup_steps=warmup_steps)

     ### Plot learning rate ###
    else:
        import matplotlib.pyplot as plt

        # Plot the learning rates
        plt.figure(figsize=(10, 5))
        plt.plot(lr_list, label='Learning Rate', color='blue')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps (every 500 steps)')
        plt.ylabel('Learning Rate')
        plt.grid()
        plt.legend()
        plt.show()  # Display the plot
