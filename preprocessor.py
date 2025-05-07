import json
import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

import build_vocab
import enter


class SpeechDataset(Dataset):
    def __init__(self, json_path, my_vocab, np_folder, sort_mode=False):
        if os.path.exists(json_path) == False:
            raise ValueError("No exist file")
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # Convert file to list of items
            data = [entry for k, entry in data.items()]
            if "transcript" not in data[0] or "audio" not in data[0]:
                raise ValueError("No [transcript, audio] in file")
            else:
                # Select columns and filter matching
                np_list = os.listdir(enter.NP_FOLDER)
                data = [{"transcript": d["transcript"], "audio": d["audio"]}
                        for d in data if d["audio"][:-4] + ".npy" in np_list]
                self.data = data
                self.np_folder = np_folder
                self.my_vocab = my_vocab

        self.sort_mode = sort_mode
        if self.sort_mode:
            # Sort self.data by the length of the transcript (number of words)
            self.data = sorted(
                data, key=lambda d: -len(d["transcript"].split()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_file = os.path.join(self.np_folder, item["audio"][:-4] + ".npy")
        audio_data = np.load(input_file)
        tgt_data = self.my_vocab.text_to_ids(item['transcript'])
        # ! Return into tuple
        return torch.tensor(audio_data), torch.tensor(tgt_data)


def collate_fn(batch):
    # Tuple(Tensor(128, T)), Tuple(Tensor(L))
    audio_data, tgt_data = zip(*batch)
    src_lens, tgt_lens = [], []
    for x, y in zip(audio_data, tgt_data):
        src_lens.append(x.size(1))
        tgt_lens.append(y.size(0))
    src_lens = torch.LongTensor(src_lens)
    tgt_lens = torch.LongTensor(tgt_lens)

    # ! Use 0 for pad
    padded_tgt = pad_sequence(tgt_data, batch_first=True)

    # Pad for maps:
    # unsqueeze(1): add 1 to shape
    # squeeze(1): remove 1 to shape if have dim size = 1
    spectrograms = []
    for sample in audio_data:
        spec = sample.transpose(-2, -1)
        # print(spec.shape) # (T, F)
        spectrograms.append(spec)
    spectrograms = pad_sequence(
        spectrograms, batch_first=True)

    return spectrograms, padded_tgt, src_lens, tgt_lens


def get_dataloader(use_sort_mode):
    my_vocab = build_vocab.Vocabulary(
        build_vocab.get_vocabulary(enter.FILE_NAMES))
    np_folder = enter.NP_FOLDER
    json_paths = enter.FILE_NAMES
    BS = enter.BS

    train_dataset = SpeechDataset(
        json_paths[0], my_vocab, np_folder, use_sort_mode)
    val_dataset = SpeechDataset(json_paths[1], my_vocab, np_folder)
    test_dataset = SpeechDataset(json_paths[2], my_vocab, np_folder)

    print("\n- Check train/val/test dataset:")
    print(f"--- Train Samples: {len(train_dataset):<10} | "
          f"Validation Samples: {len(val_dataset):<10} | "
          f"Test Samples: {len(test_dataset)}")
    print("--- Use sorted dataset:", train_dataset.sort_mode)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=False if use_sort_mode else True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BS, collate_fn=collate_fn, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        enter.SORT_TRAIN)

    # Example of iterating through the DataLoader
    for x, y, x_lens, y_lens in train_dataloader:
        print(x.shape)  # Shape will be (B, T, F)
        print(y.shape)  # Shape will be (B, L)
        print(x_lens)
        print(y_lens)
        print(x.dtype)
        print(y.dtype)
        break
