import os

import numpy as np

import enter
import preprocessor

F = 128
V = 5172

T = 1528  # 1520
L = 340  # 336


# for i, file in enumerate(sorted(os.listdir(enter.NP_FOLDER))):
#     path = os.path.join(enter.NP_FOLDER, file)
#     y = np.load(path)
#     T = max(T, max([d for d in y.shape]))
#     if i % 1000 == 0:
#         print((i, T), end=" ")
# print(T)


# train_dataloader, val_dataloader, test_dataloader = preprocessor.get_dataloader(
#     enter.SORT_TRAIN)
# d = 0
# for loader in [train_dataloader, val_dataloader, test_dataloader]:
#     for x, y, x_lens, y_lens in loader:
#         T = max(T, x_lens.max().item())
#         L = max(L, y_lens.max().item())

#         d += 1
#         if d % 100 == 0:
#             print((d, T, L), end=" ")
