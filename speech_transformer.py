import math

import torch
import torch.nn.functional as F
from torch import nn, optim

import conv_extractor

################################################################################
# Sub Layers ###################################################################
################################################################################


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        probs = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(probs, V)

    def split_heads(self, x):
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(output))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # regis buffer


################################################################################
# Encoder and Decoder ##########################################################
################################################################################

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        x = x + self.dropout(self.self_attn(self.norm1(x),
                             self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = PositionWiseFeedForward(
            d_model, d_ff, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x),
                             self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x),
                             enc_output, enc_output, src_mask))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x


################################################################################
# Speech Transformer ###########################################################
################################################################################

class SpeechTransformer(nn.Module):
    def __init__(self, input_dim, tgt_vocab_size, pos_len_encoder, pos_len_decoder, d_model=256, num_heads=4, d_ff=2048, num_e=12, num_d=6, embedded_dropout=0.1):
        super().__init__()
        self.conv = conv_extractor.VGGExtractor(input_dim=input_dim)
        self.proj = nn.Linear(self.conv.get_output_dim(), d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoder = PositionalEncoding(d_model, pos_len_encoder)
        self.positional_decoder = PositionalEncoding(d_model, pos_len_decoder)
        self.dropout = nn.Dropout(embedded_dropout)
        self.max_new_tokens = pos_len_decoder

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_e)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_d)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def create_mask_from_len(self, lengths, max_len):
        if max_len is None:
            max_len = torch.max(lengths).item()
        return (torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).int()

    def generate_mask(self, src_lens, tgt_lens, real_src, real_tgt, device):
        src_max_len, tgt_max_len = real_src.size(1), real_tgt.size(1)
        src_mask = self.create_mask_from_len(
            src_lens, src_max_len).to(device).unsqueeze(1).unsqueeze(2)
        tgt_pad_mask = self.create_mask_from_len(
            tgt_lens, tgt_max_len).to(device).unsqueeze(1).unsqueeze(2)
        lookahead_mask = torch.tril(torch.ones(
            tgt_max_len, tgt_max_len, device=device)).bool()
        lookahead_mask = lookahead_mask.unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_pad_mask & lookahead_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt, src_lens, tgt_lens):
        # src: [B, T, 128]
        # tgt: [B, S]
        tgt_emb = self.dropout(
            self.positional_decoder(self.decoder_embedding(tgt)))
        src_emb, src_conv_lens = self.conv(src, src_lens)
        src_emb = self.dropout(self.positional_encoder(self.proj(src_emb)))

        src_mask, tgt_mask = self.generate_mask(
            src_conv_lens, tgt_lens, src_emb, tgt, src.device)

        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask)
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, src_emb, src_mask, tgt_mask)

        return self.fc(tgt_emb)


################################################################################
# Main for Testing #############################################################
################################################################################

def greedy_search_decode(model: SpeechTransformer, src, src_lens, sos_id, eos_id, max_len):
    if max_len > model.max_new_tokens - 2:
        print(
            f"WAR: Limit 'max_len' from {max_len} to {model.max_new_tokens - 2}")
        max_len = model.max_new_tokens - 2

    model.eval()
    device = src.device
    batch_size = src.size(0)

    with torch.no_grad():
        # Step 1: Convolutional Feature Extraction
        src_emb, src_conv_lens = model.conv(src, src_lens)
        src_emb = model.dropout(model.positional_encoder(model.proj(src_emb)))

        # Step 2: Create Source Mask
        src_mask = model.create_mask_from_len(
            src_conv_lens, src_emb.size(1)).to(device)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        # Step 3: Encode Source
        memory = src_emb
        for layer in model.encoder_layers:
            memory = layer(memory, src_mask)

        # Step 4: Initialize decoder input with <sos>
        ys = torch.full((batch_size, 1), sos_id,
                        dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Step 5: Decode tokens one-by-one
        for _ in range(max_len):
            tgt_emb = model.dropout(
                model.positional_decoder(model.decoder_embedding(ys))
            )

            tgt_lens = torch.full((batch_size,), ys.size(
                1), dtype=torch.long, device=device)
            _, tgt_mask = model.generate_mask(
                src_conv_lens, tgt_lens, src_emb, ys, device)

            for layer in model.decoder_layers:
                tgt_emb = layer(tgt_emb, memory, src_mask, tgt_mask)

            logits = model.fc(tgt_emb[:, -1:, :])  # [B, 1, vocab_size]
            next_token = logits.argmax(dim=-1)     # [B, 1]
            ys = torch.cat([ys, next_token], dim=1)  # Expand ys

            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

    return ys


if __name__ == "__main__":
    # Create case
    tgt_vocab_size = 8
    max_frames = 100
    max_new_tokens = 100

    transformer = SpeechTransformer(
        128, tgt_vocab_size, max_frames // 4, max_new_tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(),
                           lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # Test inference loss
    src = torch.rand(4, max_frames, 128)
    src_lens = torch.LongTensor(
        [max_frames, max_frames-10, max_frames-20, max_frames])
    tgt = torch.randint(0, tgt_vocab_size, (4, max_new_tokens))
    tgt_lens = torch.LongTensor(
        [max_new_tokens, max_new_tokens-10, max_new_tokens-4, max_new_tokens])
    print("ID word max:", f"{tgt.max()}/{tgt_vocab_size}")

    with torch.no_grad():
        output = transformer(
            src, tgt[:, :-1], src_lens, tgt_lens)  # Predict next word, so not use eos
        loss = criterion(output.view(-1, tgt_vocab_size),
                         tgt[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {loss.item()}")

    # Generation ids
    sos_id = 1  # Replace with your actual <sos> token id
    eos_id = 2  # Replace with your actual <eos> token id

    with torch.no_grad():
        output_seq = greedy_search_decode(
            transformer, src, src_lens, sos_id=1, eos_id=2, max_len=50)
        print("Decoded token IDs:", output_seq.shape, output_seq)
