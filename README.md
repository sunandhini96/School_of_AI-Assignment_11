# School_of_AI-Assignment_11


# dataset link : 

https://drive.google.com/file/d/1au1rc_hcWUfSl5t0ipesuvCTr6xZq-wc/view?usp=share_link

dataset collected from imbd, wikitext data, snli data, cola data datasets. Here we collected data some of the texts data in each data.

## data file path

https://drive.google.com/file/d/1au1rc_hcWUfSl5t0ipesuvCTr6xZq-wc/view?usp=share_link

dataset collected from imbd, wikitext data, snli data, cola data datasets. Here we collected data some of the texts data in each data.

# BERT training...

Training logs:

it: 0  | loss 11.16  | Δw: 7.708

it: 10  | loss 5.64  | Δw: 2.024

it: 20  | loss 4.53  | Δw: 0.634

it: 30  | loss 4.17  | Δw: 0.302

it: 40  | loss 3.9  | Δw: 0.205

it: 50  | loss 3.64  | Δw: 0.133

it: 60  | loss 3.39  | Δw: 0.105

it: 70  | loss 3.15  | Δw: 0.094

it: 80  | loss 2.91  | Δw: 0.085

it: 90  | loss 2.68  | Δw: 0.077

it: 100  | loss 2.44  | Δw: 0.071

it: 110  | loss 2.22  | Δw: 0.066

it: 120  | loss 2.0  | Δw: 0.059

it: 130  | loss 1.79  | Δw: 0.054

it: 140  | loss 1.59  | Δw: 0.049

it: 150  | loss 1.41  | Δw: 0.045

it: 160  | loss 1.24  | Δw: 0.04

it: 170  | loss 1.09  | Δw: 0.036

it: 180  | loss 0.96  | Δw: 0.033

it: 190  | loss 0.84  | Δw: 0.03

it: 200  | loss 0.74  | Δw: 0.027





# We trained GPT model:

# Sparse attention head class:
class SparseAttentionHead(nn.Module):
    """
    One head of the self-attention layer with sparse attention
    """

    def __init__(self, head_size, num_embed, block_size, dropout,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Linear(num_embed, head_size * num_heads, bias=False)
        self.query = nn.Linear(num_embed, head_size * num_heads, bias=False)
        self.value = nn.Linear(num_embed, head_size * num_heads, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, self.num_heads,T, self.head_size)
        q = self.query(x).view(B, self.num_heads,T, self.head_size)
        v = self.value(x).view(B, self.num_heads,T, self.head_size)

        # Compute attention scores
        # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        # Apply sparse mask to attention scores
        mask = torch.ones(T, T)
        # Allow each token to attend to itself and the previous tokens
        for i in range(T):
            for j in range(i, max(0, i - self.num_heads), -1):
                mask[i, j] = 1
        mask = mask.to(x.device)
        wei = wei.masked_fill(mask == 0, float("-inf"))

        # Tril matrix is used to mask future positions
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Weighted aggregation of the values
        # (B, num_heads, T, T) @ (B, num_heads, T, head_size) -> (B, num_heads, T, head_size)
        out = wei @ v
        # (B, T, num_heads, head_size) -> (B, T, num_heads * head_size)
        out = out.view(B, T, -1)
        return out

# Training log :

step          0 | train loss 3.7530 | val loss 7.0727

step         10 | train loss 3.9015 | val loss 7.3473

step         20 | train loss 3.8938 | val loss 7.0711

step         30 | train loss 3.9754 | val loss 7.1074

step         40 | train loss 3.9272 | val loss 7.0237

step         50 | train loss 3.9315 | val loss 6.8460

step         60 | train loss 3.8266 | val loss 6.9023

step         70 | train loss 3.9328 | val loss 6.8765

step         80 | train loss 3.9035 | val loss 6.8372

step         90 | train loss 3.8801 | val loss 6.8643

step        100 | train loss 3.8784 | val loss 6.8913

step        110 | train loss 3.8990 | val loss 7.0001

step        120 | train loss 3.9180 | val loss 6.9625

step        130 | train loss 3.8650 | val loss 6.9532

step        140 | train loss 3.8525 | val loss 6.9873

step        150 | train loss 3.9149 | val loss 6.9715

step        160 | train loss 3.9298 | val loss 6.8242

step        170 | train loss 3.8535 | val loss 6.9350

step        180 | train loss 3.8898 | val loss 6.9946

step        190 | train loss 3.9201 | val loss 6.9643

step        200 | train loss 3.8471 | val loss 6.9533


# Output :

[PAD] or not bad the film, not probably. < br / > < br / > < br / > preston is never precisely by me, women and dragged proposed an actress have surprise two pilasters in dawson's oils and turns back to quietly empire inctuaries as his contract. " " " " and his ragged reagents, microbes = " " " " = = post @ - @ = = " " " " " = = december – west – aerodrome licensing = = = " "

