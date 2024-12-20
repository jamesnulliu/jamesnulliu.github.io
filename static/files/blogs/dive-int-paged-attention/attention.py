import torch
from torch import nn


class MultiHeadAttentionKernel(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_size: int = embed_dim // num_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """

        Parameters
        ----------
        q : torch.Tensor; Shape: (q_len, embed_dim)

        k : torch.Tensor; Shape: (kv_len, embed_dim)

        v : torch.Tensor; Shape: (kv_len, embed_dim)

        Note
        ----
        When prefilling, q_len equals to seq_len (number of tokens in the input seq);
        When decoding, q_len equals to 1, refering to the newly generated token. (Based
        on different sampling strategies, q_len could be larger than 1.)
        """

        q_len, kv_len = q.size(0), k.size(0)
        # q: (num_heads, q_len, head_size)
        q = q.view(q_len, self.num_heads, self.head_size).transpose(0, 1)
        # k: (num_heads, kv_len, head_size)
        k = k.view(kv_len, self.num_heads, self.head_size).transpose(0, 1)
        # v: (num_heads, kv_len, head_size)
        v = v.view(kv_len, self.num_heads, self.head_size).transpose(0, 1)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_size, dtype=torch.float32)
        )

        # logits: (num_heads, q_len, kv_len)
        logits = torch.softmax(attn_weights, dim=-1)

        # out: (num_head, q_len, head_size)
        out = torch.matmul(logits, v)
        # out: (q_len, embed_dim)
        out = out.transpose(0, 1).reshape(q_len, self.embed_dim)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.attn_kernel = MultiHeadAttentionKernel(embed_dim, num_heads)

    def forward(self, seq: torch.Tensor):
        """
        Parameters
        ----------
        seq : torch.Tensor; Shape: (1, embed_dim)
            Input sequnce, containing `seq_len` tokens, and each token have been embedded
            to a `(embed_dim,)` tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Attention output, cached K and cached V.
        """

        # q: (seq_len, embed_dim)
        q = self.Wq(seq)
        # k: (seq_len, embed_dim)
        k = self.Wk(seq)
        # v: (seq_len, embed_dim)
        v = self.Wv(seq)

        # out: (seq_len, embed_dim)
        out = self.Wo(self.attn_kernel(q, k, v))

        return out, k, v


class CachedMultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)

        self.attn_kernel = MultiHeadAttentionKernel(embed_dim, num_heads)

    def forward(
        self, seq: torch.Tensor, k_cached: torch.Tensor, v_cached: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        seq : torch.Tensor; Shape: (1, embed_dim)
            Input sequnce, containing only ONE newly generated token.
        k_cached : torch.Tensor; Shape: (kv_len, embed_dim)
            Cached K.
        v_cached : torch.Tensor; Shape: (kv_len, embed_dim)
            Cached V.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Attention output, cached K and cached V.

        Note
        ----
            When decoing, the input seq only has ONE newly generated token.
        """

        # q: (1, embed_dim)
        q = self.Wq(seq)
        # k: (1, embed_dim)
        k = self.Wk(seq)
        # v: (1, embed_dim)
        v = self.Wv(seq)

        # k_cached: (kv_len + 1, embed_dim)
        k_cached = torch.cat([k_cached, k], dim=0)
        # v_cached: (kv_len + 1, embed_dim)
        v_cached = torch.cat([v_cached, v], dim=0)

        # out: (seq_len, embed_dim)
        out = self.Wq(self.attn_kernel(q, k_cached, v_cached))

        return out, k_cached, v_cached


if __name__ == "__main__":
    seq_len = 64
    vocab_size = 1024
    embed_dim = 128
    num_heads = 32

    embedder = nn.Embedding(vocab_size, embed_dim)
    proj_to_vocab = nn.Linear(embed_dim, vocab_size)
    mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    cached_mha = CachedMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # prompt is a sentence including seq_len words, and each word can be represented with
    # one or multiple integers in range [0, vocab_size).
    # For example, prompt of ["fly", "me", "to", "the"] may be [1, 0, 1023, 5].
    prompt = torch.randint(0, vocab_size, (seq_len,))
    print(f"Original prompt shape: {prompt.shape}")  # (seq_len, )

    # Embedd each word from a integeral scalar to a vector with size embed_dim, so now we
    # get a new tensor with shape (seq_len, embed_dim).
    prompt = embedder(prompt)
    print(f"Embedded prompt shape: {prompt.shape}")  # (seq_len, embed_dim)

    # Prefilling ========================================================================
    #   Input the whole seq and use MHA to calculate out, k and v.
    # Here we omit the other parts of the model only keeping one attention layer in one 
    # Transformer.
    # out, k, v: (seq_len, embed_dim).
    out, k, v = mha(prompt)
    print(f"Out shape: {out.shape}, k shape: {k.shape}, v shape: {v.shape}")

    # logits: (vocab_size,)
    logits = proj_to_vocab(out[-1])
    # NOTE:
    #   After mapping to (vocab_size,) and applying softmax, each value in probs is now 
    #   representing the probabiliy of "the next token being this index".
    #   For example, if vocab_size is 5, probs could be [0.3, 0.2, 0.4, 0.1, 0.0],
    #   showing that the probability of "the next token being value 0" is 0.3, and
    #   the probability of "the next token being value 2" is "0.4", etc.
    # probs: (vocab_size,)
    probs = torch.softmax(logits, dim=-1)
    # next_token: (1,)
    next_token = torch.argmax(probs)
    print(f"Next token from preilling: {next_token}")

    # Decoding ==========================================================================
    #   Use cached k and v, input only the new generated token from last round. This 
    #   procedure can be a loop.
    # prompt: (1, embed_dim)
    prompt = embedder(next_token)
    # out: (1, embed_dim)
    # updated_k, updated_v: (seq_len + 1, embed_dim)
    out, updated_k, updated_v = cached_mha(prompt, k, v)
    print(
        f"Out shape: {out.shape}, updated k shape: {updated_k.shape}, updated v shape: "
        f"{updated_v.shape}"
    )

    # next_token: (1,)
    next_token = torch.argmax(torch.softmax(logits(proj_to_vocab[out[-1]]), dim=-1))
    print(f"Next token from decoding: {next_token}")