import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 * 4
block_size = 8 * 4
max_iters = 7000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('ta_input.txt', 'r', encoding = "utf-8") as f:
  text = f.read()

# Constructing Vocabulary for Character Level LM
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(chars)
print(''.join(chars))
print(vocab_size)

# Functions for encoding and decoding
stoi = { ch : i for i, ch in enumerate(chars)}
itos = { i : ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
print(data[:1000])

 # Split training set 90% and validation set 10%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # Split is to identify the train or validation set
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval() # Switching the Model to evaluation mode no computation are tracked for back prop
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train() # Switching the Model back to traning mode
  return out

class Head(nn.Module):
  """One head of self attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)  # Dimension (B, T, C)
    q = self.query(x)  # Dimension (B, T, C)
    wei = q @ k.transpose(-2, -1) * C ** -0.5 #  Dimension (B, T, C) @ (B, C, T) ----> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking the upper triangular matrix for future token being seen
    wei = F.softmax(wei, dim = -1) # Dimension (B, T, T)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v  # Dimension (B, T, T) @ (B, T, C) ----> (B, T, C)
    return out
  
class MultiHeadAttention(nn.Module):
  """ Multiple Heads of Self Attention in Parallel"""
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)  # Projection Layer
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))  # Projection Layer
    return out
  
class FeedForward(nn.Module):
  """ A Simple Linear Layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd), # Projection Layer
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  """ Transformer Block : Communication : Multi-Head Attention via Self-Attention, Communication via Feed Forward Network """

  def __init__(self, n_embd, n_head):
    # n_embd : embedding Dimension, n_head : the number of heads we like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.layer_norm1 = nn.LayerNorm(n_embd)
    self.layer_norm2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    # Residual Connection is 'x +' part
    x = x + self.sa(self.layer_norm1(x))
    # Residual Connection is 'x +' part
    x = x + self.ffwd(self.layer_norm2(x))
    return x


class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):

    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.positional_embedding_table = nn.Embedding(block_size, n_embd) # Positional embedding
    #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8 Dimensional Self Attention
    #self.ffwd = FeedForward(n_embd) # Feed Forward layer
    """
    self.blocks = nn.Sequential(
      Block(n_embd, n_head = 4),
      Block(n_embd, n_head = 4),
      Block(n_embd, n_head = 4),
      nn.LayerNorm(n_embd)
    )
    """
    self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
    self.layer_norm_f = nn.LayerNorm(n_embd) # Final Layer Norm
    self.lm_head = nn.Linear(n_embd, vocab_size) # Linear layer to project the embedding to vocab size

  def forward(self, idx, targets = None):
    B, T = idx.shape
    # idx and targets are of (B, T) tensor
    tok_emb = self.token_embedding_table(idx)  # (B, T, C)
    pos_emb = self.positional_embedding_table(torch.arange(T, device = device)) # (T, C)
    x = tok_emb + pos_emb
    # x = self.sa_heads(x)
    # x = self.ffwd(x) # Dimension (B, T, C)
    x = self.blocks(x) # Dimension (B, T, C)
    x = self.layer_norm_f(x) # Dimension (B, T, C)
    logits = self.lm_head(x)  # (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # We are converting as 2D like B*T as row and C as column
      targets = targets.view(B*T)  # For targets we have only one value as prediction
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):

    #idx is (B, T) array of indices in the current CONTEXT
    for _ in range(max_new_tokens):
      # Crop the 'idx' to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # Get the predictions
      logits, loss = self(idx_cond) # self() is forward() function
      # Get only (B, C)
      logits = logits[:, -1, :] # Dimension : (B, C)
      # Apply softmax
      probs = F.softmax(logits, dim = -1) # Dimension : (B, C)
      # Sampling from the distribution and get the 1 value
      idx_next = torch.multinomial(probs, num_samples = 1)
      # Append the sample index to the running sequence 'idx'
      idx = torch.cat((idx, idx_next), dim = 1) # Dimension : (B, T+1)

    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample the batch data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

# Save the Model
torch.save(model.state_dict(), './model.pth')
print('Model Saved')

model2 = BigramLanguageModel(vocab_size)
m2 = model2.to(device)
model2.load_state_dict(torch.load('./model.pth'))

# Let us generate from the Model
context = torch.zeros((1, 1), dtype = torch.long, device = device)
print(decode(m2.generate(context, max_new_tokens = 2000)[0].tolist()))