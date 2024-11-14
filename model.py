#
#
import torch


#
#
class Attention(torch.nn.Module):
  def __init__(self, is_causal=False, d=32):
    super().__init__()
    self.is_causal = is_causal
    self.W_Q = torch.nn.Linear(d, d)
    self.W_K = torch.nn.Linear(d, d)
    self.W_V = torch.nn.Linear(d, d)
    self.prj = torch.nn.Linear(d, d)
    self.sft = torch.nn.Softmax(dim=-1)

  def forward(self, q, k, v):
    # print("Attention.forward.00.q.shape", q.shape)
    # print("Attention.forward.00.k.shape", k.shape)
    # print("Attention.forward.00.v.shape", v.shape)
    Q = self.W_Q(q)
    K = self.W_K(k)
    V = self.W_V(v)
    K = K.transpose(1, 2) # given [0, 1, 2] we say: "transpose 1 with 2"
    # print("Attention.forward.01.Q.shape", Q.shape)
    # print("Attention.forward.01.K.shape", K.shape)
    # print("Attention.forward.01.V.shape", V.shape)
    # print("Attention.forward.02.self.is_causal", self.is_causal)
    s = (Q @ K) / (9 ** 0.5)
    # print("Attention.forward.02.s.shape", s.shape)
    if self.is_causal: s = s + torch.tril(torch.full_like(s, float('-inf')), diagonal=-1)
    o = self.sft(s) @ V
    # print("Attention.forward.03.o.shape", o.shape)
    o = self.prj(o)
    # print("Attention.forward.04.o.shape", o.shape)
    return o


#
#
class Forward(torch.nn.Module):
  def __init__(self, d=32):
    super().__init__()
    self.ins = torch.nn.Linear(d, d)
    self.rlu = torch.nn.ReLU()
    self.out = torch.nn.Linear(d, d)
    self.drp = torch.nn.Dropout(0.1)

  def forward(self, x):
    x = self.ins(x)
    x = self.rlu(x)
    x = self.drp(x)
    x = self.out(x)
    return x


#
#
class EncBlock(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.att = Attention(d=64)
    self.nr1 = torch.nn.LayerNorm(64)
    self.nr2 = torch.nn.LayerNorm(64)
    self.drp = torch.nn.Dropout(0.1)
    self.ffw = Forward(d=64)

  def forward(self, x):
    # print("EncBlock.forward.00.x.shape", x.shape)
    a = self.att(x, x, x)
    # print("EncBlock.forward.01.a.shape", a.shape)
    x = self.nr1(a + x)
    o = self.ffw(x)
    o = self.nr2(o + x)
    # print("EncBlock.forward.02.o.shape", o.shape)
    return o

#
#
class Encoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Linear(196, 64)
    self.pos = torch.nn.Embedding(16, 64)
    self.drp = torch.nn.Dropout(0.1)
    layers = [EncBlock() for _ in range(12)]
    self.bks = torch.nn.ModuleList(layers)
    self.prj = torch.nn.Linear(64, 32)

  def forward(self, x):
    # print("Encoder.forward.00.x.shape", x.shape)
    x = x.view(1, 16, -1) # [batch, patch, pix]
    n = torch.arange(16)
    x = self.emb(x)
    x = x + self.pos(n)
    # print("Encoder.forward.01.x.shape", x.shape)
    x = self.drp(x)
    for b in self.bks: x = b(x)
    # print("Encoder.forward.02.x.shape", x.shape)
    x = self.prj(x)
    # print("Encoder.forward.03.x.shape", x.shape)
    return x


#
#
class DecBlock(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.c_a = Attention(is_causal=True)
    self.x_a = Attention()
    self.nr1 = torch.nn.LayerNorm(32)
    self.nr2 = torch.nn.LayerNorm(32)
    self.nr3 = torch.nn.LayerNorm(32)
    self.drp = torch.nn.Dropout(0.1)
    self.ffw = Forward()

  def forward(self, e, x):
    # print("DecBlock.forward.00.e.shape", e.shape)
    # print("DecBlock.forward.00.x.shape", x.shape)
    c = self.c_a(x, x, x)
    # print("DecBlock.forward.01.c.shape", c.shape)
    c = self.nr1(c + x)
    a = self.x_a(c, e, e)
    a = self.nr2(a + c)
    o = self.ffw(x)
    o = self.nr3(o + x)
    # print("DecBlock.forward.02.o.shape", o.shape)
    return o


#
#
class Decoder(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(12, 32)
    self.pos = torch.nn.Embedding(12, 32)
    self.drp = torch.nn.Dropout(0.1)
    layers = [DecBlock() for _ in range(6)]
    self.bks = torch.nn.ModuleList(layers)
    self.prj = torch.nn.Linear(32, 12)

  def forward(self, e, x):
    # print("Decoder.forward.00.e.shape", e.shape)
    # print("Decoder.forward.00.x.shape", x.shape)
    n = torch.arange(x.shape[-1])
    # print("Decoder.forward.01.n", n)
    x = self.emb(x) + self.pos(n)
    # print("Decoder.forward.02.x.shape", x.shape)
    x = self.drp(x)
    for b in self.bks: x = b(e, x)
    x = self.prj(x)
    return x


#
#
class Transformer(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.enc = Encoder()
    self.dec = Decoder()

  def forward(self, i, x):
    # print("Transformer.forward.00.i.shape", i.shape)
    # print("Transformer.forward.00.x.shape", x.shape)
    e = self.enc(i)
    # print("Transformer.forward.01.e.shape", e.shape)
    return self.dec(e, x)


#
#
if __name__ == '__main__':
  t = Transformer()
  params = sum(p.numel() for p in t.parameters())
  # print("Parameters:", params)
  # print("Architecture:", t)
