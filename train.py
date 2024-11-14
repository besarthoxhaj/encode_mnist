#
#
import torch
import model
import dataset
import random
import datetime
import wandb
import eval


#
#
wandb.init(project="encode_numbers")
torch.manual_seed(57)
random.seed(55)


#
#
src = dataset.Combine()
tsf = model.Transformer()
print("Params:", sum(p.numel() for p in tsf.parameters()))
opt = torch.optim.Adam(tsf.parameters(), lr=1e-4)
crt = torch.nn.CrossEntropyLoss()


#
#
for _ in range(100_000):
  tns, ipt, lbl, img = src[0]
  opt.zero_grad()
  prd = tsf(tns, ipt)
  prd = prd.reshape(-1, 12)
  lbl = lbl.reshape(-1)
  loss = crt(prd, lbl)
  loss.backward()
  opt.step()
  if _ % 1_000 == 0: wandb.log({"loss": loss.item()})
  if _ % 5_000 == 0: print("Loss", loss.item())
  if _ % 10_000 == 0: eval.run(tsf)


#
#
ts = datetime.datetime.now().strftime("%M_%H_%d")
torch.save(tsf.state_dict(), f"weights/m_{ts}.pth")
wandb.finish()
