#
#
import torch
import torchvision
import random
import PIL
import einops


#
#
class Combine(torch.utils.data.Dataset):
  def __init__(self):
    super().__init__()
    self.tf = torchvision.transforms.ToTensor()
    self.ds = torchvision.datasets.MNIST(root='.', download=True)
    self.ln = len(self.ds)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, _):
    idx = random.sample(range(self.ln), 4)
    acc, lbl, ipt = [], [], [10]
    for i in idx: x, y = self.ds[i]; acc.append(x); lbl.append(y); ipt.append(y)
    lbl = lbl + [11]
    img = PIL.Image.new('L', (56, 56))
    img.paste(acc[0], (0, 0))
    img.paste(acc[1], (28, 0))
    img.paste(acc[2], (0, 28))
    img.paste(acc[3], (28, 28))
    tns = self.tf(img)
    tns = einops.rearrange(tns, 'b (h ph) (w pw) -> b (h w) ph pw', ph=14, pw=14)
    ipt = torch.tensor(ipt).unsqueeze(0)
    lbl = torch.tensor(lbl).unsqueeze(0)
    return tns, ipt, lbl, img

  def show(self, patch):
    fun = torchvision.transforms.ToPILImage()
    img = fun(patch)
    img.show()


#
#
if __name__ == '__main__':
  ds = Combine()
  tns, ipt, lbl, img = ds[0]
  img.show()
  ds.show(torch.cat((tns[0, 0], tns[0, 1], tns[0, 2]), 1))
  ds.show(torch.cat((tns[0, 4], tns[0, 5], tns[0, 6]), 1))
  print(tns.shape); print("Input:", ipt); print("Label:", lbl)
