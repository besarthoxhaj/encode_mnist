#
#
import torch
import dataset
import model


#
#
torch.set_printoptions(linewidth=200)


#
#
src = dataset.Combine()


#
#
def run(m):
  tns, _, lbl, img = src[0]
  acc = torch.tensor([10])
  while True:
    acc = acc.unsqueeze(0)
    prd = m(tns, acc)
    print("First:", prd.shape, prd)
    break

  print("Label:", lbl)
  img.show()


#
#
if __name__ == "__main__":
  m = model.Transformer()
  m.load_state_dict(torch.load("./weights/m_38_13_14.pth", weights_only=True))
  run(m)
