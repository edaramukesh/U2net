import torch
from model.model import U2net
from PIL import Image
import torchvision.transforms as T
from loss_functions import loss_fn


u2net = U2net()
state_dict = torch.load("checkpoints/u2net_17.pth")
u2net.load_state_dict(state_dict=state_dict)
im = Image.open("/home/mukesh/Desktop/Spyne/ai/training/U2_net/training_data/images/HipHop_HipHop4_C1_00810.png")
mask = Image.open("/home/mukesh/Desktop/Spyne/ai/training/U2_net/training_data/masks/HipHop_HipHop4_C1_00810.png")
im = T.Compose([T.PILToTensor(),T.Resize([320,320])])(im)/255
mask = T.Compose([T.PILToTensor(),T.Resize([320,320])])(mask)/255
im = im.unsqueeze(dim=0)
mask = mask.unsqueeze(dim=0)

pred = u2net(im)

print(float(loss_fn(*pred,target=mask,device="cpu")))
pred = pred[0].squeeze(dim=0)
pred = T.Compose([T.ToPILImage()])(pred)
pred.show()
# im = im.squeeze(dim=0)
# newim = torch.tensor(())
# for i in im:
#     i = i*pred


def test():
    pass