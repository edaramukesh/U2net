from model.model import U2net
from dataloader import get_train_dataloader,get_val_dataloader
from loss_functions import loss_fn
import torch
from glob import glob
import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

root_training = "/home/mukesh/Desktop/Spyne/ai/training/U2_net/training_data"
root_validation = "/home/mukesh/Desktop/Spyne/ai/training/U2_net/validation_data"
train_data_loader = get_train_dataloader(root=root_training)
val_data_loader = get_val_dataloader(root=root_validation)
u2net = U2net().to(device)


def load_model(model:torch.nn.Module,path = "checkpoints"):
    no_of_dicts = glob(os.path.join(path,"*"))
    state_dict = ""
    save_num = 0
    for i in no_of_dicts:
        model_save = int(i.split("_")[-1].split(".")[0])
        if model_save > save_num:
            save_num = model_save
            state_dict = i
    print("No_of_dicts_available:",len(no_of_dicts))
    if state_dict !="":
        checkpoint = torch.load(state_dict)
        model.load_state_dict(state_dict=checkpoint)
        print("loaded model with:",state_dict)
    else:
        print("Model is not loaded with any previous weights")

def train(model:torch.nn.Module,optimizer:torch.optim.Adam):
    batch_losses = []
    for idx,i in enumerate(tqdm(iter(train_data_loader),desc="No of batches")):
        images,masks = i
        images = images.to(device)
        masks = masks.to(device)
        preds = model(images)
        preds = [i.to(device) for i in preds]
        loss = loss_fn(*preds,target = masks,device=device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(float(loss))
        del loss
        del images
        del masks
        del i
        del preds
        
    return batch_losses

def mainFunc(model:torch.nn.Module,strtepoch,endepoch,checkpoint="checkpoints"):
    load_model(model=model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0)
    epoch_losses=[]
    for epoch in tqdm(range(strtepoch,endepoch+1),desc="No of epochs"):
        batch_losses = train(model=model,optimizer=optimizer)
        epoch_losses.append(sum(batch_losses)/len(batch_losses))
        torch.save(model.state_dict(),f"{checkpoint}/u2net_{epoch}.pth")
        del batch_losses
    print(epoch_losses)

mainFunc(model=u2net,strtepoch=14,endepoch=17)





