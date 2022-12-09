from torch.utils.data import DataLoader
from dataset import ExteriorDataset

def get_train_dataloader(root,batchsize=4,shuffle=True,num_workers=4,pin_memory=False):
    print("Training: ")
    dataset = ExteriorDataset(root)
    train_dataloader = DataLoader(dataset=dataset,batch_size=batchsize,num_workers=num_workers,shuffle=shuffle,pin_memory=pin_memory)
    return train_dataloader

def get_val_dataloader(root,batchsize=4,shuffle=False,num_workers=4,pin_memory=False):
    print("Validation: ")
    dataset = ExteriorDataset(root)
    val_dataloader = DataLoader(dataset=dataset,batch_size=batchsize,shuffle=shuffle,num_workers=num_workers,pin_memory=pin_memory) 
    return val_dataloader

# loader = get_train_dataloader(root="/home/mukesh/Desktop/Spyne/ai/training/U2_net/training_data")
# print(len(loader))
# for i in iter(loader):
#     print(i)
#     break

