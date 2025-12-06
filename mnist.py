
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import random 

seed = 42  # choisis un nombre fixe
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_csv=pd.read_csv("train.csv")
data= TensorDataset((torch.tensor(train_csv.drop(columns='label', axis=1).values,dtype=torch.float32)/255.0),torch.tensor(train_csv['label'].values, dtype=torch.long))


# Separer nos données en train/test set:

train_size=int(0.8*len(data))
test_size=len(data)-train_size
dataset_train, dataset_test=random_split(data,[train_size,test_size])
print("train_size = ",len(dataset_train)," et puis test_size = ", len(dataset_test))

# DataLoader pour chacun:

train_loader=DataLoader(dataset_train, batch_size=64, shuffle=True)
test_loader=DataLoader(dataset_test,batch_size=64, shuffle=False)

class Simplecnn(nn.Module):
    def __init__(self):
        super(Simplecnn, self).__init__()
        self.conv1=nn.Conv2d(1,16,3)
        self.conv2=nn.Conv2d(16,32,3)

        self.c1=nn.Linear(32*5*5,128)
        self.c2=nn.Linear(128,10)

    def forward(self, x):
        x=x.view(-1,1,28,28)
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2)
        x=x.flatten(1)
        x=F.relu(self.c1(x))
        x=self.c2(x)
        return x

model=Simplecnn()
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001)

def train(mod, load):
    mod.train()
    for x_batch, y_batch in load:
      y_pred=mod(x_batch)
      loss=loss_fn(y_pred, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


def evaluation(modele, load):
    modele.eval()
    nb_correct, total =0, float(len(load.dataset))
    with torch.no_grad():
      for xb,yb in load:
         y_predit=modele(xb)
         y_found=torch.argmax(y_predit,dim=1)
         nb_correct+=(y_found==yb).float().sum().item()
      print("le nombre de reponse bon trouvé : ", nb_correct)
      return nb_correct/total
