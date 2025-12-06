import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
train_data=pd.read_csv("train.csv")

#{{{{{{{{{{ Preparation des données de validations :
validation_data=pd.read_csv("test.csv")
   #Comme il a pas de label on l'utilisera que apres avoir fini toutes taches de training du modele

validation_tensor=torch.tensor(validation_data.values, dtype=torch.float32)/255.0
validation_dataset=TensorDataset(validation_tensor)
validation_loader=DataLoader(validation_dataset, batch_size=32,shuffle=False)

#}}}}}}}}}}
# Les données Pandas seront convertis en Tensor pytorch
train_tensor_x= torch.tensor(train_data.drop(columns='label',axis=1).values, dtype=torch.float32)/255.0
train_tensor_y=torch.tensor(train_data['label'].values, dtype=torch.long)

# On cree des objets Dataset pour DataLoader
dataset=TensorDataset(train_tensor_x,train_tensor_y)


# On construit un train_set et un data_set(different du test_set de validation initialement telecharger sur kaggle) pour le training 
train_size=int(0.8*len(dataset))
test_size=len(dataset)-train_size
train_set, test_set=random_split(dataset,[train_size,test_size])

#Construction des DataLoader
train_loader=DataLoader(train_set,batch_size=32,shuffle=True)
test_loader=DataLoader(test_set, batch_size=32, shuffle=False)

class Simplecnn(nn.Module):
    def __init__(self):
        super(Simplecnn, self).__init__()
        self.conv1=nn.Conv2d(1,16,3)
        self.conv2=nn.Conv2d(16,32,3)

        self.c1=nn.Linear(32*5*5, 128)
        self.c2=nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28) 
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2) # Chaque bloc 2*2 est reduit à 1
        x=x.view(x.size(0),-1)
        x=F.relu(self.c1(x))
        x=self.c2(x)
        return x
    
model=Simplecnn()    
print(model)
optimizer=optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.CrossEntropyLoss()

def train(modele, loader):
    modele.train()
    for xb, yb in loader:
        y_pred=modele(xb)
        loss=loss_fn(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#for i in range(10):
train(model, train_loader)



def prediction(modele, loader):
  modele.eval()
  oui=1
  while oui == 1:
     i=int(input("entrez une ligne "))
     print(torch.argmax(modele(validation_tensor[i]),-1))
   # Exemple : ligne 0
     img_vec = validation_data.iloc[i].values
     img_tensor = torch.tensor(img_vec, dtype=torch.float32).view(1, -1)
# Prédiction
     with torch.no_grad():
      output = modele(img_tensor)
      pred = torch.argmax(output, dim=1).item()
# Affichage
     plt.title(f"Prediction du modèle : {pred}")
     #plt.axis('off')
     plt.imshow(img_vec.reshape(28, 28), cmap='gray')
     plt.savefig("verif.png")
     oui=int(input("voulez vous testez un autre nombre ?"))
prediction(model, validation_loader)