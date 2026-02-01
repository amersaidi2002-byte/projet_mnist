!unzip digits_IA_embarque.zip
from torch import dtype
import cv2
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Optimizer
from sklearn.metrics import  confusion_matrix


def preprocess_image(image):
  gray=255-image
  blur=cv2.GaussianBlur(gray,(3,3),0)
  _,bin=cv2.threshold(blur,125,255,cv2.THRESH_BINARY_INV)
  contours,_=cv2.findContours(bin.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  best=max(contours,key=cv2.contourArea)
  x,y,w,h=cv2.boundingRect(best)
  roi=bin[y:y+h,x:x+w]

  s=max(w,h)
  square=np.zeros((s,s),dtype=np.uint8)
  dx=(s-w)//2
  dy=(s-h)//2
  square[dy:dy+h,dx:dx+w]=roi

  img28=cv2.resize(square,(28,28))
  return img28.astype(np.float32)/255.0

class MNIST_embedded(Dataset):
  def __init__(self,train=True):
    self.MNIST=MNIST(root='./data',train=train,download=True)
  def __len__(self):
    return len(self.MNIST)

  def __getitem__(self, index):
    img,label=self.MNIST[index]

    img=np.array(img,dtype=np.uint8)

    img=preprocess_image(img)

    x=torch.from_numpy(img).unsqueeze(0)
    return x,label


train_dataset=MNIST_embedded(train=True)
train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)   # aplatir l’image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MLP().to(device)
print(device)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
criterion=nn.CrossEntropyLoss()

loss_list=[]
epochs_list=[]
model.train()
def train(model, loader, epochs=5):
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_list.append(total_loss)
        epochs_list.append(epoch)
        print(f"Epoch {epoch+1}, loss = {total_loss/len(loader):.4f}")

train(model, train_loader, epochs=30)

plt.figure(figsize=(8,3))
plt.plot(epochs_list,loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss by epochs")
plt.show()

test_dataset=MNIST_embedded(train=False)
test_loader=DataLoader(test_dataset,batch_size=64,shuffle=False)

model.eval()
y_pred=[]
y_true=[]
correct=0
total=0
with torch.no_grad():
  for x,y in test_loader:
    x,y=x.to(device),y.to(device)
    output=model(x)
    pred=output.argmax(dim=1)
    correct+=(pred==y).sum().item()
    total+=y.size(0)
    y_pred.extend(pred.cpu().numpy())
    y_true.extend(y.cpu().numpy())
  acc=100*correct/total
  print(f"Accuracy={acc:.2f}%")

plt.figure(figsize=(6, 6))
cm = confusion_matrix(y_true, y_pred)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

classes = np.arange(10)  # MNIST: 0..9
plt.xticks(classes, classes)
plt.yticks(classes, classes)

plt.xlabel("Predicted label")
plt.ylabel("True label")

# Écriture des valeurs dans chaque case
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black"
        )

plt.tight_layout()
plt.show()

