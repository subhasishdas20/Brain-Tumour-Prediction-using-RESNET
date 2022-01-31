#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import torchvision.models as models 


# In[22]:


data_path = r"D:\Python class\Brain_Tumor\brain_tumor_dataset"


# In[23]:


img_size = 100
img_transform = transforms.Compose([
transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])


# In[24]:


img_data = ImageFolder(root = data_path , transform=img_transform)


# In[25]:


img_data.class_to_idx


# In[26]:


len(img_data)


# In[27]:


train_data , val_data , test_data = random_split(img_data , [200,40,13])


# In[28]:


batch_size = 32

train_loader = DataLoader(train_data ,batch_size= batch_size , shuffle =True)
val_loader = DataLoader(val_data,batch_size = batch_size , shuffle =False)


# In[29]:


for image,label in train_loader:
    print(image.shape, label.shape)
    break


# In[30]:


def show_image(data):
    for images , labels in data:
        plt.figure(figsize = (16,10))
        plt.imshow(make_grid(images,nrow=16).permute(1,2,0))
        plt.show()
        break


# In[31]:


show_image(train_loader)


# In[32]:


show_image(val_loader)


# In[47]:


model_conv = torchvision.models.resnet18(pretrained=True)
for parm in model_conv.parameters():   ## Gradient is set false - there wont be no update in weights. conv layer will be freezed
  parm.requires_grad = False

num_fts = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_fts,2)


# In[49]:


model_conv


# In[35]:


##n_features=model_VGG.classifier[-1]


# In[36]:


##n_features.in_features


# In[37]:


##model_VGG.classifier=nn.Linear(n_features.in_features,out_features=2)


# In[50]:


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_conv.parameters(), lr = 0.001)


# In[4]:


torch.cuda.is_available()


# In[53]:


def train(model, loss_fn, optimizer, epochs=10):
    
    
    training_loss=[]
    training_acc = []
    validation_loss = []
    validation_acc = []
    
    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0
        model.train()
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            
            loss = loss_fn(output, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            prediction = torch.argmax(output,1)
            train_accuracy +=(prediction == labels).sum().item()
        training_acc.append(train_accuracy/len(train_data))
        training_loss.append(train_loss/len(train_loader))

            
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for images , labels in val_loader:
                output = model(images)
                loss = loss_fn(output , labels)
                val_loss += loss.item()
                prediction  = torch.argmax(output , 1)
                val_accuracy += (prediction == labels).sum().item()
            validation_acc.append(val_accuracy/len(val_data))
            validation_loss.append(val_loss/len(val_loader))

        


        print("Epoch {} , Traning Accuracy {:.2f} , Training Loss {:.2f} ,  Val Accuracy {:.2f} , Val Loss {:.2f}".format(
            epoch +1 , train_accuracy /len(train_data) , train_loss/len(train_loader),val_accuracy/len(val_data),val_loss/len(val_loader)
        ))


# In[54]:


train(model_conv,loss_fn,optimizer)


# In[ ]:




