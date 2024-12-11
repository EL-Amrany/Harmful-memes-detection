import torch

import os
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict
from textwrap import wrap
from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from clip_data_loader import collate_batch_val_test, collate_batch,PostDataset


import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import sklearn
from sklearn.model_selection import train_test_split


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F






def create_data_loader(df, max_len, batch_size):
    

    dataset = PostDataset(
                post_id = df["id"].to_numpy(),
                title = df["caption"].to_numpy(),
                label = df["label"].to_numpy(),
                image = df["img"].to_numpy(),
                max_len = max_len
              )

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch, num_workers=2, pin_memory=True, prefetch_factor=2)


def val_test_create_data_loader(df, max_len, batch_size):
    

    dataset = PostDataset(
                post_id = df["id"].to_numpy(),
                title = df["caption"].to_numpy(),
                label = df["label"].to_numpy(),
                image = df["img"].to_numpy(),
                max_len = max_len
              )
    

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch_val_test, num_workers=2, pin_memory=True, prefetch_factor=2)




class CLIPFakeNewsDetector(nn.Module):
    
    def __init__(self, num_classes):
        super(CLIPFakeNewsDetector, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.fc_title = nn.Linear(in_features=512, out_features=32, bias=True)
        self.fc_image = nn.Linear(in_features=512, out_features=32, bias=True)

        self.d_model = ((32*32)+64)  
        self.nhead = 4       
        self.num_transformer_layers = 1  

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_transformer_layers)

        self.fc_last = nn.Linear(self.d_model, 1)
        self.drop = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,  texts, images):
        if not isinstance(texts, list):
            raise TypeError("Texts should be a list of strings")

        for i, t in enumerate(texts):
            
            if not isinstance(t, str):
                
                raise TypeError(f"Element at index {t} in texts is not a string: {t}")


        
        inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77).to('mps')
        
        outputs = self.clip_model(**inputs)

        text_features = outputs.text_embeds  
        image_features = outputs.image_embeds  
        
        text_features = self.fc_title(text_features)
        text_features = self.relu(text_features)
        text_features = self.drop(text_features)
        
        image_features = self.fc_image(image_features)
        image_features = self.relu(image_features)      
        image_features = self.drop(image_features)
        
        A_expanded = text_features.unsqueeze(2)  
        B_expanded = image_features.unsqueeze(1)  

        result = A_expanded * B_expanded 
   
        combined_features = torch.cat((result.flatten(start_dim=1, end_dim=2),text_features,image_features),axis=1) 
        
        transformed_features = self.transformer_encoder(combined_features.unsqueeze(0))  
        transformed_features = self.drop(transformed_features)
        fusion = self.fc_last(transformed_features.squeeze(0))  
        fusion = self.sigmoid(fusion)
        
        
        return fusion,1


class LRScheduler():
    
    def __init__(self, optimizer, patience=LR_PATIENCE, min_lr=MIN_LR, factor=LR_FACTOR):
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode = "min",
            patience = self.patience,
            factor = self.factor,
            min_lr = self.min_lr,
            verbose = True
        )
        
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping():
    
    def __init__(self, patience=ES_PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}.")
            if self.counter >= self.patience:
                print("INFO: Early Stopping!")
                self.early_stop = True


def get_class_weights(dataframe):

    print(dataframe["label"].value_counts().sort_index())

    label_count = [dataframe["label"].value_counts().sort_index()[0],
                   dataframe["label"].value_counts().sort_index()[1],

                   ]


    class_weights = [1 - (x / sum(label_count)) for x in label_count]
    class_weights = torch.FloatTensor(class_weights)

    return class_weights


class_weights = get_class_weights(df_train)
print(class_weights)


EPOCHS = 1

optimizer = AdamW(fn_detector.parameters(), lr=1e-4, correct_bias=False)

total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_function = torch.nn.BCELoss().to(device)

lr_scheduler = LRScheduler(optimizer)
early_stopping = EarlyStopping()

def train_model(model, data_loader, loss_function, optimizer, device, scheduler, num_examples):
    print("Training model in progress...")
    print("-" * 10)
    
    model = model.train()
    
    train_losses = []
    correct_preds = 0
    
    for data in tqdm(data_loader):


        titles = data["title"]
        images = data["image"].to(device)
        labels = data["label"].to(device)
        

        outputs,_ = model(

                titles, images
        )


        preds = torch.round(outputs)
        
        

        train_loss = loss_function(outputs.to(device), labels.to(torch.float32).reshape(-1,1))  


        correct_preds +=  torch.sum(preds.to(device).reshape(-1,) == labels)
        
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
    correct_preds = correct_preds.to(torch.float32)
    return correct_preds/ num_examples, np.mean(train_losses).astype(np.float32)


def test_model(model, data_loader, loss_function, device, num_examples):
    print("Testing model in progress...")
    print("-" * 10)

    model.eval()
    test_losses = []
    correct_preds = 0
    post_titles = []
    predictions = []
    prediction_probs = []
    real_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            titles = data["title"]
            images = data["image"].to(device)

            labels = data["label"].to(device)


            outputs,_ = model(

                    titles, images
            )
            

            preds = torch.round(outputs)

            test_loss = loss_function(outputs.to(device), labels.to(torch.float32).reshape(-1,1))  # Ensure labels are float for BCEWithLogitsLoss

            correct_preds += torch.sum(preds.to(device).reshape(-1,) == labels)
            test_losses.append(test_loss.item())
            post_titles.extend(titles)
            predictions.extend(preds)
            
            prediction_probs.extend(outputs)
            real_labels.extend(labels)
    
    correct_preds = correct_preds.to(torch.float32)
    test_acc = correct_preds / num_examples
    test_loss = np.mean(test_losses)
    predictions = torch.stack(predictions)
    prediction_probs = torch.stack(prediction_probs)
    real_labels = torch.stack(real_labels)
    
    return test_acc, test_loss, post_titles, predictions, prediction_probs, real_labels



def evaluate_model(model, data_loader, loss_function, device, num_examples):
    print("Validating model in progress...")
    print("-" * 10)
    print(num_examples)

    model = model.eval()
    

    val_losses = []
    correct_preds = 0

    with torch.no_grad():
        i=0
        for data in tqdm(data_loader):

            titles = data["title"]
            images = data["image"].to(device)
            labels = data["label"].to(device)


            outputs,_ = model(

                    titles, images
           )

            preds = torch.round(outputs)

            val_loss = loss_function(outputs.to(device), labels.to(torch.float32).reshape(-1,1))  # Ensure labels are float for BCEWithLogitsLoss


            correct_preds +=  torch.sum(preds.to(device).reshape(-1,) == labels)
            
            
            val_losses.append(val_loss.item())
    print("correct"+str(correct_preds)+" ==> "+str(num_examples))
    correct_preds = correct_preds.to(torch.float32)
    return correct_preds/ num_examples, np.mean(val_losses).astype(np.float32)




def main():

    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    device

    df_train=pd.read_csv('/Users/samir.el-amrany/Desktop/PhD/Hateful memes/harmful_memes.csv')

    df_validate=pd.read_csv('/Users/samir.el-amrany/Desktop/PhD/Hateful memes/my_harmful_memes_vali_cap.csv')
    df_test=pd.read_csv('/Users/samir.el-amrany/Desktop/PhD/Hateful memes/my_harmful_memes_testi_cap.csv')

    

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.255]
        )
    ])


    BATCH_SIZE = 32
    MAX_LEN = 230

    train_data_loader = create_data_loader(df_train, MAX_LEN, BATCH_SIZE)
    validate_data_loader = val_test_create_data_loader(df_validate, MAX_LEN, BATCH_SIZE)
    test_data_loader = val_test_create_data_loader(df_test, MAX_LEN, BATCH_SIZE)

    train_data = next(iter(train_data_loader))
    validate_data = next(iter(validate_data_loader))
    test_data = next(iter(test_data_loader))

    print(test_data["post_id"])
    print()
    print(test_data["title"])
    print()



    print(train_data["label"].shape)

    print()
    print()


    print(validate_data["image"].shape)

    print(validate_data["title"])

    num_classes = 2  
    fn_detector = CLIPFakeNewsDetector(num_classes=num_classes).to('mps')





    titles = train_data["title"]
    images = train_data["image"].to(device)

    labels = train_data["label"].to(device)


    outputs= fn_detector(
            titles, images
    )

    outputs

    print(fn_detector)

    LR_FACTOR=0.5
    MIN_LR=1e-6
    LR_PATIENCE=2

    ES_PATIENCE = 4
    MIN_DELTA = 0

    test_acc, test_loss= evaluate_model(
            fn_detector,
            test_data_loader,
            loss_function,
            device,
            len(df_test)
    )



    print(test_acc)


if __name__ == "__main__":
    main()