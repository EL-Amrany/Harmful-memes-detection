
import torch
# Importing needed modules for DistilBert model
from transformers import BertTokenizer, RobertaTokenizer
from PIL import Image, ImageFile, UnidentifiedImageError
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
Image.MAX_IMAGE_PIXELS = 933120000

ImageFile.LOAD_TRUNCATED_IMAGES = True


train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.255]
    #)
])

# Transform function for image processing (validation and testing)
# No data augmentation in validation and test data splits in order to
# define constant validation and testing process
val_test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.255]
    #)
])

# Loading DistilBert tokenizers adjusted for lower case English text corpus
# for tokenization of title input sequence
#title_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#title_tokenizer =RobertaTokenizer.from_pretrained("roberta-base")
#from transformers import GPT2Tokenizer
# Load GPT-2 tokenizer with padding token '[PAD]'
#title_tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token="[PAD]")

#from transformers import RobertaTokenizer

#title_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class PostDataset(Dataset):
    
    # Constructor initialized with relevant attributes plus tokenizer information
    def __init__(self, post_id, title,label, image ,max_len):
        self.post_id = post_id
        self.title = title

        self.label = label
        self.max_length = max_len
        self.img = image
        
    # Returns length of the dataset for internal looping 
    def __len__(self):
        return len(self.label)
    
    # Internal function to fetch next sample within dataset object
    def __getitem__(self, idx):
        # Iteration function to retireve next sample
        post_id = self.post_id[idx]
        text = self.title[idx]
        label = self.label[idx]
        image_path=self.img[idx]

        # Saving id, clean_title and label entries per post
        # in sample dictionary
        sample = {
            "post_id": post_id,
            "clean_title": text,
            "image_path":image_path,
            "label": label
        }
        
        # Return sample dictionary containing all needed attributes
        return sample
    




def collate_batch(batch):
    global nf, ose, uni
    nf = 0
    ose = 0
    uni = 0
    
    # List to save processed batch samples
    batch_processed = []
    
    # Iteration over input batch of size 16
    for i in range(len(batch)):
        
        # Saving attributes in local variables
        post_id = batch[i]["post_id"]
        text = batch[i]["clean_title"]
        label = batch[i]["label"]
        image_path = batch[i]["image_path"]

        
        # Leveraging DistilBertTokenizer to generate
        # encoding of input text sequence
#######################################################
        # encoding = title_tokenizer.encode_plus(
        #     text,
        #     max_length=23,
        #     padding="max_length",
        #     truncation=True,
        #     add_special_tokens=True,
        #     return_token_type_ids=False,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
#######################################################

        # Try-Except-Else clause to process image data
        # Fetch images from image_set folder via post_id, transform and reshape tensor
        try:
            image = Image.open(f"/Users/samir.el-amrany/Desktop/Hateful memes detection Datasets/hateful_memes-2/{image_path}")
        # Handling FileNotFoundError and randomly initializing pixels
        except FileNotFoundError:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
            print('fn')
            #continue

        # Handling UnidentifiedImageError and randomly initializing pixels
        except UnidentifiedImageError:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
            #print('uni')
            #continue
        # Handling OSError and randomly initializing pixels
        except OSError as e:
            if "image file is truncated" in str(e):
                print(f"Truncated image: {post_id}. Skipping...")
                image = torch.rand(3, 224, 224)
                image = torch.unsqueeze(image, 0)
                #continue
            else:
                print(f"Error processing image file: {e}")
                image = torch.rand(3, 224, 224)
                image = torch.unsqueeze(image, 0)
                #continue

        # Else: Convert image to RGB, process with train_transform
        # and reshape to tensor of shape = [1, 3, 224, 224] for
        # [sample_count, color_channels, height in pixel, width in pixel]
        else:
            image = image.convert("RGB")
            
            image = train_transform(image)
            #print(image.shape)
            image = torch.unsqueeze(image, 0)

        #print(image.shape)
        
        # Storing processed attributes of sample in sample
        # dictionary: post_id, title (text), input_ids,
        # attention_mask, image and label
        #print("fnf="+str(nf))
        #print("uni="+str(uni))
        #print("ose="+str(ose))
        sample = {
            "post_id": post_id,
            "text": text,
            "image": image.flatten(),
            "label": torch.tensor(label, dtype=torch.long)
        }
        
        # Append current samples dictionary to processed
        # batch list --> List of sample dictionaries
        batch_processed.append(sample)
        
    # Complex operation in order to unpack list of dictionaries and
    # merge dictionary entries into correct PyTorch tensor for forward processing
    postId = []
    title = []
    
    # For-loop to stack sample dictionary keys into appropriate format
    for i in range(len(batch_processed)):
        # If first sample of batch, initialize attribute tensors and reshape
        if i == 0:
            postId.append(batch_processed[i]["post_id"])
            title.append(batch_processed[i]["text"])
            image_tensor = batch_processed[i]["image"].reshape(-1, 3, 224, 224)

            label_tensor = batch_processed[i]["label"].reshape(-1,)
            continue
        # Stack attributes of sample dictionary keys to generate correct tensor shape
        postId.append(batch_processed[i]["post_id"])
        title.append(batch_processed[i]["text"])
        image_tensor = torch.cat((image_tensor, batch_processed[i]["image"].reshape(-1, 3, 224, 224)))
        label_tensor = torch.cat((label_tensor, batch_processed[i]["label"].reshape(-1,)))

    
    # Returning batch list of sample dictionaries containing 16 processed samples
    return {
        "post_id": postId,
        "title": title,
        "image": image_tensor,
        "label": label_tensor
    }

# Transform function for image processing (training)
# Performing data augmentation by random resizing, cropping
# and flipping images in order to artificially create new
# image data per training epoch


def collate_batch_val_test(batch):
    
    # List to save processed batch samples
    batch_processed = []
    
    # Iteration over input batch of size 16
    for i in range(len(batch)):
        
        # Saving attributes in local variables
        post_id = batch[i]["post_id"]
        text = batch[i]["clean_title"]

        label = batch[i]["label"]
        image_path = batch[i]["image_path"]
   
        # Leveraging DistilBertTokenizer to generate


        # Try-Except-Else clause to process image data
        # Fetch images from image_set folder via post_id, transform and reshape tensor
        try:
            image = Image.open(f"/Users/samir.el-amrany/Desktop/Hateful memes detection Datasets/hateful_memes-2/{image_path}")
        # Handling FileNotFoundError and randomly initializing pixels
        except FileNotFoundError:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
        # Handling UnidentifiedImageError and randomly initializing pixels
        except UnidentifiedImageError:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
        # Handling OSError and randomly initializing pixels
        except OSError:
            image = torch.rand(3, 224, 224)
            image = torch.unsqueeze(image, 0)
        # Else: Convert image to RGB, process with train_transform
        # and reshape to tensor of shape = [1, 3, 224, 224] for
        # [sample_count, color_channels, height in pixel, width in pixel]
        else:
            image = image.convert("RGB")
            image = train_transform(image)
            image = torch.unsqueeze(image, 0)
        
        # Storing processed attributes of sample in sample
        # dictionary: post_id, title (text), input_ids,
        # attention_mask, image and label
        sample = {
            "post_id": post_id,
            "text": text,
            "image": image.flatten(),

            "label": torch.tensor(label, dtype=torch.long)
        }
        
        # Append current samples dictionary to processed
        # batch list --> List of sample dictionaries
        batch_processed.append(sample)
        
    # Complex operation in order to unpack list of dictionaries and
    # merge dictionary entries into correct PyTorch tensor for forward processing
    postId = []
    title = []
    
    # For-loop to stack sample dictionary keys into appropriate format
    for i in range(len(batch_processed)):
        # If first sample of batch, initialize attribute tensors and reshape
        if i == 0:
            postId.append(batch_processed[i]["post_id"])
            title.append(batch_processed[i]["text"])
            image_tensor = batch_processed[i]["image"].reshape(-1, 3, 224, 224)

            label_tensor = batch_processed[i]["label"].reshape(-1,)

            continue

        # Stack attributes of sample dictionary keys to generate correct tensor shape
        postId.append(batch_processed[i]["post_id"])
        title.append(batch_processed[i]["text"])
        image_tensor = torch.cat((image_tensor, batch_processed[i]["image"].reshape(-1, 3, 224, 224)))
        label_tensor = torch.cat((label_tensor, batch_processed[i]["label"].reshape(-1,)))
    
    # Returning batch list of sample dictionaries containing 16 processed samples
    return {
        "post_id": postId,
        "title": title,
        "image": image_tensor,
        "label": label_tensor
    }
