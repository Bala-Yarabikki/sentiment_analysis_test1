import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import torch
from transformers import BertForSequenceClassification
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score
import random


df = pd.read_csv("/content/drive/MyDrive/sentiment/data/train.csv") # Load the dataset
df.head()

df.sentiment.value_counts() # Check the distribution of the labels

df = df[df.sentiment.isin(['neutral','positive','negative'])] # Remove rows with missing sentiment labels

df.sentiment.value_counts() #

possible_labels = df.sentiment.unique() # Get the possible labels
label_dict = {}
for index, possible_label in enumerate(possible_labels): # Create a dictionary to map the labels to integers
    label_dict[possible_label] = index #


df.sentiment = df['sentiment'].map(label_dict) # Map the labels to integers
df.head()


"""#### Training and Validation split"""


X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.sentiment.values,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=df.sentiment.values) # Split the data into training and
# validation sets

df['data_type'] = ['not_set']*df.shape[0] # Create a column to store the data type

df.head()

df.loc[X_train, 'data_type'] = 'train' # Set the data type for the training data
df.loc[X_val, 'data_type'] = 'val'

df.groupby(['sentiment', 'data_type']).count() #

"""### Loading Tokenizer and Encoding our Data"""

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased') # Load the tokenizer

encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].content.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt') # Encode the training data

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].content.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt') # Encode the validation data

input_ids_train = encoded_data_train['input_ids'] # Get the input ids for the training data
attention_masks_train = encoded_data_train['attention_mask'] # Get the attention masks for the training data
labels_train = torch.tensor(df[df.data_type=='train'].sentiment.values) # Get the labels for the training data

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].sentiment.values)

dataset_train = TensorDataset(input_ids_train,
                              attention_masks_train,
                              labels_train) # Create the training dataset

dataset_val = TensorDataset(input_ids_val,
                            attention_masks_val,
                            labels_val) # Create the validation dataset

len(dataset_train)


"""###  Setting up BERT Pretrained Model"""

model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels = len(label_dict),
    output_attentions = False,
    output_hidden_states = False) # Load the pretrained model

"""### Creating Data Loaders"""

batch_size = 4

dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size) # Create the training data loader

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=32) # Create the validation data loader


"""### Setting Up Optimizer and Scheduler"""

optimizer = AdamW(
    model.parameters(),
    lr = 1e-5,
    eps = 1e-8) # Create the optimizer

epochs = 3

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps = len(dataloader_train)*epochs) # Create the scheduler


"""### Defining our Performance Metrics"""


def f1_score_func(preds, labels):
    # Get the f1 score
    """

    sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')


def accuracy_per_class(preds, labels):

    # Get the predictions and labels for each class

    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        # Get the indices for the predictions and labels for the current class
        y_preds = preds_flat[labels_flat==label] # Get the predictions for the current class
        y_true = labels_flat[labels_flat==label] # Get the labels for the current class
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')


"""###  Creating our Training Loop"""

seed_val = 17 # Set the seed value
random.seed(seed_val)
np.random.seed(seed_val) # Set the seed for numpy and random
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) # Move the model parameters to the GPU
print(device)


def evaluate(dataloader_val):

    """Turn on evaluation mode which disables dropout.
        This will evaluate our model on the test dataset.

    Keyword arguments:
    dataloader_val
    Return: Set the model to evaluation mode
    """

    model.eval() # Set the model to evaluation mode

    loss_val_total = 0 # Initialize the validation loss
    predictions, true_vals = [], [] # Initialize the predictions and true labels

    for batch in tqdm(dataloader_val):
        # Add batch to GPU

        batch = tuple(b.to(device) for b in batch) # Move batch to GPU

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  } # Create a dictionary of inputs

        with torch.no_grad():
            # Get the output from the model
            outputs = model(**inputs) #
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item() # Add the loss to the validation loss

        logits = logits.detach().cpu().numpy() # Detach the logits from the GPU and move to the CPU
        label_ids = inputs['labels'].cpu().numpy() # Detach the labels from the GPU and move to the CPU
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val) # Calculate the validation loss

    predictions = np.concatenate(predictions, axis=0) # Concatenate the predictions
    true_vals = np.concatenate(true_vals, axis=0) # Concatenate the true labels

    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)): # Loop over the epochs
    # Train the model for one epoch

    model.train()
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train,
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False,
                        disable=False) # Create a progress bar

    for batch in progress_bar:
        # Add batch to GPU
        model.zero_grad()
        batch = tuple(b.to(device) for b in batch) # Move batch to GPU
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }

        outputs = model(**inputs) # Get the output from the model
        loss = outputs[0]
        loss_train_total +=loss.item() # Add the loss to the training loss
        loss.backward()
        # Gradient accumulation step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the gradients
        # Update optimizer and scheduler
        optimizer.step()
        scheduler.step()

        # Update the progress bar and print the loss
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))}) # Set the postfix

    torch.save(model.state_dict(), '/content/drive/MyDrive/sentiment/model/cased.model') # save model

    tqdm.write('\nEpoch {epoch}') # print epoch

    loss_train_avg = loss_train_total/len(dataloader_train) # calculate average loss
    tqdm.write(f'Training loss: {loss_train_avg}') # print average loss

    val_loss, predictions, true_vals = evaluate(dataloader_val) # evaluate model on validation set
    val_f1 = f1_score_func(predictions, true_vals) # calculate f1 score
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')

model.load_state_dict(torch.load('/content/drive/MyDrive/sentiment/model/cased.model', map_location=torch.device('cpu'))) # load model

_,predictions, true_vals = evaluate(dataloader_val) # evaluate model on validation set

accuracy_per_class(predictions, true_vals) # get accuracy per class

