# Multi-classification Sentiment Analysis with Bert



** Open trail1.py and see every single line of code is commented 

![Predictions](img.png)

## Description about the project
- The multi-language dataset contains 3 sentiments POSITIVE, NEUTRAL and NEGATIVE.



## How to run the Project
- Create virtual environment with python 3.7 or above 
- Install the requirements.txt file
- Use your own data binary classification or multiclass classification
- I trained only 3 epochs, and it has taken more than 2 hours to complete.
- Used bert-base-multilingual-cased

## How to improve accuracy of the project
- Change parameters and play with them like learning rate etc.
- Train more epochs and make sure that you have no computational issues.

## Details about each step of the project.

### About  BertTokenizer and Encoding the Data

- Tokenization is a process to take raw texts and split into tokens, which are numeric data to represent words.
- Constructs a BERT tokenizer. Based on WordPiece.
- Instantiate a pre-trained BERT model configuration to encode our data.
- To convert all the titles from text into encoded form, we use a function called batch_encode_plus , and we will proceed train and validation data separately.
- The 1st parameter inside the above function is the title text.
- add_special_tokens=True means the sequences will be encoded with the special tokens relative to their model.
- When batching sequences together, we set return_attention_mask=True, so it will return the attention mask according to the specific tokenizer defined by the max_length attribute.
- We also want to pad all the titles to certain maximum length.
- We actually do not need to set max_length=256, but just to play it safe.
- return_tensors='pt' to return PyTorch.
- And then we need to split the data into input_ids, attention_masks and labels.
- Finally, after we get encoded data set, we can create training data and validation data.


### BERT Pre-trained Model
We are treating each title as its unique sequence, so one sequence will be classified to one of the three labels.
- bert-base-multilingual-cased is a pre-trained model.
- Using num_labels to indicate the number of output labels.
- We don’t really care about output_attentions.
- We also don’t need output_hidden_states.


### Data Loaders

- DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset.
- We use RandomSampler for training and SequentialSampler for validation.
- Given the limited memory in my environment, I set batch_size=4.


### Optimizer & Scheduler

- To construct an optimizer, we have to give it an iterable containing the parameters to optimize. Then, we can specify optimizer-specific options such as the learning rate, epsilon, etc.
- I found epochs=10 works well for this data set.
- Create a schedule with a learning rate that decreases linearly from the initial learning rate set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial learning rate set in the optimizer.


### Performance Metrics

- We will use f1 score and accuracy per class as performance metrics.



