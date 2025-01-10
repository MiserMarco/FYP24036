import math
import random
import io
import time
import datetime
import pandas as pd
import numpy as np
import os
import chardet
import torch
from transformers import *
import torch.nn as nn
import torch.nn.functional as F
import accelerate
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model_name = "pborchert/BusinessBERT"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#initializing hyperparameters
max_seq_length = 256 # !!!!!! check this, whats the longest sentences
batch_size = 64

num_hidden_layers_generator = 1;
num_hidden_layers_discriminator = 1;
noise_size = 100
out_dropout_rate = 0.2

multi_gpu = True
apply_scheduler = False
num_train_epochs = 10
warmup_proportion = 0.1
learning_rate_generator = 5e-5
learning_rate_discriminator = 5e-5
print_steps = 10 # print for every 10 steps
epsilon = 1e-8

label_list = ['T', 'F']


#read csv file with its encoding format
def readcsv(file):
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())
        print(result, file)
        

    df = pd.read_csv(file, encoding = result['encoding'])
    return df

#return a dictionary containing the sentences with there labels
#{'sentences':[], 'label':[]}
#!!make sure the format is Risky Sentences, T/F
def get_label_and_sentences(file):

    df = readcsv(file)
    sentences = df['Risky Sentences'].tolist()
    label = df['T/F'].tolist()
    return {'sentences':sentences, 'label':label}

#retrieve from a given directory, get labelled or unlabelled data, shuffling data in the end return the raw datasets, in this case the directory is item1, item1a, item7......
def retrieve_sentences_from_directory(directory):
    corpus_labelled = []
    corpus_unlabelled = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)

            df = get_label_and_sentences(file_path)
            sentences = df['sentences']
            labels = df['label']
            label_list_temp = ['T', 'F']
            for i in range(len(df['sentences'])):
                if labels[i] not in label_list_temp:
                    corpus_unlabelled.append({'sentence':sentences[i], 'label':0})
                
                else:
                    label = 1 if labels[i] == 'T' else 0
                    corpus_labelled.append({'sentence':sentences[i], 'label':label})

    return {'labelled':corpus_labelled, 'unlabelled':corpus_unlabelled}

#to combine the labelled and unlabelled dataset together {dataset, labelled_rate}
def combine_dataset(dataset):
    labelled_dataset = dataset['labelled']
    unlabelled_dataset = dataset['unlabelled']
    combine_dataset = labelled_dataset + unlabelled_dataset
    labelled_rate = len(labelled_dataset) / (len(labelled_dataset) + len(unlabelled_dataset))
    return {'dataset':combine_dataset, 'labelled_rate':labelled_rate}
                    
#generate dataloader for the model
#train_dataloader = generate_data_loader(train_examples, train_label_masks, label_map, do_shuffle = True, balance_label_examples = apply_balance)
#The labeled (train) dataset is assigned with a mask set to True
#The unlabeled (train) dataset is assigned with a mask set to False
#the dataset is the result of retrieve_sentences_from_directory(directory)
def generate_data_loader(dataset, label_mask, do_shuffle = False, balance_label_examples = False):
    
    examples = [] #to store the mixed datasets
    combined_dataset = combine_dataset(dataset)
    large_dataset = combined_dataset['dataset']
    labelled_rate = combined_dataset['labelled_rate']
    labelled_dataset = dataset['labelled']
    unlabelled_dataset = dataset['unlabelled']
    
    for i in range(len(large_dataset)):
        examples.append(((large_dataset[i]['sentence'], large_dataset[i]['label']), label_mask[i]))
        
    if balance_label_examples: #make sure to check the rate condition
        #followed the instructions on paper
        balance = int(1/labelled_rate)
        balance = int(math.log(balance,2))
        
        if balance < 1:
            balance = 1
        
        for i in range(0, int(balance)):
            for j in range(len(labelled_dataset)):
                examples.append(((labelled_dataset[j]['sentence'],labelled_dataset[j]['label']), label_mask[j]))
        
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []
        
    for (text, lab_mask) in examples:
        encoded_sentences = tokenizer.encode(text[0], add_special_tokens=True, max_length=max_seq_length, padding="max_length", truncation=True)
        input_ids.append(encoded_sentences)
            
        label_id_array.append(text[1])
        label_mask_array.append(lab_mask)
          
    #ignoring the padded input wordpieces
    for sentence in input_ids:
        att_mask = [int(id > 2) for id in sentence]
        input_mask_array.append(att_mask)
        
    input_ids = torch.tensor(input_ids)
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)
    
    final_dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)
    
    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler
    
        
    return DataLoader(final_dataset, sampler = sampler(final_dataset), batch_size = batch_size)
 
directory = "/Users/songdelin/Desktop/gan_mask_bert/trial" #change it to the directory of labelled / unlabelled data
raw_dataset = retrieve_sentences_from_directory(directory)
labelled = raw_dataset["labelled"]
unlabelled = raw_dataset["unlabelled"]
train_label_masks = np.ones(len(labelled), dtype=bool)
tmp_masks = np.zeros(len(unlabelled), dtype=bool)
train_label_masks = np.concatenate([train_label_masks, tmp_masks])

train_dataloader = generate_data_loader(raw_dataset, train_label_masks, do_shuffle = True, balance_label_examples = True)



#The Generator
class Generator(nn.Module):
    def __init__(self, noise_size=100, output_size=512, hidden_sizes=[512], dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1],output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep


#The Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size=512, hidden_sizes=[512], num_labels=2, dropout_rate=0.1):
        super(Discriminator, self).__init__()
        self.input_dropout = nn.Dropout(p=dropout_rate)
        layers = []
        hidden_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])

        self.layers = nn.Sequential(*layers) #per il flatten
        self.logit = nn.Linear(hidden_sizes[-1],num_labels+1) # +1 for the probability of this sample being fake/real.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_rep):
        input_rep = self.input_dropout(input_rep)
        last_rep = self.layers(input_rep)
        logits = self.logit(last_rep)
        probs = self.softmax(logits)
        return last_rep, logits, probs

#define the configuration
config = AutoConfig.from_pretrained(model_name)
hidden_size = int(config.hidden_size)
hidden_levels_generator = [hidden_size for i in range(0, num_hidden_layers_generator)]
hidden_levels_discriminator = [hidden_size for i in range(0, num_hidden_layers_discriminator)]
print("hidden_size of businessbert: ", hidden_size)

#instantiate the generator and discriminator
generator = Generator(noise_size=noise_size, output_size = hidden_size, hidden_sizes = hidden_levels_generator, dropout_rate=out_dropout_rate)

discriminator = Discriminator(input_size=hidden_size, hidden_sizes=hidden_levels_discriminator, num_labels=2, dropout_rate=out_dropout_rate)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Put everything in the GPU if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    model.cuda()
    if multi_gpu:
        model = torch.nn.DataParallel(model)

#the variables of different models
total_t0 = time.time()
transformer_vars = [i for i in model.parameters()]
discriminator_vars = transformer_vars + [var for var in discriminator.parameters()]
generator_vars = [var for var in generator.parameters()]

#optimizer
discriminator_optimizer = torch.optim.AdamW(discriminator_vars, lr = learning_rate_discriminator)
generator_optimizer = torch.optim.AdamW(generator_vars, lr = learning_rate_generator)

#scheduler
if apply_scheduler:
    num_train_examples = len(labelled) + len(unlabelled)
    num_train_steps = int(num_train_examples / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    
    scheduler_discriminator = get_constant_schedule_with_warmup(discriminator_optimizer, num_warmup_steps = num_warmup_steps)
    
    scheduler_generator = get_constant_schedule_with_warmup(generator_optimizer, num_warmup_steps = num_warmup_steps)

print("finished")
    
#beginning of training

for epoch in range(0, num_train_epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, num_train_epochs))
    print('Training...')
    
    t0 = time.time()
    tr_generator_loss = 0
    tr_discriminator_loss = 0
    
    model.train()
    generator.train()
    discriminator.train()
    
    for step, batch in enumerate(train_dataloader):
        if step % print_steps == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            
            print(' Batch {:>5,} of {:>5}.  Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
        
        batch_input_ids = batch[0].to(device)
        
        #print("batch_input_ids")
        #print(batch_input_ids)
        batch_input_mask = batch[1].to(device)
        #print("batch_input_mask")
        #print(batch_input_mask)
        batch_labels = batch[2].to(device)
        #print("batch_labels")
        #print(batch_labels)
        batch_label_mask = batch[3].to(device)
        #print("batch_label_mask")
        #print(batch_label_mask)
        
        
        
        real_batch_size = batch_input_ids.shape[0]
        
        #Encoding the batched data in the transformer
        model_outputs = model(batch_input_ids, attention_mask = batch_input_mask)
        hidden_states = model_outputs[-1]
        
        #uniform distribution to generate noise
        noise = torch.zeros(real_batch_size, noise_size, device = device).uniform_(0,1)
        generated_fake = generator(noise) #generate the fake data
        
        discriminator_input = torch.cat([hidden_states, generated_fake], dim = 0)
        features, logits, probs = discriminator(discriminator_input)
        #features: last layer of the discriminator
        #logits: the logits of the last layer
        #probs: softmax probability of the last layer
        
        #Separating the fake and real data
        features_list = torch.split(features, real_batch_size)
        discriminator_real_features = features_list[0]
        discriminator_fake_features = features_list[1]
        
        logits_list = torch.split(logits, real_batch_size)
        discriminator_real_logits = logits_list[0]
        discriminator_fake_logits = logits_list[1]
        
        probs_list = torch.split(logits, real_batch_size)
        discriminator_real_probs = probs_list[0]
        discriminator_fake_probs = probs_list[1]
        
        #Loss Evaluation
        #generator loss
        unsup = -1 * torch.mean(torch.log(1 - discriminator_fake_probs[: -1] + epsilon))
        feature_matching = torch.mean(torch.pow(torch.mean(discriminator_real_features, dim = 0) - torch.mean(discriminator_fake_features, dim = 0), 2)) #should be similar to the real one
        
        generator_loss = unsup + feature_matching
        
        
        #discriminator loss (unlabelled data is ignored)
        logits = discriminator_real_logits[:, 0:-1]
        log_probs = F.log_softmax(logits, dim = -1)
        label_to_onehot = torch.nn.functional.one_hot(batch_labels, len(label_list))
        single_loss = -torch.sum(label_to_onehot * log_probs, dim=-1)
        #print("shape of single loss:", single_loss.shape)
        #print("shape of batch label mask:", batch_label_mask.shape)
        single_loss = torch.masked_select(single_loss, batch_label_mask.to(device))
        labelled_num = single_loss.type(torch.float32).numel()
        
        if labelled_num == 0:
            d_sup = 0
        else:
            d_sup = torch.div(torch.sum(single_loss.to(device)), labelled_num)
        
        d_unsup1 = -1 * torch.mean(torch.log(1 - discriminator_real_probs[:, -1] + epsilon))
        d_unsup2 = -1 * torch.mean(torch.log(discriminator_fake_probs[:, -1] + epsilon))
        
        discriminator_loss = d_sup + d_unsup1 + d_unsup2
        
        #optimization
        generator_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        
        #keeps the computational graph in memory, allowing for additional backward passes without having to recreate the graph.
        generator_loss.backward(retain_graph=True)
        discriminator_loss.backward()
        
        generator_optimizer.step()
        discriminator_optimizer.step()
        
        tr_generator_loss += generator_loss.item()
        tr_discriminator_loss += discriminator_loss.item()
        
        if apply_scheduler:
            scheduler_discriminator.step()
            scheduler_generator.step()
    
    avg_train_loss_generator = tr_generator_loss / len(train_dataloader)
    avg_train_loss_discriminator = tr_discriminator_loss / len(train_dataloader)
    
    training_time = format_time(time.time() - t0)
    
    print("-" * 20)
    print("Average training loss of generator : {0:.3f}".format(avg_train_loss_generator))
    print("Average training loss of discriminator: {0:.3f}".format(avg_train_loss_discriminator))
    print("Training epoc took: {:}".format(training_time))
    

print("\nTraining complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
        
        
        
    


    
    








