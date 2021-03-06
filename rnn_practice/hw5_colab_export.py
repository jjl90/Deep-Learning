# -*- coding: utf-8 -*-
"""Copy of CS1699: Homework5 colab Release

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BYVie1jxpKuVApMkjuIh72YAupIF6751
"""

import collections
import copy
import csv
import os
from io import StringIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import categorical
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm as tqdm
from google.colab import files
import matplotlib.pyplot as plt

"""# Before you started
You need to save a copy in your own Google Drive then you could edit on this colab.

Google offers free GPU in the colab environments, but you may need to configure the environment. 

You can turn on the GPU mode in `Edit -> Notebook Settings` and change the `Runtime type` to be `Python3` and `Hardware accelerator` to be `GPU`.
"""

print("GPU Model: %s" % torch.cuda.get_device_name(0))
print("You should get either a Tesla P100 or Tesla T4 GPU.")
print("Tesla P100 is probably 3x faster than T4 but both should work.")

PADDING_TOKEN = 0

"""# RNN modules"""

class GRUCell(nn.Module):
  """Implementation of GRU cell from https://arxiv.org/pdf/1406.1078.pdf."""

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Learnable weights and bias for `update gate`
    self.W_z = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_z = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_z', None)

    # Learnable weights and bias for `reset gate`
    self.W_r = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b_r = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b_r', None)

    # Learnable weights and bias for `output gate`
    self.W = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.b = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('b', None)

    self.reset_parameters()
    self.count_parameters()

  def forward(self, x, prev_state):
  
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    z = torch.sigmoid(F.linear(concat_hx, self.W_z, self.b_z))
    r = torch.sigmoid(F.linear(concat_hx, self.W_r, self.b_r))
    h_tilde = torch.tanh(
        F.linear(torch.cat((r * prev_h, x), dim=1), self.W, self.b))
    next_h = (1 - z) * prev_h + z * h_tilde
    return next_h

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return

class LSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    # Learnable weights and bias for `forget gate`
    self.W_fg = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_fg_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_fg_bi', None)
    # Learnable weights and bias for `input gate`
    self.W_ig= nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ig_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ig_bi', None)    
    # Learnable weights and bias for `other gate`
    self.W_og = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_og_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_og_bi', None) 
    # Learnable weights and bias for `output layer`
    self.W_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ol_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ol_bi', None) 

    self.reset_parameters()
    self.count_parameters()

    return

  def forward(self, x, prev_state, c_state=None):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state
    if c_state is None:
      batch = x.shape[0]
      c_st = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      c_st = c_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    ft = torch.sigmoid(F.linear(concat_hx, self.W_fg, self.W_fg_bi))
    it = torch.sigmoid(F.linear(concat_hx, self.W_ig ,self.W_ig_bi))
    ct_t = torch.sigmoid(F.linear(concat_hx, self.W_og ,self.W_og_bi))
    ct = ft * c_st + it * ct_t
    ot = torch.sigmoid(F.linear(concat_hx, self.W_ol, self.W_ol_bi))
    ht = ot * torch.tanh(ct)
    return ht

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return

class PeepholedLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

   # Learnable weights and bias for `forget gate`
    self.W_fg = nn.Parameter(torch.Tensor(hidden_size, hidden_size*2 + input_size))
    if bias:
      self.W_fg_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_fg_bi', None)
    # Learnable weights and bias for `input gate`
    self.W_ig= nn.Parameter(torch.Tensor(hidden_size, hidden_size*2 + input_size))
    if bias:
      self.W_ig_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ig_bi', None)    
    # Learnable weights and bias for `other gate`
    self.W_og = nn.Parameter(torch.Tensor(hidden_size, hidden_size*2 + input_size))
    if bias:
      self.W_og_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_og_bi', None) 
    # Learnable weights and bias for `output layer`
    self.W_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
    if bias:
      self.W_ol_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ol_bi', None) 

    self.reset_parameters()
    self.count_parameters()
    return

  def forward(self, x, prev_state, c_state=None):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state
    if c_state is None:
      batch = x.shape[0]
      c_st = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      c_st = c_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    concat_hxc = torch.cat((concat_hx, c_st), dim=1)
    ft = torch.sigmoid(F.linear(concat_hxc, self.W_fg, self.W_fg_bi))
    it = torch.sigmoid(F.linear(concat_hxc, self.W_ig ,self.W_ig_bi))
    ct_t = torch.sigmoid(F.linear(concat_hxc, self.W_og ,self.W_og_bi))
    ct = ft * prev_h + it * ct_t
    ot = torch.sigmoid(F.linear(ct, self.W_ol, self.W_ol_bi))
    ht = ot * torch.tanh(ct)
    return ht

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return

class CoupledLSTMCell(nn.Module):

  def __init__(self, input_size, hidden_size, bias=False):
    super().__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

   # Learnable weights and bias for `forget gate`
    self.W_fg = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_fg_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_fg_bi', None)
    # Learnable weights and bias for `input gate`
    self.W_ig= nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ig_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ig_bi', None)    
    # Learnable weights and bias for `other gate`
    self.W_og = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_og_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_og_bi', None) 
    # Learnable weights and bias for `output layer`
    self.W_ol = nn.Parameter(torch.Tensor(hidden_size, hidden_size + input_size))
    if bias:
      self.W_ol_bi = nn.Parameter(torch.Tensor(hidden_size))
    else:
      self.register_parameter('W_ol_bi', None) 

    self.reset_parameters()
    self.count_parameters()

    return

  def forward(self, x, prev_state, c_state=None):
    if prev_state is None:
      batch = x.shape[0]
      prev_h = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      prev_h = prev_state
    if c_state is None:
      batch = x.shape[0]
      c_st = torch.zeros((batch, self.hidden_size), device=x.device)
    else:
      c_st = c_state

    concat_hx = torch.cat((prev_h, x), dim=1)
    ft = torch.sigmoid(F.linear(concat_hx, self.W_fg, self.W_fg_bi))
    it = torch.sigmoid(F.linear(concat_hx, self.W_ig ,self.W_ig_bi))
    ct_t = torch.sigmoid(F.linear(concat_hx, self.W_og ,self.W_og_bi))
    ct = ft * c_st + (1 - ft) * ct_t
    ot = torch.sigmoid(F.linear(concat_hx, self.W_ol, self.W_ol_bi))
    ht = ot * torch.tanh(ct)
    return ht

  def reset_parameters(self):
    sqrt_k = (1. / self.hidden_size)**0.5
    with torch.no_grad():
      for param in self.parameters():
        param.uniform_(-sqrt_k, sqrt_k)
    return

  def extra_repr(self):
    return 'input_size={}, hidden_size={}, bias={}'.format(
        self.input_size, self.hidden_size, self.bias is not True)

  def count_parameters(self):
    print('Total Parameters: %d' %
          sum(p.numel() for p in self.parameters() if p.requires_grad))
    return

RNN_MODULES = {
  'gru': GRUCell,
  'lstm': LSTMCell,
  'peepholed_lstm': PeepholedLSTMCell,
  'coupled_lstm': CoupledLSTMCell,
}

"""# Upload data
Please use the following code snippet to upload

* imdb_train.csv
* imdb_test.csv
* shakespeare.txt

You can choose multiple files to upload all at once.
"""

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

train_dataset_text = uploaded['imdb_train.csv']
test_dataset_text = uploaded['imdb_test.csv']
shakespeare_text = uploaded['shakespeare.txt']

"""# Part I: Sentiment analysis"""

### Hyperparameters for training (previously defined in FLAGS)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
BATCH_SIZE = 4096
EPOCHS = 100
GRADIENT_CLIP_NORM = 1.0

### Hyperparameters for sentence analysis model
EMBEDDING_DIM = 128
HIDDEN_SIZE = 100
REVIEW_MAX_LENGTH = 200
VOCABULARY_MIN_COUNT = 100
VOCABULARY_MAX_SIZE = 20000
RNN_MODULE = 'coupled_lstm'    # You need to try 'lstm', 'peepholed_lstm', 'coupled_lstm'

class IMDBReviewDataset(Dataset):

  def __init__(self,
               csv_text,
               vocabulary=None,
               vocab_min_count=10,
               vocab_max_size=None,
               review_max_length=200):
    self.csv_text = csv_text
    self.vocab_min_count = vocab_min_count
    self.vocab_max_size = vocab_max_size
    self.review_max_length = review_max_length - 2

    self.data = []

    encoded_text = csv_text.strip().decode(encoding='utf-8')
    fp = StringIO(encoded_text)
    reader = csv.DictReader(fp, delimiter=',')
    for row in tqdm(reader):
      self.data.append((row['review'].split(' ')[:review_max_length],
                        int(row['sentiment'] == 'positive')))
    fp.close()

    if vocabulary is not None:
      print('Using external vocabulary - vocab-related configs ignored.')
      self.vocabulary = vocabulary
    else:
      self.vocabulary = self._build_vocabulary()

    self.word2index = {w: i for (i, w) in enumerate(self.vocabulary)}
    self.index2word = {i: w for (i, w) in enumerate(self.vocabulary)}
    self.oov_token_id = self.word2index['OOV_TOKEN']
    self.pad_token_id = self.word2index['PAD_TOKEN']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    review, label = self.data[index]
    review = ['BEGIN_TOKEN'] + review + ['END_TOKEN']
    token_ids = [self.word2index.get(w, self.oov_token_id) for w in review]
    return token_ids, label

  def _build_vocabulary(self):
    special_tokens = ['PAD_TOKEN', 'BEGIN_TOKEN', 'OOV_TOKEN', 'END_TOKEN']

    counter = collections.Counter()
    for review, _ in self.data:
      counter.update(review)

    vocab = counter.most_common(self.vocab_max_size - 4)
    if self.vocab_min_count is not None:
      vocab_tokens = [w for (w, c) in vocab if c >= self.vocab_min_count]
    else:
      vocab_tokens, _ = zip(vocab)

    return special_tokens + vocab_tokens

  def get_vocabulary(self):
    return self.vocabulary

  def print_statistics(self):
    reviews, labels = zip(*self.data)
    lengths = [len(x) for x in reviews]
    positive = np.sum(labels)
    negative = len(labels) - positive
    print('Total instances: %d, positive: %d, negative: %d' %
          (len(self.data), positive, negative))
    print('Review lengths: max: %d, min: %d, mean: %d, median: %d' %
          (max(lengths), min(lengths), np.mean(lengths), np.median(lengths)))
    print('Vocabulary size: %d' % len(self.vocabulary))
    return


def imdb_collate_fn(batch_data, padding_token_id=PADDING_TOKEN):
  """Padding variable-length sequences."""
  batch_tokens, batch_labels = zip(*batch_data)
  lengths = [len(x) for x in batch_tokens]
  max_length = max(lengths)

  padded_tokens = []
  for tokens, length in zip(batch_tokens, lengths):
    padded_tokens.append(tokens + [padding_token_id] * (max_length - length))

  padded_tokens = torch.tensor(padded_tokens, dtype=torch.int64)
  lengths = torch.tensor(lengths, dtype=torch.int64)
  labels = torch.tensor(batch_labels, dtype=torch.int64)

  return padded_tokens, lengths, labels

class SentimentClassification(nn.Module):

  def __init__(self,
               vocabulary_size,
               embedding_dim,
               rnn_module,
               hidden_size,
               bias=False):
    super().__init__()
    self.vocabulary_size = vocabulary_size
    self.rnn_module = rnn_module
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.bias = bias

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = self.rnn_module(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     bias=bias)
    self.classifier = nn.Linear(hidden_size, 2)
    return

  def forward(self, batch_reviews, batch_lengths):
    data = self.embedding(batch_reviews)

    state = None
    batch_size, total_steps, _ = data.shape
    full_outputs = []
    for step in range(total_steps):
      next_state = self.rnn_model(data[:, step, :], state)
      if isinstance(next_state, tuple):
        h, c = next_state
        full_outputs.append(h)
      else:
        full_outputs.append(next_state)
      state = next_state

    full_outputs = torch.stack(full_outputs, dim=1)
    outputs = full_outputs[torch.arange(batch_size), batch_lengths - 1, :]
    logits = self.classifier(outputs)
    return logits

def imdb_trainer(batch_size, epochs):
  train_dataset = IMDBReviewDataset(csv_text=train_dataset_text,
                                    vocab_min_count=VOCABULARY_MIN_COUNT,
                                    vocab_max_size=VOCABULARY_MAX_SIZE,
                                    review_max_length=REVIEW_MAX_LENGTH)
  train_dataset.print_statistics()
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            collate_fn=imdb_collate_fn)
  vocabulary = train_dataset.get_vocabulary()

  # Validation dataset should use the same vocabulary as the training set.
  val_dataset = IMDBReviewDataset(csv_text=test_dataset_text,
                                  vocabulary=vocabulary,
                                  review_max_length=REVIEW_MAX_LENGTH)
  val_dataset.print_statistics()
  val_loader = DataLoader(val_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=8,
                          collate_fn=imdb_collate_fn)

  best_model = None
  best_acc = 0.0

  full_train_loss = []
  full_train_accuracy = []
  full_val_loss = []
  full_val_accuracy = []

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = SentimentClassification(vocabulary_size=len(vocabulary),
                                  embedding_dim=EMBEDDING_DIM,
                                  rnn_module=RNN_MODULES[RNN_MODULE],
                                  hidden_size=HIDDEN_SIZE)
  model.to(device)

  print('Model Architecture:\n%s' % model)

  criterion = nn.CrossEntropyLoss(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(),
                               lr=LEARNING_RATE,
                               weight_decay=WEIGHT_DECAY)
  for epoch in range(epochs):
    for phase in ('train', 'eval'):
      if phase == 'train':
        model.train()
        dataset = train_dataset
        data_loader = train_loader
      else:
        model.eval()
        dataset = val_dataset
        data_loader = val_loader

      running_loss = 0.0
      running_corrects = 0

      for step, (reviews, lengths, labels) in tqdm(enumerate(data_loader)):
        reviews = reviews.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(reviews, lengths)
          _, preds = torch.max(outputs, 1)
          loss = criterion(outputs, labels)

          if phase == 'train':
            loss.backward()

            # RNN model is easily getting exploded gradients, thus we perform
            # gradients clipping to mitigate this issue.
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

        running_loss += loss.item() * reviews.size(0)
        running_corrects += torch.sum(preds == labels.data)

      epoch_loss = running_loss / len(dataset)
      epoch_acc = running_corrects.double() / len(dataset)
      if phase == 'train': 
        full_train_accuracy.append(epoch_acc)
        full_train_loss.append(epoch_loss)
      elif phase == 'eval':
        full_val_accuracy.append(epoch_acc)
        full_val_loss.append(epoch_loss)
      
      print('[Epoch %d] %s accuracy: %.4f, loss: %.4f' %
            (epoch + 1, phase, epoch_acc, epoch_loss))

      if phase == 'eval':
        if epoch_acc > best_acc:
          best_acc = epoch_acc
          best_model = copy.deepcopy(model.state_dict())
      
  state_dict = {"model": best_model, "vocabulary": vocabulary}
  print("Best validation accuracy: %.4f" % best_acc)
  logs = (full_train_loss, full_train_accuracy, full_val_loss, full_val_accuracy)
  

  return state_dict, logs

state_dict, logs = imdb_trainer(BATCH_SIZE, EPOCHS)

### You can make a plot using matplotlib with logs

##logs = (full_train_loss, full_train_accuracy, full_val_loss, full_val_accuracy)


epoch_arr = np.arange(0,100)
#training loss
plt.plot(epoch_arr, logs[0])
plt.ylabel('training loss')
plt.xlabel('epochs')
plt.show()
#training accuracy
plt.plot(epoch_arr, logs[1])
plt.ylabel('training accuracy')
plt.xlabel('epochs')
plt.show()
#validation loss
plt.plot(epoch_arr, logs[2])
plt.ylabel('validation loss')
plt.xlabel('epochs')
plt.show()
#accuracy loos
plt.plot(epoch_arr, logs[3])
plt.ylabel('accuracy loss')
plt.xlabel('epochs')
plt.show()

"""# Part II: Language model and sentence generation"""

### Hyperparameters for training (previously defined in FLAGS)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
BATCH_SIZE = 4096
EPOCHS = 10

### Hyperparameters for sentence analysis model
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
RNN_MODULE = 'gru'
HISTORY_LENGTH = 100

### Hyperparameters for generating new sentence
GENERATION_LENGTH = 2000
START_STRING = 'ROMEO'
TEMPERATURE = 1.0

class ShakespeareDataset(Dataset):

  def __init__(self, encoded_text, history_length):
    self.encoded_text = encoded_text
    self.history_length = history_length

    raw_text = self.encoded_text.strip().decode(encoding='utf-8')

    self.vocab = sorted(set(raw_text))
    self.char2index = {x: i for (i, x) in enumerate(self.vocab)}
    self.index2char = {i: x for (i, x) in enumerate(self.vocab)}

    self.data = [(raw_text[i:i + history_length], raw_text[i + history_length])
                 for i in range(len(raw_text) - history_length)]
    return

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    history, label = self.data[index]
    history = np.array([self.char2index[x] for x in history])
    label = self.char2index[label]
    return history, label

  def get_vocabulary(self):
    return self.vocab

class SentenceGeneration(nn.Module):

  def __init__(self,
               vocabulary_size,
               embedding_dim,
               rnn_module,
               hidden_size,
               bias=False):
    super().__init__()
    self.vocabulary_size = vocabulary_size
    self.rnn_module = rnn_module
    self.embedding_dim = embedding_dim
    self.hidden_size = hidden_size
    self.bias = bias

    self.embedding = nn.Embedding(num_embeddings=vocabulary_size,
                                  embedding_dim=embedding_dim,
                                  padding_idx=PADDING_TOKEN)
    self.rnn_model = self.rnn_module(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     bias=bias)
    self.classifier = nn.Linear(hidden_size, vocabulary_size)
    return

  def forward(self, batch_reviews, state=None):
    data = self.embedding(batch_reviews)

    batch_size, total_steps, _ = data.shape
    for step in range(total_steps):
      next_state = self.rnn_model(data[:, step, :], state)
      if isinstance(next_state, tuple):
        h, c = next_state
        outputs = h
      else:
        outputs = next_state
      state = next_state

    logits = self.classifier(outputs)
    return logits, state

  def reset_parameters(self):
    with torch.no_grad:
      for param in self.parameters():
        param.reset_parameters()
    return

def shakespeare_trainer(batch_size, epochs):
  train_dataset = ShakespeareDataset(encoded_text=shakespeare_text,
                                     history_length=HISTORY_LENGTH)
  
  print('Train dataset: %d' % len(train_dataset))

  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8)
  vocabulary = train_dataset.get_vocabulary()

  best_model = None
  best_loss = 0.0
  full_loss = []

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = SentenceGeneration(vocabulary_size=len(vocabulary),
                             embedding_dim=EMBEDDING_DIM,
                             rnn_module=RNN_MODULES[RNN_MODULE],
                             hidden_size=HIDDEN_SIZE)
  model.to(device)

  print('Model Architecture:\n%s' % model)

  criterion = nn.CrossEntropyLoss(reduction='mean')
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  for epoch in range(epochs):
    model.train()
    dataset = train_dataset
    data_loader = train_loader

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (sequences, labels) in progress_bar:
      total_step = epoch * len(data_loader) + step
      sequences = sequences.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()

      outputs, _ = model(sequences)
      _, preds = torch.max(outputs, 1)
      loss = criterion(outputs, labels)
      corrects = torch.sum(preds == labels.data)

      loss.backward()
      optimizer.step()

      progress_bar.set_description(
          'Loss: %.4f, Accuracy: %.4f' %
          (loss.item(), corrects.item() / len(labels)))
      full_loss.append(loss.item())

  state_dict = {"model": model.cpu().state_dict(),
                "vocabulary": vocabulary}
                 
  return state_dict, full_loss

final_model, loss = shakespeare_trainer(batch_size=BATCH_SIZE,
                                        epochs=EPOCHS)

### You can make a plot using matplotlib with loss

epoch_arr = np.arange(0,2730)
#training loss
plt.plot(epoch_arr, loss)
plt.ylabel('loss')
plt.xlabel('epochs')
plt.show()

def sample_next_char_id(predicted_logits):
  next_char_id = categorical.Categorical(logits=predicted_logits).sample()
  return next_char_id

def shakespeare_writer(state_dict, start_string):
  """Generates new sentences using trained language model."""
  device = 'cpu'

  vocabulary = state_dict['vocabulary']

  char2index = {x: i for (i, x) in enumerate(vocabulary)}
  index2char = {i: x for (i, x) in enumerate(vocabulary)}

  inputs = torch.tensor([char2index[x] for x in start_string])
  inputs = inputs.view(1, -1)
  print(inputs)
  model = SentenceGeneration(vocabulary_size=len(vocabulary),
                             embedding_dim=EMBEDDING_DIM,
                             rnn_module=RNN_MODULES[RNN_MODULE],
                             hidden_size=HIDDEN_SIZE)

  model.load_state_dict(state_dict['model'])
  model.eval()

  generated_chars = []
  num_generate = GENERATION_LENGTH;
  
  for i in range(num_generate):
      #forward(self, x, prev_state)
      predictions, states = model(inputs)
      predictions = predictions / TEMPERATURE
      pred_id = sample_next_char_id(predictions)
      inputs = inputs.squeeze(0)
      generated_chars.append(index2char[pred_id.item()])
      history = torch.cat((inputs, pred_id), 0)
      history = history.unsqueeze(0)
      inputs = history

  #####################################################################

  return start_string + ''.join(generated_chars)

# generated_text = shakespeare_writer(final_model, START_STRING)
print(generated_text)

"""# Please use the original assignment to complete Part III
Because it doesn't require model training.
"""