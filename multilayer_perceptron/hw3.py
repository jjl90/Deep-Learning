import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import TensorBoard
import torchvision.models as models
import skimage.transform
from torchvision import transforms, datasets

torch.cuda.empty_cache()
TRAIN_DIRECTORY_PATH = "cifar10"
IMAGE_SIZE = 32 * 32 * 3
OUTPUT_SIZE = 10

#hyperparameters
BATCH_SIZE = 32
HIDDEN_SIZE = 3548
DROPOUT = None
LEAERNING_RATE = 0.000001
WEIGHT_D = 0.01
EPOCHS = 25

#13 hours of late time used


def read_image(image_path, resize=None):
  """Loads an image using `matplotlib` library."""
  img = np.array(Image.open(image_path), np.float32)
  img = skimage.transform.resize(img,(224,224,3))
  ret = img
  return ret

def print_params():
  print("Batch Size: ",BATCH_SIZE)
  print("Hidden Size: ",HIDDEN_SIZE)
  print("Learning Rate: ",LEAERNING_RATE)
  print("Wight Decay: ",WEIGHT_D)
  print("Epochs: ",EPOCHS)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class CifarDataset(torch.utils.data.Dataset):
  def __init__(self, root_dir, subFolderSize=10, transform=None):
    """Initializes a dataset containing images and labels."""
    self.samples = []
    self.onehot = {}
    i = 0
    for folder in os.listdir(root_dir):
      label = os.path.basename(folder)
      label_filepath = os.path.join(root_dir, label)

      #create one hot array for each folder
      oneHotInit = np.zeros(subFolderSize,dtype=np.float32)
      oneHotInit[i] = 1
      self.onehot[label] = oneHotInit
      i = i+1

      for file in os.listdir(label_filepath):
        picture_filepath = os.path.join(label_filepath, file)
        print(picture_filepath)
        if(transform):
          img = read_image(picture_filepath,True)
        else:
          img = read_image(picture_filepath)

        self.samples.append((self.onehot[label],img))
    self.transform = transform
    
  def __len__(self):
    """Returns the size of the dataset."""
    return len(self.samples)

  def __getitem__(self, index):
    """Returns the index-th data item of the dataset."""
    return self.samples[index]

class MultilayerPerceptron(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super().__init__()
    # self.drop_layer = torch.nn.Dropout(p = dropout)
    self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False)
    self.tanh = torch.nn.Tanh()
    self.fc2 = torch.nn.Linear(hidden_size, OUTPUT_SIZE, bias=False)

  def forward(self, x):
    try:
      x = x.view(x.size(0), IMAGE_SIZE)
    except:
      print(x)
    hidden = self.fc1(x)
    tanh = self.tanh(hidden)
    output = self.fc2(tanh)
    return output
  
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(axis=0) 


def training(model, dataloader, perm=None):
  loss_set = []
  epoch_set = []
  model.train()
  for epoch in range(EPOCHS):
    for i, (labels, images) in enumerate(dataloader):
      
      #move tensors
      images = images.cuda() #x
      labels = labels.cuda() #y
      #forward pass

      if(perm):
        images = images.permute(0,3,1,2)

      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if (i % 10 == 0):
        print("Epoch: ", epoch+1 ," Loss: ",loss)
        
      loss_set.append(loss) 
      epoch_set.append(epoch + 1)

  #print loss diagram
  plt.plot(epoch_set, loss_set, 'o') 
  plt.ylabel('loss') 
  plt.xlabel('epoch') 
  plt.legend() 
  plt.savefig('MLP_TRAINING_LOSS.png')
  plt.clf()
  return model

def evaluate(model, dataloader, dType, perm=None):
  model.eval()
  with torch.no_grad():
    total = 0
    correct = 0
    accuracyArr = []
    iters = []
    for i, (labels, images) in enumerate(dataloader):
      if(i % 100 == 0):
        print("Iteration: ",i)
      images = images.cuda()
      if(perm):
        images = images.permute(0,3,1,2)
      labels = labels.cuda()
      outputs = model(images)
      for x in range(outputs.shape[0]):
        outputSM = softmax(outputs[x])
        max1 = torch.argmax(outputSM)
        max2 = torch.argmax(labels[x])
        if max1 == max2 :
          correct += 1
      total += outputs.shape[0]
      accuracy =  correct / total
      accuracyArr.append(accuracy)
      iters.append(total)

  plt.plot(iters, accuracyArr, 'o') 
  plt.ylabel('accuracy') 
  plt.xlabel('iterations') 
  plt.legend() 
  stringF = 'MLP_ACCURACY_'+ dType +'.png'
  plt.savefig(stringF)
  plt.clf()
  print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
  


if __name__ == "__main__":
  #change depending on how you want to train your model
  model_name = "MobileNetV2"

  #note . . . the split up between part 2 and 3 is for simplicity of grading. 
  if(model_name == "user_model"):
    train_dataset = CifarDataset('cifar10/cifar10_train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #create the model
    neural_network = MultilayerPerceptron(IMAGE_SIZE, HIDDEN_SIZE).cuda()

    #optimizers and loss
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(neural_network.parameters(), lr = LEAERNING_RATE, weight_decay = WEIGHT_D)

    #train the model
    neural_network = training(neural_network, train_dataloader)

    #evaluate on train set
    evaluate(neural_network, eval_dataloader, 'train')

    #evaluation on test set
    test_dataset =  CifarDataset('cifar10/cifar10_test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    evaluate(neural_network, eval_dataloader, 'test')

    #print hyperparameters
    print_params()
  
  elif(model_name == "MobileNetV2"):

    feature_extract = False

    mobile_net = models.mobilenet_v2(pretrained=True)
    set_parameter_requires_grad(mobile_net, feature_extract)
    mobile_net.classifier = torch.nn.Linear(1280,10)
    print(mobile_net)
    params_to_update = mobile_net.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in mobile_net.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
                
    else:
        for name,param in mobile_net.named_parameters():
            if param.requires_grad == True:
                None

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(mobile_net.parameters(), lr = LEAERNING_RATE, weight_decay = WEIGHT_D)

    train_dataset = CifarDataset('cifar10/cifar10_train',10,True)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset =  CifarDataset('cifar10/cifar10_test',10,True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    mobile_net.cuda()
    mobile_net = training(mobile_net,train_dataloader,True)

    #evaluate on train set
    evaluate(mobile_net, eval_dataloader, 'train',True)

    #evaluation on test set
    evaluate(mobile_net, eval_dataloader, 'test',True)