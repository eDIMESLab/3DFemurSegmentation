import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.multiprocessing as mp
from sklearn.model_selection import KFold
import sys
import os
import copy
import numpy as np
from sklearn.model_selection import StratifiedKFold

def set_seed(seed):
  # torch.backends.cudnn.deterministic = True
  # torch.backends.cudnn.benchmark = False
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  # random.seed(seed)

#%%

class EarlyStopping:
  """Early stops the training if validation loss doesn't improve after a given patience."""
  def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
    """
    Args:
         patience (int): How long to wait after last time validation loss improved.
                         Default: 7
         verbose (bool): If True, prints a message for each validation loss improvement.
                         Default: False
         delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                         Default: 0
         path (str): Path for the checkpoint to be saved to.
                         Default: 'checkpoint.pt'
         trace_func (function): trace print function.
                         Default: print
    """
    self.patience = patience
    self.verbose = verbose
    self.counter = 0
    self.best_score = None
    self.early_stop = False
    self.val_loss_min = np.Inf
    self.delta = delta
    self.path = path
    self.trace_func = trace_func
  def __call__(self, val_loss, model):
    score = -val_loss
    if self.best_score is None:
     self.best_score = score
     self.save_checkpoint(val_loss, model)
    elif score < self.best_score + self.delta:
      self.counter += 1
      # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
      if self.counter >= self.patience:
          self.early_stop = True
    else:
      self.best_score = score
      self.save_checkpoint(val_loss, model)
      self.counter = 0

  def save_checkpoint(self, val_loss, model):
    '''Saves model when validation loss decrease.'''
    if self.verbose:
      self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    torch.save(model.state_dict(), self.path)
    self.val_loss_min = val_loss


#%%
class LogisticRegressionModel(nn.Module):
  def __init__(self, input_size, output_dim):
      super(LogisticRegressionModel, self).__init__()
      self.linear = nn.Linear(input_dim, output_dim)

  def forward(self, x):
      out = self.linear(x)
      return out


class FeedforwardNeuralNetModel(nn.Module):
  def __init__(self, input_size, hidden_dim, output_dim):
    super(FeedforwardNeuralNetModel, self).__init__()
    # Input -> Hidden Layer
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    # ReLU
    self.relu = nn.ReLU()
    # Hidden Layer -> Output
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out


class PopulationDataset(Dataset):
  """Dataset class for column dataset.
  Args:
  """
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return [self.features[idx], self.labels[idx] ]

class FoldTorch():
  """
  """
  def __init__(self, Train, Test, Model, Optimizer, Criterion, Epoch):
    self.train_dataset = Train
    self.test_dataset = Test
    self.model = Model
    self.optimizer = Optimizer
    self.criterion = Criterion
    self.epoch = Epoch
    self.result = []
    self.train_losses = []
    self.valid_losses = []

  def train_epoch(self, TLoader):
    self.model.train()
    for batch_idx, (feats, labs) in enumerate(TLoader):
      # feats = feats.view(-1, 28*28).requires_grad_()
      feats = feats.requires_grad_()
      self.optimizer.zero_grad()
      outputs = self.model(feats)
      loss = self.criterion(outputs, labs.max(1)[1])
      loss.backward()
      self.optimizer.step()
      self.train_losses.append(loss.item())

  def test_epoch(self,TLoader):
    self.model.eval()
    with torch.no_grad():
      for (fs, l) in TLoader:
        outputs = self.model(fs)
        loss = self.criterion(outputs, l.max(1)[1])
        self.valid_losses.append(loss.item())

  def train(self, dataloader_kwargs, seed=131, dataloadertest_kwargs={}, patience=7, earlyStop_opt=False):
    set_seed(seed)
    train_loader = DataLoader(self.train_dataset, **dataloader_kwargs)
    if earlyStop_opt:
      early_stopping = EarlyStopping(patience=patience, verbose=True)
      test_loader = DataLoader(self.test_dataset, **dataloadertest_kwargs)
      for epoch in range(self.epoch):
        self.train_epoch(TLoader=train_loader)
        self.test_epoch(TLoader=test_loader)
        train_loss = np.average(self.train_losses)
        valid_loss = np.average(self.valid_losses)
        self.train_losses = []
        self.valid_losses = []
        early_stopping(valid_loss, self.model)
        if early_stopping.early_stop:
          print("Early stopping at epoch number:",epoch)
          break
    else:
      for epoch in range(self.epoch):
        self.train_epoch(TLoader=train_loader)

  def test(self, dataloader_kwargs):
    self.model.eval()
    test_loader = DataLoader(self.test_dataset, **dataloader_kwargs)
    with torch.no_grad():
      for (fs, l) in test_loader:
        # fs = fs.view(-1, 28*28)
        outputs = self.model(fs)
        predicted = outputs.max(1)[1]
        self.result.append((l.max(1)[1], predicted))
        # self.result.append((l, predicted))

  def LoadStateDict(self):
    self.model.load_state_dict(torch.load('checkpoint.pt'))

  def __len__(self):
    return len(self.result)

  def __getitem__(self, idx):
    return self.result[idx]

  @property
  def result(self):
    return self._result
  @result.setter
  def result(self, value):
    self._result=value

  @property
  def model(self):
    return self._model
  @model.setter
  def model(self, value):
    self._model=value



#%%
#
# class KFoldTorch():
#   """
#   """
#   def __init__(self, AllFeatures, AllLabels, Split=None, seed=101):
#     self.N = len(AllFeatures.index)
#     self.M = len(AllLabels)
#     self.data = torch.Tensor(AllFeatures.T.values)
#     le = LabelEncoder()
#     le.fit(AllLabels)
#     value_idxs = le.transform(AllLabels)
#     self.labels = torch.eye(len(AllLabels.cat.categories))[value_idxs]
#     self.LeaveOut = self.M if Split is None else Split
#     self.kf = KFold(n_splits=self.LeaveOut, random_state=seed, shuffle=True)
#     self.Folds = []
#     self.mod = None
#     self.opt = None
#     self.crit = None
#     self.nepoch = None
#     self.kwargs_train = None
#     self.kwargs_test = None
#
#   def RunFold(self, trn_split, tst_split, getAcc=False):
#     Trn = PopulationDataset(self.data[trn_split], self.labels[trn_split])
#     Tst = PopulationDataset(self.data[tst_split], self.labels[tst_split])
#     mod_ = copy.deepcopy(self.mod)
#     opt_ = copy.deepcopy(self.opt)
#     FoldObj = FoldTorch(Trn, Tst, mod_, opt_, self.crit, self.nepoch)
#     FoldObj.train(self.kwargs_train, getAcc=getAcc)
#     FoldObj.test(self.kwargs_test)
#     return FoldObj
#
#   def AppendFolds(self, ReturnedFolds):
#     for FoldObj in ReturnedFolds:
#       self.Folds.append(FoldObj)
#
#   def CollectPredictions(self):
#     outlist = []
#     for F in self.Folds:
#       outlist.append(F.result)
#     return outlist
#
#   def Run(self, mod, opt, crit, nepoch, kwargs_train, kwargs_test, cores=1, getAcc=False):
#     self.mod = mod
#     self.opt = opt
#     self.crit = crit
#     self.nepoch = nepoch
#     self.kwargs_train = kwargs_train
#     self.kwargs_test = kwargs_test
#     if cores<2:
#       tot = len([x for x in self.kf.split(self.data)])
#       norm_term = 1/tot*100
#       i=1
#       for train_idxs, test_idxs in self.kf.split(self.data):
#         ret = self.RunFold(train_idxs,test_idxs, getAcc=getAcc)
#         self.Folds.append(ret)
#         print("       ", end="\r", flush=True)
#         print("".join([str(i*norm_term),"%"]), end="\r", flush=True)
#         i+=1
#     # else:
#     #   p = mp.Pool(cores)
#     #   for train_idxs, test_idxs in self.kf.split(self.data):
#     #     ret = p.apply_async(self.RunFold,
#     #                         args=(train_idxs,test_idxs),
#     #                         callback=self.AppendFolds)
#     #     ret.get()
#     #   p.close()
#     #   p.join()
#
#   def __len__(self):
#     return len(self.Folds)
#
#   def __getitem__(self, idx):
#     return self.Folds[idx]


#%%

allData  = [f for f in os.listdir("/home/PERSONALE/daniele.dallolio3/NMF_samples/") if f.startswith("NMF_diagnosis_")]
allDataR = [f for f in os.listdir("/home/PERSONALE/daniele.dallolio3/NMF_samples/") if f.startswith("NMF_relapse_")]
allData  = sorted(allData, key=lambda x: int(x.replace("NMF_diagnosis_", "").replace(".csv", "")))
allDataR = sorted(allDataR, key=lambda x: int(x.replace("NMF_relapse_", "").replace(".csv", "")))
allData  = [os.path.join("/home/PERSONALE/daniele.dallolio3/NMF_samples/", f) for f in allData]
allDataR = [os.path.join("/home/PERSONALE/daniele.dallolio3/NMF_samples/", f) for f in allDataR]

label_df = pd.ExcelFile("Evolution_trajectories_80samples_250620.xlsx")
label_df = label_df.parse(label_df.sheet_names[0])
labels_ = label_df["Trajectory_19_HIGH_RISK"].astype('category').cat.reorder_categories(['S', 'B', 'L', 'D'])
pat_idxs = label_df["CLONALV_new1_SNP"]

AllLabels = labels_
M = len(AllLabels)
le = LabelEncoder()
le.fit(AllLabels)
value_idxs = le.transform(AllLabels)
labels = torch.eye(len(AllLabels.cat.categories))[value_idxs]
LeaveOut = 4
seed = 101
# kf = KFold(n_splits=LeaveOut, random_state=seed, shuffle=True)
kf = StratifiedKFold(n_splits=LeaveOut, random_state=seed, shuffle=True)
kf_split = [(x,y) for x,y in kf.split(AllLabels,AllLabels)]
tot = len(kf_split)
norm_term = 1/tot*100
output_dim = len(set(labels_))
epoch = int(1e4)

#%%

for fn, fnR in zip(allData, allDataR):
  idx = os.path.basename(fn).replace("NMF_diagnosis_", "").replace(".csv", "")
  print("       ", end="\r", flush=True)
  print(idx)
  data_diagnosis = pd.read_csv(fn).astype('float64')
  data_diagnosis.columns = [ c.replace("_E", "") for c in data_diagnosis.columns ]
  data_relapse   = pd.read_csv(fnR).astype('float64')
  data_relapse.columns = [ c.replace("_R", "") for c in data_relapse.columns ]
  # 1 diagnosis
  # data_ = data_diagnosis.copy()
  # 2 difference
  # data_ = (data_diagnosis-data_relapse).copy()
  # 3 appended
  data_ = data_diagnosis.append(data_relapse)
  data_ = data_.reset_index(drop=True).copy()
  assert label_df["CLONALV_new1_SNP"].tolist() == data_.columns.tolist()

  AllFeatures = data_
  # Split=len(data_.T)

  N = len(AllFeatures.index)
  data = torch.Tensor(AllFeatures.T.values)

  input_dim = len(data_)
  # batch_size = len(data_.T) -1

  i=0
  output = pd.DataFrame({"CLONALV_new1_SNP":[], "Expected":[], "Predicted":[]})
  for train_idxs, test_idxs in kf_split:
    Trn = PopulationDataset(data[train_idxs], labels[train_idxs])
    Tst = PopulationDataset(data[test_idxs], labels[test_idxs])
    mod_ = FeedforwardNeuralNetModel(input_dim, input_dim, output_dim)
    opt_ = torch.optim.Adam(mod_.parameters(131), lr=0.1)
    FoldObj = FoldTorch(Trn, Tst, mod_, opt_, nn.CrossEntropyLoss(), epoch)
    FoldObj.train({'batch_size': len(Trn), 'shuffle': True, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)},
                  dataloadertest_kwargs={'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)},
                  earlyStop_opt=True,
                  patience=100)
    FoldObj.test({'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)})
    # output.loc[i,"Expected"] = FoldObj.result[0][0].item()
    # output.loc[i,"Predicted"] = FoldObj.result[0][1].item()
    tmp = pd.DataFrame({"CLONALV_new1_SNP":pat_idxs[test_idxs].tolist(), "Expected":[x[0].item() for x in FoldObj.result], "Predicted":[x[1].item() for x in FoldObj.result]})
    output = output.append(tmp)
    print("       ", end="\r", flush=True)
    i+=1
    print("".join([str(i*norm_term),"%"]), end="\r", flush=True)
  output = output.reset_index(drop=True)
  output.to_csv(os.path.join("/home/PERSONALE/daniele.dallolio3/NMF_samples/pop_opt", "".join(["predictions_",idx,".csv"]) ), index=True)



#%%
from sklearn.metrics import f1_score
acc = []
fld = "appendWithEarly"
for idx in range(2,40):
  fn = os.path.join("/home/PERSONALE/daniele.dallolio3/NMF_samples/pop_opt", fld,"".join(["predictions_",str(idx),".csv"]) )
  df = pd.read_csv(fn)
  # acc.append((df["Expected"] == df["Predicted"]).sum()/len(df.index)*100
  acc.append(f1_score(df["Expected"].tolist(), df["Predicted"].tolist(), average='weighted'))

import matplotlib.pylab as plb
plb.hist(acc)
max(acc)

plb.plot(np.arange(2,40), acc)

np.arange(2,40)[np.argmax(acc)]
labels_.value_counts()

#%%


labels_naive = [ "L" for i in range(len(labels_))]
f1_score(labels_.tolist(), labels_naive, average='weighted')


#%%
# Train/Test on all dataset
Trn = PopulationDataset(data[train_idxs], labels[train_idxs])
Tst = PopulationDataset(data[test_idxs], labels[test_idxs])
mod_ = FeedforwardNeuralNetModel(input_dim, input_dim, output_dim)
opt_ = torch.optim.Adam(mod_.parameters(131), lr=0.1)
FoldObj = FoldTorch(Trn, Tst, mod_, opt_, nn.CrossEntropyLoss(), epoch)
FoldObj.train({'batch_size': len(Trn), 'shuffle': True, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)},
              dataloadertest_kwargs={'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)},
              earlyStop_opt=True,
              patience=100)
FoldObj.test({'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)})





#%%
#
# TEST CLASS WITH KNOWN DATASET
#
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
#
# train_dataset = dsets.MNIST(root='./data',
#                             train=True,
#                             transform=transforms.ToTensor(),
#                             download=True)
# test_dataset = dsets.MNIST(root='./data',
#                            train=False,
#                            transform=transforms.ToTensor())
# batch_size = 100
# n_iters = 3000
# num_epochs = n_iters / (len(train_dataset) / batch_size)
# num_epochs = int(num_epochs)
#
# input_dim = 28*28
# hidden_dim = 100
# output_dim = 10
# model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
# learning_rate = 0.1
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# FoldObj = FoldTorch(train_dataset, test_dataset, model, optimizer, nn.CrossEntropyLoss(), num_epochs)
# FoldObj.train({'batch_size': batch_size, 'shuffle': True})
# FoldObj.test({'batch_size': batch_size, 'shuffle': False})
#
# correct = [(x == y).sum().item() for x,y in FoldObj.result]
# sum(correct)/sum([len(x) for x,y in FoldObj.result]) #0.9721


#%%
# input_dim = len(data_)
# batch_size = len(data_.T) -1
# output_dim = len(set(labels_))
# # model = LogisticRegressionModel(input_dim, output_dim)
# model = FeedforwardNeuralNetModel(input_dim, input_dim, output_dim)
# # optimizer_prova = torch.optim.SGD(model.parameters(131), lr=0.001)
#
#
# KFoldObj = KFoldTorch(data_, labels_, Split=len(data_.T))
# KFoldObj.Run(mod=model,
#              opt=optimizer_prova,
#              crit=nn.CrossEntropyLoss(),
#              nepoch=int(1e2),
#              kwargs_train={'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)},
#              kwargs_test={'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'worker_init_fn':np.random.seed(0)},
#              cores=1,
#              getAcc=True)
# len(KFoldObj)
# [ f.result for f in KFoldObj.Folds ]




#%%

###########
# THE END #
###########
