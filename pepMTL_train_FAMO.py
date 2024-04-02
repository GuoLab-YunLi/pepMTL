# -*- coding: utf-8 -*-
"""
@author: Deep
"""
cd C:\Users\deep\Desktop\pepMTL


import pickle

Train = "./fine_tune_data/Train_data.pkl"
Test = "./fine_tune_data/Test_data.pkl"
with open(Train, 'rb') as f:
     Train_data = pickle.load(f)
with open(Test, 'rb') as f:
     Test_data = pickle.load(f)

Train_data = Train_data.assign(CE=42)
Test_data = Test_data.assign(CE=42)

## unknown:0, QE:1 (QE, QE+, QEHF, QEHFX, Exploris),Tribrid:2 (Lumos, Fusion, Eclipse,  Ascend),
## timsTOF:3, SciexTOF:4）
Train_data = Train_data.assign(Analyzer=3)
Test_data = Test_data.assign(Analyzer=3)
 

import pandas as pd
import numpy as np
import math

def data_norm(Train, Test):
    train = Train.copy(); test = Test.copy()
    ## RT
    train_RT = train['Retention time']
    test_RT = test['Retention time']
    RT = pd.concat([train_RT, test_RT], ignore_index=True)
    max_RT = math.ceil(max(RT)); min_RT = math.floor(min(RT))
    RT_scale = {'max_RT':max_RT, 'min_RT':min_RT}
    train_RT_norm = (train_RT-min_RT)/(max_RT-min_RT)
    test_RT_norm = (test_RT-min_RT)/(max_RT-min_RT)
    ## CCS
    train_CCS = train['CCS']
    test_CCS = test['CCS']
    CCS = pd.concat([train_CCS, test_CCS], ignore_index=True)
    max_CCS = math.ceil(max(CCS)); min_CCS = math.floor(min(CCS))
    CCS_scale = {'max_CCS':max_CCS, 'min_CCS':min_CCS}
    train_CCS_norm = (train_CCS-min_CCS)/(max_CCS-min_CCS)
    test_CCS_norm = (test_CCS-min_CCS)/(max_CCS-min_CCS)
    ## MSMS
    train_MSMS = train['Intensities']
    test_MSMS = test['Intensities']
    train_MSMS_norm = train_MSMS.apply(lambda x: x / x.max())
    test_MSMS_norm = test_MSMS.apply(lambda x: x / x.max())
    ## MS1 Intensity
    Intensity = pd.concat([train['Intensity'], test['Intensity']], ignore_index=True)
    train_Intensity_norm = train['Intensity']*10/max(Intensity)
    test_Intensity_norm = test['Intensity']*10/max(Intensity)
    ##
    train['Retention time'] = train_RT_norm
    test['Retention time'] = test_RT_norm
    train['CCS'] = train_CCS_norm
    test['CCS'] = test_CCS_norm
    train['Intensities'] = train_MSMS_norm
    test['Intensities'] = test_MSMS_norm
    train['Intensity'] = train_Intensity_norm
    test['Intensity'] = test_Intensity_norm
    return train, test, RT_scale, CCS_scale

Train_data, Test_data, RT_scale, CCS_scale = data_norm(Train_data, Test_data)



from datasets import Dataset
Train_dataset = Dataset.from_pandas(Train_data)
Test_dataset = Dataset.from_pandas(Test_data)
print(Train_dataset[0])



from transformers import AutoTokenizer
pre_trained_path = "./pretrain_models/esm2_t12_35M_UR50D_MMPD"
tokenizer = AutoTokenizer.from_pretrained(pre_trained_path, model_max_length=72) 
print(tokenizer) 

def tokenize_dataset(data):
    return tokenizer(data['Modified sequence'], max_length=72, truncation=True, padding="max_length")
Train_dataset = Train_dataset.map(tokenize_dataset)
Test_dataset = Test_dataset.map(tokenize_dataset)
print(Train_dataset)

Train_dataset.set_format("torch")
Test_dataset.set_format("torch")
print(Train_dataset)
print(Train_dataset[0])



## Empty cache
import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

from torch import nn
from transformers import AutoModel, AutoConfig
pre_trained_path = "./pretrain_models/esm2_t12_35M_UR50D_MMPD"
config = AutoConfig.from_pretrained(pre_trained_path, hidden_dropout_prob = 0.1)
print(config)

## MTL prediction model
class pepMTL(nn.Module): 
    def __init__(self, pre_trained_path, Drop_RT, Drop_CCS, Drop_MSMS, MS2_dim):
        super().__init__()  
        self.Meta_dim = 4
        self.FT_embed = nn.Embedding(10, self.Meta_dim, padding_idx=0)
        self.A_embed = nn.Embedding(10, self.Meta_dim, padding_idx=0)
        self.SL_embed = nn.Linear(1, self.Meta_dim)
        self.C_embed = nn.Linear(1, self.Meta_dim) 
        self.CE_embed = nn.Linear(1, self.Meta_dim)
        self.Int_embed = nn.Linear(1, self.Meta_dim)
        self.Embedding_dim = config.hidden_size
        self.Embedding = AutoModel.from_pretrained(pre_trained_path, config=config)
        ## RT Prediction header
        self.RT_header = nn.Sequential(nn.Linear(self.Embedding_dim, 128),
                                       nn.BatchNorm1d(128),                                      
                                       nn.GELU(),
                                       nn.Dropout(Drop_RT),
                                       nn.Linear(128, 1))
        ## CCS Prediction header        
        self.CCS_header = nn.Sequential(nn.Linear(self.Embedding_dim+self.Meta_dim, 128),
                                        nn.BatchNorm1d(128),
                                        nn.GELU(),
                                        nn.Dropout(Drop_CCS),
                                        nn.Linear(128, 1))
        ## MSMS Prediction header
        self.MSMS_header = nn.Sequential(nn.Linear(self.Embedding_dim+self.Meta_dim*6, 256),
                                         nn.Dropout(Drop_MSMS),
                                         nn.GELU(),
                                         nn.Linear(256, MS2_dim))        
    def forward(self, input_ids, attention_mask, Charge, SeqLen, CE, FT, Analyzer, Intensity):
        ## CCS_Meta_embedding
        CCS_Meta_embedding = self.C_embed(Charge)
        ## MSMS_Meta_embedding
        MSMS_Meta_embedding = torch.cat((self.FT_embed(FT), self.C_embed(Charge), self.A_embed(Analyzer),
                                         self.SL_embed(SeqLen), self.CE_embed(CE), self.Int_embed(Intensity)), 1)
        MSMS_Meta_embedding = MSMS_Meta_embedding.unsqueeze(1).repeat(1,config.max_position_embeddings-1,1)
        Embeddings = self.Embedding(input_ids, attention_mask) 
        ## last_hidden_state, pooler_output = Seq_embedding[0], Seq_embedding[1]
        Seq_embedding = Embeddings[0]
        ## RT_Seq_embedding
        RT_Seq_embedding = Seq_embedding[:, 0, :] 
        ## CCS_Seq_embedding
        CCS_Seq_embedding = torch.cat((Seq_embedding[:, 0, :], CCS_Meta_embedding), dim=1)
        ## MSMS_Seq_embedding
        MSMS_Seq_embedding = torch.cat((Seq_embedding[:, 1:, :], MSMS_Meta_embedding), dim=2)
        ### Prediction headers
        Pred_RT = self.RT_header(RT_Seq_embedding)
        Pred_RT = Pred_RT.squeeze(-1)
        Pred_CCS = self.CCS_header(CCS_Seq_embedding)
        Pred_CCS = Pred_CCS.squeeze(-1)
        Pred_MSMS = self.MSMS_header(MSMS_Seq_embedding)
        return Pred_RT, Pred_CCS, Pred_MSMS

import numpy as np
import ms_entropy as me
import numpy.linalg as L
from scipy.stats import pearsonr
from scipy.stats import spearmanr

## Unweighted entropy similarity
def UES(msms, msms_pre, masses): ## msms/msms_pre (n,1) array
    m_msms = np.stack((masses, msms), axis=-1)
    m_msms_pre = np.stack((masses, msms_pre), axis=-1)
    ues = me.calculate_unweighted_entropy_similarity(m_msms, m_msms_pre)
    return round(ues, 4)

## cosine similarity, it gives the same result as the dot product
def COS(msms, msms_pre): ## msms/msms_pre (n,1) array
    dot = np.dot(msms,msms_pre)
    return round(dot/(L.norm(msms)*L.norm(msms_pre)),4)

## Pearson correlation coefficient
def PCC(msms,msms_pre): ## msms/msms_pre (n,1) array
    return round(pearsonr(msms,msms_pre)[0],4)

## Spearman correlation coefficient
def SPC(msms,msms_pre):
    return round(spearmanr(msms,msms_pre)[0],4)

###  spectral contrast angle
def SA(msms, msms_pre):   ## msms/msms_pre (n,1) array
    L2normed_act = msms / L.norm(msms)
    L2normed_pred = msms_pre / L.norm(msms_pre)
    inner_product = np.dot(L2normed_act, L2normed_pred)
    return round(1 - 2*np.arccos(inner_product)/np.pi,4)


## Model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
criterion = nn.MSELoss()
MS2_dim = 8

## https://github.com/Cranial-XIX/FAMO
from famo import FAMO
#from famo_lion import FAMO
weight_opt = FAMO(n_tasks=3, device=device)

from tqdm import tqdm
import time
import sklearn.metrics
from lion_pytorch import Lion
from math import cos, pi

## Warm up + Cosine Anneal
def warmup_cosine(optimizer, current_step, max_step, lr_min, lr_max, warmup_step):
    if current_step < warmup_step:
        lr = lr_max * current_step / warmup_step
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_step - warmup_step) / (max_step - warmup_step))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def model_out(model, batch, criterion, device):
    attention_mask = batch['attention_mask'].to(device)
    input_ids = batch['input_ids'].to(device)
    RT = batch['Retention time'].to(device)
    CCS = batch['CCS'].to(device)
    MSMS = batch['Intensities'].to(device) 
    MSMS = MSMS.unsqueeze(-1).reshape(-1, MS2_dim, int(MSMS.shape[1]/MS2_dim)).transpose(1, 2)
    Masses = batch['Masses'].to(device)
    ## Meta info
    Charge = batch['Charge'].float().unsqueeze(-1).to(device)
    CE = batch['CE'].float().unsqueeze(-1).to(device)
    FT = batch['Fragmentation'].to(torch.long).to(device)
    SeqLen = batch['Length'].float().unsqueeze(-1).to(device)
    Analyzer = batch['Analyzer'].to(torch.long).to(device)
    Intensity = batch['Intensity']
    Intensity = Intensity.unsqueeze(-1).to(device)
    ##
    Pred_RT, Pred_CCS, Pred_MSMS = model(input_ids, attention_mask, Charge, SeqLen, CE, FT, Analyzer, Intensity)
    ## Loss
    Loss_RTi = criterion(Pred_RT, RT)
    Loss_CCSi = criterion(Pred_CCS, CCS)
    Loss_MSMSi = criterion(Pred_MSMS, MSMS)
    ## R2
    RT_R2i = sklearn.metrics.r2_score(RT.cpu().detach().numpy(), Pred_RT.cpu().detach().numpy())
    CCS_R2i = sklearn.metrics.r2_score(CCS.cpu().detach().numpy(), Pred_CCS.cpu().detach().numpy())     
    ## Unweighted entropy similarity
    MSMS_UESi = []
    for i in range(len(MSMS)):
        pred_msmsi = Pred_MSMS[i,:,:]
        msmsi = MSMS[i,:,:]
        pred_msmsi = pred_msmsi.transpose(1, 0).reshape(msmsi.shape[0]*msmsi.shape[1])
        msmsi = msmsi.transpose(1, 0).reshape(msmsi.shape[0]*msmsi.shape[1])
        massesi = Masses[i,:]
        indexi = (massesi != 0)
        massesi = massesi[indexi]
        msmsi = msmsi[indexi]
        pred_msmsi = pred_msmsi[indexi]
        msms_UESi = UES(msmsi.cpu().detach().numpy(), pred_msmsi.cpu().detach().numpy(), massesi.cpu().detach().numpy())
        MSMS_UESi.append(msms_UESi)
    return Loss_RTi, Loss_CCSi, Loss_MSMSi, RT_R2i, CCS_R2i, MSMS_UESi
    
def model_train(model, criterion, train_data, test_data, batch_size, max_learning_rate, min_learning_rate, epochs):    
    train_data = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_data = DataLoader(dataset=test_data, batch_size=batch_size, drop_last=True)
    #
    criterion = criterion
    optimizer = Lion(model.parameters(),weight_decay=1e-2)
    ## Total step
    max_step = len(train_data)*epochs
    start = time.perf_counter()                 
    for epoch in range(epochs):
        RT_R2 = []; CCS_R2 = []; MSMS_UES = []
        model.train()
        loop = tqdm(enumerate(train_data), total =len(train_data))
        for step, batch in loop:
            # warmup_cosine
            current_step = len(train_data)*epoch + step + 1
            warmup_cosine(optimizer, current_step, max_step, min_learning_rate, max_learning_rate, max_step/10)
            ##使用 model_out 获取训练的结果
            Loss_RTi, Loss_CCSi, Loss_MSMSi, RT_R2i, CCS_R2i, MSMS_UESi = model_out(model, batch, criterion, device)
            RT_R2.append(RT_R2i)
            CCS_R2.append(CCS_R2i)
            MSMS_UES.append(MSMS_UESi)
            optimizer.zero_grad()  ## important
            ## FAMO
            weight_opt.backward(torch.stack((Loss_RTi, Loss_CCSi, Loss_MSMSi)))
            optimizer.step() 
            # update the task weighting
            with torch.no_grad():
                 Loss_RTii, Loss_CCSii, Loss_MSMSii, RT_R2ii, CCS_R2ii, MSMS_UESii = model_out(model, batch, criterion, device)
                 weight_opt.update(torch.stack((Loss_RTii, Loss_CCSii, Loss_MSMSii)).detach())
            ##
            loop.set_description(f'>> Epoch: [{epoch+1}/{epochs}]')
            loop.set_postfix(RT_loss='{:.2e}'.format(Loss_RTi), CCS_loss='{:.2e}'.format(Loss_CCSi), MSMS_loss='{:.2e}'.format(Loss_MSMSi),
                             LR='{:.2e}'.format(optimizer.param_groups[0]['lr']),RT_R2='{:.4f}'.format(RT_R2i), CCS_R2='{:.4f}'.format(CCS_R2i),
                             MSMS_UES='{:.4f}'.format(np.mean(MSMS_UESi)))         
        print('>> Epoch:', '%04d' % (epoch+1))
        MSMS_UES = np.concatenate(MSMS_UES, axis=0)
        RT_R2 = np.array(RT_R2); CCS_R2 = np.array(CCS_R2)
        print('>> @@ model training @@ ## RT_R2:', '{:.4f}'.format(np.mean(RT_R2)), '## CCS_R2:', '{:.4f}'.format(np.mean(CCS_R2)),
              '## Mean_UES:', '{:.4f}'.format(np.mean(MSMS_UES)), '## Median_UES:', '{:.4f}'.format(np.median(MSMS_UES)),', n=', len(train_data)*batch_size)
        ##################################################################################################################
        model.eval()
        with torch.no_grad():
            # testing
            Loss_RT_t = []; Loss_CCS_t = []; Loss_MSMS_t = []
            RT_R2_t = []; CCS_R2_t = []; MSMS_UES_t = []
            for batch_t in test_data:
                Loss_RTi_t, Loss_CCSi_t, Loss_MSMSi_t, RT_R2i_t, CCS_R2i_t, MSMS_UESi_t = model_out(model, batch_t, criterion, device)
                Loss_RT_t.append(Loss_RTi_t.item())
                Loss_CCS_t.append(Loss_CCSi_t.item())
                Loss_MSMS_t.append(Loss_MSMSi_t.item())
                RT_R2_t.append(RT_R2i_t)
                CCS_R2_t.append(CCS_R2i_t)
                MSMS_UES_t.append(MSMS_UESi_t)
            MSMS_UES_t = np.concatenate(MSMS_UES_t, axis=0)
            RT_R2_t = np.array(RT_R2_t); CCS_R2_t = np.array(CCS_R2_t)    
            print('>> && model testing && $$ RT_R2:', '{:.4f}'.format(np.mean(RT_R2_t)), '$$ CCS_R2:', '{:.4f}'.format(np.mean(CCS_R2_t)),
                  '$$ Mean_UES:', '{:.4f}'.format(np.mean(MSMS_UES_t)), '$$ Median_UES:', '{:.4f}'.format(np.median(MSMS_UES_t)),
                  ', n=', len(test_data)*batch_size)
    end = time.perf_counter() 
    print('>> Model training and testing time cost: %s minute0s'%((end-start)/60))
    return model


### Instantiating a model
model = pepMTL(pre_trained_path, Drop_RT=0, Drop_CCS=0.1, Drop_MSMS=0.1, MS2_dim=8).to(device)
# Print the model and show the structure
print(model) 
## Model training
## stage 1
Model = model_train(model, criterion, Train_dataset, Test_dataset, batch_size = 128, 
                    max_learning_rate = 2e-4, min_learning_rate = 1e-4, epochs = 10)
## stage 2
Model = model_train(Model, criterion, Train_dataset, Test_dataset, batch_size = 128, 
                    max_learning_rate = 1e-4, min_learning_rate = 1e-5, epochs =5) 

# Model = torch.load('./training results/R2/pep_MTL_R2.pth')


Peaks = pd.read_excel("./fine_tune_data/Ion_index/Maxquant_ion_index.xlsx")
Peaks_dict = dict(zip(Peaks.ion,Peaks.num))

def model_test(model, Peaks_dict, data, batch_size, RT_scale, CCS_scale):
    datasets = DataLoader(dataset=data, batch_size=batch_size)
    matches = np.array(list(Peaks_dict.keys()))
    model.eval()
    with torch.no_grad():
        All_RT = []; All_CCS = []; All_MSMS = []; All_Masses=[]; All_Seq=[]; All_Len=[]; All_Charge=[]
        Pred_RT = []; Pred_CCS = []; Pred_MSMS = []
        for batch in tqdm(datasets):
            attention_mask = batch['attention_mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            RT = batch['Retention time'];CCS = batch['CCS'];MSMS = batch['Intensities']
            MSMS = MSMS.unsqueeze(-1).reshape(-1, MS2_dim, int(MSMS.shape[1]/MS2_dim)).transpose(1, 2)
            Masses = batch['Masses']; Seq = batch['Modified sequence']
            ## Meta
            Charge = batch['Charge'].float().unsqueeze(-1).to(device)
            CE = batch['CE'].float().unsqueeze(-1).to(device)
            FT = batch['Fragmentation'].to(torch.long).to(device)
            SeqLen = batch['Length'].float().unsqueeze(-1).to(device)
            Analyzer = batch['Analyzer'].to(torch.long).to(device)
            Intensity = batch['Intensity'].unsqueeze(-1).to(device)
            ##
            Pred_RTi, Pred_CCSi, Pred_MSMSi = model(input_ids, attention_mask, Charge, SeqLen, CE, FT, Analyzer, Intensity)
            ##
            All_RT.append(RT);All_CCS.append(CCS);All_MSMS.append(MSMS);All_Masses.append(Masses);All_Seq.append(Seq)
            Pred_RT.append(Pred_RTi.cpu().detach().numpy())
            Pred_CCS.append(Pred_CCSi.cpu().detach().numpy())
            Pred_MSMS.append(Pred_MSMSi.cpu().detach().numpy())
            All_Len.append(SeqLen.cpu().detach().numpy())
            All_Charge.append(Charge.cpu().detach().numpy())
        All_RT = np.concatenate(All_RT, axis=0);All_CCS = np.concatenate(All_CCS, axis=0)
        All_MSMS = np.concatenate(All_MSMS, axis=0);All_Masses = np.concatenate(All_Masses, axis=0)
        All_Seq = np.concatenate(All_Seq, axis=0); All_Len = np.concatenate(All_Len, axis=0);
        All_Charge = np.concatenate(All_Charge, axis=0)
        Pred_RT = np.concatenate(Pred_RT, axis=0);Pred_CCS = np.concatenate(Pred_CCS, axis=0) 
        Pred_MSMS = np.concatenate(Pred_MSMS, axis=0)
        ## Reverse RT
        All_RT = All_RT*(RT_scale['max_RT']-RT_scale['min_RT'])+RT_scale['min_RT']
        Pred_RT = Pred_RT*(RT_scale['max_RT']-RT_scale['min_RT'])+RT_scale['min_RT']
        ## Reverse CCS
        All_CCS = All_CCS*(CCS_scale['max_CCS']-CCS_scale['min_CCS'])+CCS_scale['min_CCS']
        Pred_CCS = Pred_CCS*(CCS_scale['max_CCS']-CCS_scale['min_CCS'])+CCS_scale['min_CCS']
        ## Unweighted entropy similarity
        MSMS_UES = []; MSMS_result = []
        for i in range(len(All_MSMS)):
            pred_msmsi = Pred_MSMS[i,:,:];msmsi = All_MSMS[i,:,:]
            pred_msmsi = pred_msmsi.transpose(1, 0).reshape(msmsi.shape[0]*msmsi.shape[1])
            msmsi = msmsi.transpose(1, 0).reshape(msmsi.shape[0]*msmsi.shape[1])
            massesi = All_Masses[i,:]
            indexi = (massesi != 0)
            massesi = massesi[indexi];msmsi = msmsi[indexi];pred_msmsi = pred_msmsi[indexi]
            matchesi = matches[indexi]
            pred_msmsi = np.where(pred_msmsi < 0, 0, pred_msmsi)
            MSMS_resulti = np.stack([matchesi,massesi, msmsi, pred_msmsi], axis=-1)
            ##
            MSMS_UESi = UES(msmsi, pred_msmsi, massesi)
            MSMS_UES.append(MSMS_UESi);MSMS_result.append(MSMS_resulti)
        ##
        results = pd.DataFrame()
        results['Sequence'] = All_Seq;results['Length'] = All_Len;results['Charge'] = All_Charge
        results['RT'] = All_RT;results['RT_pre'] = Pred_RT;results['CCS'] = All_CCS
        results['CCS_pre'] = Pred_CCS;results['MSMS_UES'] = MSMS_UES;results['MSMS_out'] = MSMS_result
        ##
        df_RT = results.drop_duplicates(subset='Sequence', keep='first')
        ##
        df_CCS = results.drop_duplicates(subset= ['Sequence','Charge'], keep='first')
        ## RMSE、R2、UES
        RMSE_RT = sklearn.metrics.mean_squared_error(df_RT['RT'], df_RT['RT_pre'], squared=False)
        RMSE_CCS = sklearn.metrics.mean_squared_error(df_CCS['CCS'], df_CCS['CCS_pre'], squared=False)
        RT_R2 = sklearn.metrics.r2_score(df_RT['RT'], df_RT['RT_pre'])
        CCS_R2 = sklearn.metrics.r2_score(df_CCS['CCS'], df_CCS['CCS_pre'])
        print('The prediction results of the dataset is:')
        print('RT_R2=', round(RT_R2, 4))
        print('RT_RMSE=', round(RMSE_RT, 4))
        print('CCS_R2=', round(CCS_R2, 4))
        print('CCS_RMSE=', round(RMSE_CCS, 4))        
        print("Mean UES =", round(np.mean(MSMS_UES), 4))
        print("Median UES =", round(np.median(MSMS_UES), 4))
    return results

## Outputs the model predictions
Train_results = model_test(Model, Peaks_dict, Train_dataset, 512, RT_scale, CCS_scale)            
Test_results = model_test(Model, Peaks_dict, Test_dataset, 512, RT_scale, CCS_scale)            
        
Train_RT = np.array(Train_results[['RT','RT_pre']])
Test_RT = np.array(Test_results[['RT','RT_pre']])
Train_CCS = np.array(Train_results[['CCS','CCS_pre']])
Test_CCS = np.array(Test_results[['CCS','CCS_pre']])
Train_UES = np.array(Train_results['MSMS_UES'])
Test_UES = np.array(Test_results['MSMS_UES'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(Train_RT[:,0], Train_RT[:,1], s=0.1)
ax.plot([0, 120], [0, 120], ls='-', c='black',lw=0.5)
plt.show()
fig, ax = plt.subplots()
ax.scatter(Test_RT[:,0], Test_RT[:,1], s=0.1)
ax.plot([0, 120], [0, 120], ls='-', c='black',lw=0.5)
plt.show()
fig, ax = plt.subplots()
ax.scatter(Train_CCS[:,0], Train_CCS[:,1], s=0.1)
ax.plot([300, 1050], [300, 1050], ls='-', c='black',lw=0.5)
plt.show()
fig, ax = plt.subplots()
ax.scatter(Test_CCS[:,0], Test_CCS[:,1], s=0.1)
ax.plot([300, 1050], [300, 1050], ls='-', c='black',lw=0.5)
plt.show()

######################################################################################
### Saving the model
torch.save(Model, './training results/pep_MTL_R1.pth')

### Saving model results
Result = {'Train_results':Train_results,'Test_results':Test_results}
with open("./training results/training_results.pkl", 'wb') as f:
    pickle.dump(Result, f)





















