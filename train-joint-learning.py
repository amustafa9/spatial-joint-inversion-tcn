# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from os.path import join
import zipfile
from core.utils import extract, standardize
from core.datasets import SeismicDataset2D
from torch.utils.data import DataLoader
from core.model2D import Model2D
from sklearn.metrics import r2_score
import errno
import argparse

    
def preprocess(no_wells_marmousi, no_wells_seam):
    """Function initializes data, performs standardization, and train test split
    
    Parameters:
    ----------
    no_wells_marmousi : int,
        number of evenly spaced wells and seismic samples to be evenly sampled 
        from marmousi section.
        
    no_wells_seam : int
        number of evenly spaced wells and seismic samples to be evenly sampled from SEAM
        
    Returns
    -------
    seismic_marmousi : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section for marmousi
        
    seismic_seam : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section for SEAM
        
    model_marmousi : array_like, shape(num_wells, depth samples)
        2-D array containing model section from marmousi 2
        
    model_seam : array_like, shape(num_wells, depth samples)
        2-D array containing model section from SEAM
    
    """
    
    # get project root directory
    project_root = os.getcwd()
    
    if ~os.path.isdir('data'): # if data directory does not exists then extract
        extract('data.zip', project_root)
        
    
    # Load data
    seismic_marmousi = np.load(join('data','marmousi_synthetic_seismic.npy')).squeeze()
    seismic_seam = np.load(join('data','poststack_seam_seismic.npy')).squeeze()[:, 50:]
    seismic_seam = seismic_seam[::2, :]
    
    # Load targets and standardize data
    model_marmousi = np.load(join('data', 'marmousi_Ip_model.npy')).squeeze()[::5, ::4]
    model_seam = np.load(join('data','seam_elastic_model.npy'))[::3,:,::2][:, :, 50:]
    model_seam = model_seam[:,0,:] * model_seam[:,2,:]
    
    # standardize
    seismic_marmousi, model_marmousi = standardize(seismic_marmousi, model_marmousi, no_wells_marmousi)
    seismic_seam, model_seam = standardize(seismic_seam, model_seam, no_wells_seam)
    
    return seismic_marmousi, seismic_seam, model_marmousi, model_seam


def train(**kwargs):
    """Function trains 2-D TCN as specified in the paper"""
    
    # obtain data
    seismic_marmousi, seismic_seam, model_marmousi, model_seam = preprocess(kwargs['no_wells_marmousi'],\
                                                                            kwargs['no_wells_seam'])
    
    # specify width of seismic image samples around each pseudolog
    width = 7
    offset = int(width/2)
    
    # specify pseudolog positions for training and validation
    traces_marmousi_train = np.linspace(451, 2199, kwargs['no_wells_marmousi'], dtype=int)
    traces_seam_train = np.linspace(offset, len(model_seam)-offset-1, kwargs['no_wells_seam'], dtype=int)
    traces_seam_validation = np.linspace(offset, len(model_seam)-offset-1, 3, dtype=int)
    
    # set up dataloaders
    marmousi_dataset = SeismicDataset2D(seismic_marmousi, model_marmousi, traces_marmousi_train, width)
    marmousi_loader = DataLoader(marmousi_dataset, batch_size = 16)
    
    seam_train_dataset = SeismicDataset2D(seismic_seam, model_seam, traces_seam_train, width)
    seam_train_loader = DataLoader(seam_train_dataset, batch_size = len(seam_train_dataset))
    
    seam_val_dataset = SeismicDataset2D(seismic_seam, model_seam, traces_seam_validation, width)
    seam_val_loader = DataLoader(seam_val_dataset, batch_size = len(seam_val_dataset))
    
    
    # define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up models
    model_marmousi = Model2D(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)
    model_seam = Model2D(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)
    
    # define weight sharing value
    gamma = torch.tensor([kwargs['gamma']], requires_grad=True, dtype=torch.float, device=device)  # learnable weight for weight mismatch loss
    
    # Set up loss
    criterion = torch.nn.MSELoss()
    
    # Define Optimizer
    optimizer_marmousi = torch.optim.Adam(model_marmousi.parameters(),
                                     weight_decay=0.0001,
                                     lr=0.001)
    
    optimizer_seam = torch.optim.Adam(model_seam.parameters(),
                                     weight_decay=0.0001,
                                     lr=0.001)
    
    # start training 
    for epoch in range(kwargs['epochs']):
    
      loss1 = torch.tensor([0.0], requires_grad=True).float().cuda()
      model_marmousi.train()
      model_seam.train()
      optimizer_marmousi.zero_grad()
      optimizer_seam.zero_grad()
      
      for i, (x,y) in enumerate(marmousi_loader):  
    
        y_pred, x_hat = model_marmousi(x)
        loss1 += criterion(y_pred, y) + criterion(x_hat, x)
    
      loss1 = loss1/i  
      
      for x,y in seam_train_loader:
        y_pred, x_hat = model_seam(x)
        loss2 = criterion(y_pred, y) + criterion(x_hat, x)
    
      for x, y in seam_val_loader:
        model_seam.eval()
        y_pred, _ = model_seam(x)
        val_loss = criterion(y_pred, y)
        
      weight_mismatch_loss = 0
      for param1, param2 in zip(model_marmousi.parameters(), model_seam.parameters()):
        weight_mismatch_loss += torch.sum((param1-param2)**2)
    
      loss = loss1 + loss2 + gamma*weight_mismatch_loss  # original gamma val = 0.0001
      loss.backward()
      optimizer_marmousi.step()
      optimizer_seam.step()
      
      print('Epoch: {} | Marmousi Loss: {:0.4f} | Seam Loss: {:0.4f} | Val Loss: {:0.4f} | Mismatch Loss: {:0.4f} | Gamma: {:0.4f}\
            '.format(epoch, loss1.item(), loss2.item(), val_loss.item(), \
            weight_mismatch_loss.item(), gamma.item()))  
    
    # save trained models
    if not os.path.isdir('saved_models'):  # check if directory for saved models exists
        os.mkdir('saved_models')
        
    torch.save(model_seam.state_dict(), 'saved_models/model_seam.pth')
    torch.save(model_marmousi.state_dict(), 'saved_models/model_marmousi.pth')

def test(**kwargs):
    """Function tests the trained network on SEAM and Marmousi sections and 
    prints out the results"""
    
    # obtain data
    seismic_marmousi, seismic_seam, model_marmousi, model_seam = preprocess(kwargs['no_wells_marmousi'],\
                                                                            kwargs['no_wells_seam'])
    
    # define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # specify width of seismic image samples around each pseudolog
    width = 7
    offset = int(width/2)
    
    # specify pseudolog positions for testing 
    traces_marmousi_test = np.linspace(451, 2199, 2199-451+1, dtype=int)
    traces_seam_test = np.linspace(offset, len(model_seam)-offset-1, len(model_seam)-int(2*offset), dtype=int)
    
    marmousi_test_dataset = SeismicDataset2D(seismic_marmousi, model_marmousi, traces_marmousi_test, width)
    marmousi_test_loader = DataLoader(marmousi_test_dataset, batch_size = 16)
    
    seam_test_dataset = SeismicDataset2D(seismic_seam, model_seam, traces_seam_test, width)
    seam_test_loader = DataLoader(seam_test_dataset, batch_size = 8)
    
    # load saved models
    if not os.path.isdir('saved_models'):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'saved_models')
        
    # set up models
    model_marmousi = Model2D(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)
    model_seam = Model2D(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)

    model_seam.load_state_dict(torch.load('saved_models/model_seam.pth'))
    model_marmousi.load_state_dict(torch.load('saved_models/model_marmousi.pth'))
    
    # infer on SEAM
    print("Inferring on SEAM...")
    x, y = seam_test_dataset[0]  # get a sample
    AI_pred = torch.zeros((len(seam_test_dataset), y.shape[-1])).float().to(device)
    AI_act = torch.zeros((len(seam_test_dataset), y.shape[-1])).float().to(device)
    
    mem = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(seam_test_loader):
          model_seam.eval()
          y_pred, _ = model_seam(x)
          AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
          AI_act[mem:mem+len(x)] = y.squeeze().data
          mem += len(x)
          del x, y, y_pred
    
    vmin, vmax = AI_act.min(), AI_act.max()

    AI_pred = AI_pred.detach().cpu().numpy()
    AI_act = AI_act.detach().cpu().numpy()
    print('r^2 score: {:0.4f}'.format(r2_score(AI_act.T, AI_pred.T)))
    print('MSE: {:0.4f}'.format(np.sum((AI_pred-AI_act).ravel()**2)/AI_pred.size))
    print('MAE: {:0.4f}'.format(np.sum(np.abs(AI_pred - AI_act)/AI_pred.size)))
    print('MedAE: {:0.4f}'.format(np.median(np.abs(AI_pred - AI_act))))
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,12))
    ax1.imshow(AI_pred.T, vmin=vmin, vmax=vmax, extent=(0,35000,15000,0))
    ax1.set_aspect(35/30)
    ax1.set_xlabel('Distance Eastimg (m)')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Predicted')
    ax2.imshow(AI_act.T, vmin=vmin, vmax=vmax, extent=(0,35000,15000,0))
    ax2.set_aspect(35/30)
    ax2.set_xlabel('Distance Eastimg (m)')
    ax2.set_ylabel('Depth (m)')
    ax2.set_title('Ground-Truth')
    plt.show()

    
    # infer on marmousi 2
    print('\nInferring on Marmousi...')
    x, y = marmousi_test_dataset[0]  # get a sample
    AI_pred = torch.zeros((len(marmousi_test_dataset), y.shape[-1])).float().to(device)
    AI_act = torch.zeros((len(marmousi_test_dataset), y.shape[-1])).float().to(device)
    
    mem = 0
    with torch.no_grad():
        for i, (x,y) in enumerate(marmousi_test_loader):
          model_marmousi.eval()
          y_pred, _ = model_marmousi(x)
          AI_pred[mem:mem+len(x)] = y_pred.squeeze().data
          AI_act[mem:mem+len(x)] = y.squeeze().data
          mem += len(x)
          del x, y, y_pred
          
        vmin, vmax = AI_act.min(), AI_act.max()

    AI_pred = AI_pred.detach().cpu().numpy()
    AI_act = AI_act.detach().cpu().numpy()
    
    print('r^2 score: {:0.4f}'.format(r2_score(AI_act.T, AI_pred.T)))
    print('MSE: {:0.4f}'.format(np.sum((AI_pred-AI_act).ravel()**2)/AI_pred.size))
    print('MAE: {:0.4f}'.format(np.sum(np.abs(AI_pred - AI_act)/AI_pred.size)))
    print('MedAE: {:0.4f}'.format(np.median(np.abs(AI_pred - AI_act))))
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,12))
    ax1.imshow(AI_pred.T, vmin=vmin, vmax=vmax)
    ax1.set_title('Predicted')
    ax2.imshow(AI_act.T, vmin=vmin, vmax=vmax)
    ax2.set_title('Ground-truth')
    plt.show()
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparams')
    
    parser.add_argument('--epochs', nargs='?', type=int, default=900,
                        help='Number of epochs. Default = 1000')
    parser.add_argument('--no_wells_marmousi', nargs='?', type=int, default=50,
                        help='Number of sampled pseudologs for marmousi. Default = 50.')
    parser.add_argument('--no_wells_seam', nargs='?', type=int, default=12,
                        help='Number of sampled pseudologs for marmousi. Default = 12.')
    parser.add_argument('--gamma', nargs='?', type=float, default=1e-4,
                        help='Gamma value for soft sharing loss. Default = 1e-4')

    args = parser.parse_args()
    
    train(no_wells_marmousi=args.no_wells_marmousi, no_wells_seam=args.no_wells_seam, 
          epochs=args.epochs, gamma=args.gamma)
    
    test(no_wells_marmousi=args.no_wells_marmousi, no_wells_seam=args.no_wells_seam)