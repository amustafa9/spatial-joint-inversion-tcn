# Set up dataloaders

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np


class SeismicDataset2D(Dataset):
    """Dataset class for loading 2D seismic images for 2-D TCN"""
    def __init__(self, seismic, model, trace_indices, width):
        self.seismic = seismic
        self.model = model
        self.trace_indices = trace_indices
        self.width = width

        assert min(trace_indices) - int(width/2) >= 0 and max(trace_indices) + int(width/2) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"

    def __getitem__(self, index):
        offset = int(self.width/2)
        trace_index = self.trace_indices[index]
        x = torch.tensor(self.seismic[trace_index-offset:trace_index+offset+1].T[np.newaxis, :, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = torch.tensor(self.model[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return x, y
  
    def __len__(self):
        return len(self.trace_indices)  
    
    
class SeismicDataset1D(Dataset):
      """Dataset class for 1D TCN"""
      def __init__(self, seismic, model, trace_indices):
        self.seismic = seismic
        self.model = model
        self.trace_indices = trace_indices
    
    
        assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
    
      def __getitem__(self, index):
        trace_index = self.trace_indices[index]
        x = torch.tensor(self.seismic[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = torch.tensor(self.model[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return x, y
      
      def __len__(self):
        return len(self.trace_indices)     
    
    
class UnsupervisedSeismicDataset(Dataset):
      """Dataset class for unsupervised loss in Motaz """
      def __init__(self, seismic, trace_indices):
        self.seismic = seismic
        self.trace_indices = trace_indices
       
        assert min(trace_indices) >= 0 and max(trace_indices) + 1 <= len(seismic),"Seismic patch accessing traces out of geometry of the data!"
    
      def __getitem__(self, index):
        trace_index = self.trace_indices[index]
        x = torch.tensor(self.seismic[trace_index][np.newaxis, :], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return x
      
      def __len__(self):
        return len(self.trace_indices)     