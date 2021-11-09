import numpy as np
import pickle
from torchvision import datasets

datanames = ['PCA_MNIST_5', 'PCA_MNIST_10', 'PCA_MNIST_20', 'PCA_MNIST_30', 'PCA_FMNIST_5', 'PCA_FMNIST_10', 'PCA_FMNIST_20', 'PCA_FMNIST_30',
            'AE_MNIST_5', 'AE_MNIST_10', 'AE_MNIST_20', 'AE_MNIST_30', 'AE_FMNIST_5', 'AE_FMNIST_10', 'AE_FMNIST_20', 'AE_FMNIST_30',
            'mnist_784', 'fmnist_784']

class MNIST_DataLoader:
    def __init__(self,name, path='../datasets/real_datasets/'):
        self.name = name
        self.path = path
    
    def load(self):
        name = self.name
        if name=='PCA_MNIST_5':
            X = pickle.load(open(self.path + '/embedding/PCA_MNIST_X_5.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_MNIST_Hidden_5.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_MNIST_Y_5.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_MNIST_10':
            X = pickle.load(open(self.path + '/embedding/PCA_MNIST_X_10.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_MNIST_Hidden_10.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_MNIST_Y_10.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_MNIST_20':
            X = pickle.load(open(self.path + '/embedding/PCA_MNIST_X_20.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_MNIST_Hidden_20.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_MNIST_Y_20.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_MNIST_30':
            X = pickle.load(open(self.path + '/embedding/PCA_MNIST_X_30.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_MNIST_Hidden_30.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_MNIST_Y_30.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_FMNIST_5':
            X = pickle.load(open(self.path + '/embedding/PCA_FMNIST_X_5.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Hidden_5.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Y_5.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_FMNIST_10':
            X = pickle.load(open(self.path + '/embedding/PCA_FMNIST_X_10.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Hidden_10.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Y_10.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_FMNIST_20':
            X = pickle.load(open(self.path + '/embedding/PCA_FMNIST_X_20.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Hidden_20.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Y_20.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='PCA_FMNIST_30':
            X = pickle.load(open(self.path + '/embedding/PCA_FMNIST_X_30.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Hidden_30.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/PCA_FMNIST_Y_30.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_MNIST_5':
            X = pickle.load(open(self.path + '/embedding/AE_MNIST_X_5.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_MNIST_Hidden_5.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_MNIST_Y_5.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_MNIST_10':
            X = pickle.load(open(self.path + '/embedding/AE_MNIST_X_10.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_MNIST_Hidden_10.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_MNIST_Y_10.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_MNIST_20':
            X = pickle.load(open(self.path + '/embedding/AE_MNIST_X_20.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_MNIST_Hidden_20.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_MNIST_Y_20.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_MNIST_30':
            X = pickle.load(open(self.path + '/embedding/AE_MNIST_X_30.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_MNIST_Hidden_30.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_MNIST_Y_30.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_FMNIST_5':
            X = pickle.load(open(self.path + '/embedding/AE_FMNIST_X_5.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_FMNIST_Hidden_5.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_FMNIST_Y_5.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_FMNIST_10':
            X = pickle.load(open(self.path + '/embedding/AE_FMNIST_X_10.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_FMNIST_Hidden_10.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_FMNIST_Y_10.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_FMNIST_20':
            X = pickle.load(open(self.path + '/embedding/AE_FMNIST_X_20.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_FMNIST_Hidden_20.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_FMNIST_Y_20.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='AE_FMNIST_30':
            X = pickle.load(open(self.path + '/embedding/AE_FMNIST_X_30.pkl','rb'))
            Hidden = pickle.load(open(self.path + '/embedding/AE_FMNIST_Hidden_30.pkl','rb'))
            Y = pickle.load(open(self.path + '/embedding/AE_FMNIST_Y_30.pkl','rb'))
            return X,np.ascontiguousarray(Hidden),Y
        
        if name=='mnist_784':
            dataset2 = datasets.MNIST(self.path, train=True, download=True)
            N=dataset2.test_data.shape[0]
            X=dataset2.test_data.numpy().reshape(N,784)/255
            Y_true=dataset2.test_labels.numpy()
            return X,Y_true
        
        if name=='fmnist_784':
            dataset2 = datasets.FashionMNIST(self.path, train=True, download=True)
            N=dataset2.test_data.shape[0]
            X=dataset2.test_data.numpy().reshape(N,784)/255
            Y_true=dataset2.test_labels.numpy()
            return X,Y_true