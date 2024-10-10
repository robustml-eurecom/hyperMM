import torch
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import os

from datetime import datetime

class PairedDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None, weights=None):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase

        self.folder_path_PET = self.root_dir + "/preprocess_PET/{}".format(self.phase) 
        self.folder_path_MRI =  self.root_dir + "/preprocess_MRI/{}".format(self.phase)  ##### CHANGE BACK HERE

        self.records_PET = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_PET)]))
        self.records_MRI = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_MRI)]))
        self.transform = transform
    
    def __len__(self):
        return len(self.records_MRI)

    def __getitem__(self, index):
        #MRI Item
        tuple_data = []

        try : 
            slice_list_MRI = [j for j in os.listdir(self.folder_path_MRI) if j.startswith(self.records_MRI[index])]
            slice_list_MRI.sort()
            
            ref_MRI = np.load(os.path.join(self.folder_path_MRI, slice_list_MRI[0]), allow_pickle=True).item()
            dims_MRI = (len(slice_list_MRI),) + (3, 224, 224) #Because the VGG19 do a center crop of the images at 224

            array_volume_MRI = np.zeros(dims_MRI)

            for s in slice_list_MRI:
                array = np.load(os.path.join(self.folder_path_MRI, s), allow_pickle=True).item()["data"]
                array_slice = np.tile(array[:, :], [1, 1, 3])
                array_slice = Image.fromarray((array_slice*255).astype(np.uint8))
                
                if self.transform:
                    array_slice = self.transform(array_slice)
                
                array_volume_MRI[slice_list_MRI.index(s), :, :, :] = array_slice

            array_volume_MRI = torch.FloatTensor(array_volume_MRI)

            #Label 
            label_str = ref_MRI["label"]
            label = 0 if label_str == "CN" else 1
            label = torch.FloatTensor([label])

            #Modality
            modality_MRI = torch.LongTensor([1])

            tuple_data.append([array_volume_MRI, modality_MRI])

        except Exception as e :
            print("No MRI")
        
        #PET item
        try :
            slice_list_PET = [j for j in os.listdir(self.folder_path_PET) if j.startswith(self.records_PET[index])]
            slice_list_PET.sort()

            ref_PET = np.load(os.path.join(self.folder_path_PET, slice_list_PET[0]), allow_pickle=True).item()
            #ref_data = ref_PET["data"]
            dims_PET = (len(slice_list_PET),) + (3, 224, 224) #Because the VGG19 do a center crop of the images at 224

            array_volume_PET = np.zeros(dims_PET)

            for s in slice_list_PET:
                array = np.load(os.path.join(self.folder_path_PET, s), allow_pickle=True).item()["data"]
                array_slice = np.tile(array[:, :], [1, 1, 3])
                array_slice = Image.fromarray((array_slice*255).astype(np.uint8))
                
                if self.transform:
                    array_slice = self.transform(array_slice)
                
                array_volume_PET[slice_list_PET.index(s), :, :, :] = array_slice

            array_volume_PET = torch.FloatTensor(array_volume_PET)
            
            #Labels
            label_str = ref_PET["label"]
            label = 0 if label_str == "CN" else 1
            label = torch.FloatTensor([label])
            
            #Modality
            modality_PET = torch.LongTensor([0])

            tuple_data.append([array_volume_PET, modality_PET])
        
        except Exception as e:
            print("No PET")

        tuple_data.append(label)

        return tuple_data



class IncompletePairedDataset(Dataset):
    def __init__(self, root_dir, phase="train", transform=None, missing_pattern = None, missing_list = None):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.missing_pattern = missing_pattern
        self.missing_list = missing_list

        self.folder_path_PET = self.root_dir + "/preprocess_PET/{}".format(self.phase) 
        self.folder_path_MRI =  self.root_dir + "/preprocess_MRI/{}".format(self.phase)

        if not self.missing_pattern : 
            self.records_PET = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_PET)]))
            self.records_MRI = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_MRI)]))
        else :
            if self.missing_pattern == "MRI":
                MRIs = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_MRI)]))
                self.records_MRI = [i for i in MRIs if i[:10] not in self.missing_list]
                self.records_PET = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_PET)]))
            
            elif self.missing_pattern == "PET":
                PETs = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_PET)]))
                self.records_PET = [i for i in PETs if i[:10] not in self.missing_list]
                self.records_MRI = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_MRI)]))
            
            elif self.missing_pattern == "both":
                MRIs = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_MRI)]))
                PETs = list(np.unique([i.rsplit("_",1)[0] for i in os.listdir(self.folder_path_PET)]))
                self.records_MRI = [i for i in MRIs if i[:10] not in self.missing_list[0]]
                self.records_PET = [i for i in PETs if i[:10] not in self.missing_list[1]]
       
        self.transform = transform
    
    def __len__(self):
        return max(len(self.records_MRI), len(self.records_PET))

    def __getitem__(self, index):
        #MRI Item
        tuple_data = []

        try : 
            slice_list_MRI = [j for j in os.listdir(self.folder_path_MRI) if j.startswith(self.records_MRI[index])]
            slice_list_MRI.sort()
            
            ref_MRI = np.load(os.path.join(self.folder_path_MRI, slice_list_MRI[0]), allow_pickle=True).item()
            dims_MRI = (len(slice_list_MRI),) + (3, 224, 224) #Because the VGG19 do a center crop of the images at 224

            array_volume_MRI = np.zeros(dims_MRI)

            for s in slice_list_MRI:
                array = np.load(os.path.join(self.folder_path_MRI, s), allow_pickle=True).item()["data"]
                array_slice = np.tile(array[:, :], [1, 1, 3])
                array_slice = Image.fromarray((array_slice*255).astype(np.uint8))
                
                if self.transform:
                    array_slice = self.transform(array_slice)
                
                array_volume_MRI[slice_list_MRI.index(s), :, :, :] = array_slice

            array_volume_MRI = torch.FloatTensor(array_volume_MRI)

            #Label 
            label_str = ref_MRI["label"]
            label = 0 if label_str == "CN" else 1
            label = torch.FloatTensor([label])

            #Modality
            modality_MRI = torch.LongTensor([1])

            tuple_data.append([array_volume_MRI, modality_MRI])

        except Exception:
            pass
            #print("No MRI")
        

        #PET item
        try :
            slice_list_PET = [j for j in os.listdir(self.folder_path_PET) if j.startswith(self.records_PET[index])]
            slice_list_PET.sort()

            ref_PET = np.load(os.path.join(self.folder_path_PET, slice_list_PET[0]), allow_pickle=True).item()
            #ref_data = ref_PET["data"]
            dims_PET = (len(slice_list_PET),) + (3, 224, 224) #Because the VGG19 do a center crop of the images at 224

            array_volume_PET = np.zeros(dims_PET)

            for s in slice_list_PET:
                array = np.load(os.path.join(self.folder_path_PET, s), allow_pickle=True).item()["data"]
                array_slice = np.tile(array[:, :], [1, 1, 3])
                array_slice = Image.fromarray((array_slice*255).astype(np.uint8))
                
                if self.transform:
                    array_slice = self.transform(array_slice)
                
                array_volume_PET[slice_list_PET.index(s), :, :, :] = array_slice

            array_volume_PET = torch.FloatTensor(array_volume_PET)
            
            #Labels
            label_str = ref_PET["label"]
            label = 0 if label_str == "CN" else 1
            label = torch.FloatTensor([label])
            
            #Modality
            modality_PET = torch.LongTensor([0])

            tuple_data.append([array_volume_PET, modality_PET])
        
        except Exception:
            pass
        #print("No PET")

        tuple_data.append(label)

        return tuple_data