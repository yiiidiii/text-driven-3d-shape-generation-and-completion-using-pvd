from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from datasets.dataset import CLIPEmbeddingDataset
from torch.utils.data import Dataset

class ShapeNetText(Dataset):
    def __init__(self, dataroot_clip, csv_file_clip, categories, synsnet_path_clip, shape_dataset):
        super().__init__()
        
        self.csv_file = csv_file_clip
        self.synsnet_path_clip = synsnet_path_clip
        
        self.shape_dataset = shape_dataset
        self.clip_dataset = CLIPEmbeddingDataset(dataroot_clip, csv_file_clip, categories)
        
        self.all_points_mean = self.shape_dataset.all_points_mean
        self.all_points_std =  self.shape_dataset.all_points_std
        
        self.shape_text_data = []
        for shape_sample in self.shape_dataset:
            id = shape_sample['name'].split('/')[-1]
            
            if id in self.clip_dataset.data_dict:
                shape_text_sample = shape_sample
                shape_text_sample['text_embedding'] = self.clip_dataset.data_dict[id]['text_embedding']
                self.shape_text_data.append(shape_text_sample)
            # shape_text_sample = shape_sample
            # shape_text_sample['text_embedding'] = self.clip_dataset.data[0]['text_embedding']
            # self.shape_text_data.append(shape_text_sample)
            
    def __len__(self):
        return len(self.shape_text_data)

    def __getitem__(self, index):
        return self.shape_text_data[index]