from torch.utils.data import Dataset as TorchDataset
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # return self.dataset.iloc[index].input
        data = self.dataset.iloc[index].input
        data.func = self.dataset.iloc[index].func  # Add existing 'func' attribute for CodeBERT input
        return data

    def get_loader(self, batch_size, shuffle=True):
        return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=True)
