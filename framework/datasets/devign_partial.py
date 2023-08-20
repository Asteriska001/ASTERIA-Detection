import pandas as pd

from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join

from torch.utils.data import Dataset
#from torch_geometric.data import DataLoader

def get_ratio(dataset, ratio):
    approx_size = int(len(dataset) * ratio)
    return dataset[:approx_size]

def load(path, pickle_file, ratio=1):
    dataset = pd.read_pickle(path + pickle_file)
    dataset.info(memory_usage='deep')
    if ratio < 1:
        dataset = get_ratio(dataset, ratio)

    return dataset


def loads(data_sets_dir, ratio=1):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

    if ratio < 1:
        data_sets_files = get_ratio(data_sets_files, ratio)

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        dataset = dataset.append(load(data_sets_dir, ds_file))

    return dataset

def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    train = train_false.append(train_true)
    val = val_false.append(val_true)
    test = test_false.append(test_true)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return Devign_Partial(train), Devign_Partial(test), Devign_Partial(val)


class Devign_Partial(Dataset):
    def __init__(self, split: str, input_path="./devign_partial_data/input"):
        input_dataset = loads(input_path)
        #port dataset
        assert split in ['train', 'val', 'test']

        train_dataset,test_dataset,val_dataset = train_val_test_split(input_dataset, shuffle='True')
        
        if split == 'train':
            self.dataset = train_dataset
        elif split == 'test':
            self.dataset = test_dataset
        else:
            self.dataset = val_dataset
        #train_loader, val_loader, test_loader = list(
        #map(lambda x: x.get_loader(context.batch_size, shuffle=context.shuffle),
        #    train_val_test_split(input_dataset, shuffle=context.shuffle)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input

    #def get_loader(self, batch_size, shuffle=True):
    #    return DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)