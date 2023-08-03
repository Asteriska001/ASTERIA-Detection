import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional


from torch.nn.utils.rnn import pad_sequence
#from torchtext.data.utils import get_tokenizer
#from torchtext.vocab import build_vocab_from_iterator

'''
class Tokenize:
    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')

    def __call__(self, input_x: str, label: Tensor) -> Tuple[Tensor, Tensor]:
        tokens = self.tokenizer(input_x)
        input_x = torch.tensor([hash(token) for token in tokens])
        return input_x, label

class VocabularyMapping:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        input_x = torch.tensor([self.vocab[token] for token in input_x])
        return input_x, label
'''
class LengthNormalization:
    def __init__(self, max_length: int = 500):
        self.max_length = max_length

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if len(input_x) < self.max_length:
            input_x = pad_sequence([input_x, torch.zeros(self.max_length - len(input_x))], batch_first=True)
        else:
            input_x = input_x[:self.max_length]
        return input_x, label

class Shuffle:
    def __init__(self, seed: int = 1234):
        self.seed = seed

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        torch.manual_seed(self.seed)
        perm = torch.randperm(input_x.nelement())
        input_x = input_x.view(-1)[perm].view(input_x.size())
        return input_x, label

class Normalize:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        input_x = (input_x - self.mean) / self.std
        return input_x, label


class PadSequence:
    def __init__(self, pad_value: float = 0.0, max_length: int = 500):
        self.pad_value = pad_value
        self.max_length = max_length

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
        if input_x.size(0) < self.max_length:
            padding = torch.full((self.max_length - input_x.size(0),), self.pad_value)
            input_x = torch.cat([input_x, padding])
        else:
            input_x = input_x[:self.max_length]

        return input_x, label


class OneHotEncode:
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

    def __call__(self, input_x: Tensor, label: int) -> Tuple[Tensor, Tensor]:
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1
        return input_x, label_tensor

class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:

        for transform in self.transforms:
            input_x, label = transform(input_x, label)

        return input_x, label

def get_preprocess(size: Union[int, Tuple[int], List[int]], preprocess_format):
    preprocess_methods = {

        "Normalize": Normalize(),
        "PadSequence": PadSequence(),
        "OneHotEncode": OneHotEncode(),
        #'Tokenize': Tokenize(),
        #'VocabularyMapping': VocabularyMapping(vocab),
        'LengthNormalization': LengthNormalization(),
        'Shuffle': Shuffle()

    }

    if preprocess_format is None:
        return Compose([])
    else:
        print("Preprocessing Compose: ")
        print(preprocess_format)
        return Compose([preprocess_methods[method] for method in preprocess_format])

if __name__ == '__main__':
    input_x = torch.randn(300)
    label_x = 1
    preprocess_format = [
        "Normalize",
        "PadSequence",
        "OneHotEncode"
    ]
    preprocess = get_preprocess(500, preprocess_format)
    #fix.preprocess
    input_x, label_x = preprocess(input_x, label_x)
    print(input_x.size(), label_x.size())
'''


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, input_x: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:

        for transform in self.transforms:
            input_x, label = transform(input_x, label)

        return input_x, label


'''