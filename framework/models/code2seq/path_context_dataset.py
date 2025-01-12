from os.path import exists
from typing import Dict, List

import numpy
from omegaconf import DictConfig
from torch.utils.data import Dataset

from models.code2seq.data_classes import PathContextSample, FROM_TOKEN, PATH_NODES, TO_TOKEN, ContextPart
from framework.utils.converting import strings_to_wrapped_numpy
from framework.utils.vocabulary import Vocabulary_c2s


class PathContextDataset(Dataset):

    _separator = "|"

    def __init__(self, data_file_path: str, config: DictConfig,
                 vocabulary: Vocabulary_c2s, random_context: bool):
        if not exists(data_file_path):
            raise ValueError(f"Can't find file with data: {data_file_path}")
        self._data_file_path = data_file_path
        self._hyper_parameters = config.hyper_parameters
        self._random_context = random_context
        self._line_offsets = []
        cumulative_offset = 0
        with open(self._data_file_path, "r") as data_file:
            for line in data_file:
                self._line_offsets.append(cumulative_offset)
                cumulative_offset += len(line.encode(data_file.encoding))
        self._n_samples = len(self._line_offsets)

        self._context_parts: List[ContextPart] = [
            ContextPart(FROM_TOKEN, vocabulary.token_to_id,
                        config.dataset.token),
            ContextPart(PATH_NODES, vocabulary.node_to_id,
                        config.dataset.path),
            ContextPart(TO_TOKEN, vocabulary.token_to_id,
                        config.dataset.token),
        ]

    def __len__(self):
        return self._n_samples

    def _read_line(self, index: int) -> str:
        with open(self._data_file_path, "r") as data_file:
            data_file.seek(self._line_offsets[index])
            line = data_file.readline().strip()
        return line

    @staticmethod
    def _split_context(context: str) -> Dict[str, str]:
        from_token, path_nodes, to_token = context.split(",")
        return {
            FROM_TOKEN: from_token,
            PATH_NODES: path_nodes,
            TO_TOKEN: to_token,
        }

    def __getitem__(self, index) -> PathContextSample:
        raw_sample = self._read_line(index)
        str_label, *str_contexts = raw_sample.split()

        # choose random paths
        n_contexts = min(len(str_contexts), self._hyper_parameters.max_context)
        context_indexes = numpy.arange(n_contexts)
        if self._random_context:
            numpy.random.shuffle(context_indexes)

        # convert string label to wrapped numpy array
        wrapped_label = int(str_label)

        # convert each context to list of ints and then wrap into numpy array
        splitted_contexts = [
            self._split_context(str_contexts[i]) for i in context_indexes
        ]
        contexts = {}
        for _cp in self._context_parts:
            str_values = [_sc[_cp.name] for _sc in splitted_contexts]
            contexts[_cp.name] = strings_to_wrapped_numpy(
                str_values, _cp.to_id, _cp.parameters.is_splitted,
                _cp.parameters.max_parts, _cp.parameters.is_wrapped)

        return PathContextSample(contexts=contexts,
                                 label=wrapped_label,
                                 n_contexts=n_contexts)
    def get_n_samples(self):
        return self._n_samples