from typing import Tuple, Dict, List, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from omegaconf import DictConfig
from models.sysevr.data_classes import SYSBatch
from torch.optim import Adam, Optimizer, SGD, Adamax, RMSprop
import pprint as pp
import torch.nn as nn
import numpy
import torch.nn.functional as F
from utils.training import cut_sys_encoded_contexts

class SYS_BGRU(nn.Module):
    _negative_value = -numpy.inf
    _activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "lkrelu": nn.LeakyReLU(0.3)
    }
    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self._config = config
        self.save_hyperparameters()
        self.pre_truth = []
        self.ground_truth = []
        self.pre_union_ground_truth = []
        self.init_layers()

    def _get_activation(self, activation_name: str) -> torch.nn.Module:
        if activation_name in self._activations:
            return self._activations[activation_name]
        raise KeyError(f"Activation {activation_name} is not supported")

    def _get_optimizer(self, optimizer_name: str) -> torch.nn.Module:
        if optimizer_name in self._optimizers:
            return self._optimizers[optimizer_name]
        raise KeyError(f"Optimizer {optimizer_name} is not supported")

    def init_layers(self):
        self.dropout_rnn = nn.Dropout(self._config.encoder.rnn_dropout)
        # BLSTM layer
        self.blstm_layer = nn.LSTM(
            input_size=self._config.encoder.embedding_size,
            hidden_size=self._config.encoder.rnn_size,
            num_layers=self._config.encoder.rnn_num_layers,
            bidirectional=self._config.encoder.use_bi_rnn,
            dropout=self._config.encoder.rnn_dropout
            if self._config.encoder.rnn_num_layers > 1 else 0,
            batch_first=True)
        # BGUR layer
        self.bgru_layer = nn.GRU(
            input_size=self._config.encoder.embedding_size,
            hidden_size=self._config.encoder.rnn_size,
            num_layers=self._config.encoder.rnn_num_layers,
            bidirectional=self._config.encoder.use_bi_rnn,
            dropout=self._config.encoder.rnn_dropout
            if self._config.encoder.rnn_num_layers > 1 else 0,
            batch_first=True)
        # layer for attention
        # self.att_layer = LuongAttention(self.hidden_size)
        # self.att_layer = LocalAttention(self._config.encoder.hidden_size)

        # MLP
        layers = [
            nn.Linear(self._config.encoder.rnn_size,
                      self._config.classifier.hidden_size),
            self._get_activation(self._config.classifier.activation),
            nn.Dropout(0.5)
        ]
        if self._config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({self._config.classifier.n_hidden_layers})"
            )
        for _ in range(self._config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(self._config.classifier.hidden_size,
                          self._config.classifier.hidden_size),
                self._get_activation(self._config.classifier.activation),
                nn.Dropout(0.5)
            ]
        self.hidden_layers = nn.Sequential(*layers)

        self.out_layer = nn.Linear(self._config.classifier.hidden_size, 2)

    def forward(self, gadgets: torch.Tensor,
                words_per_label: List[int]) -> torch.Tensor:
        """
        :param gadgets: (total word length, input size)
        :param words_per_label: word length for each label
        :return: (batch size, output size)
        """

        batch_size = len(words_per_label)
        # x: (batch size, seq len, input size), masks: (batch size, sen_len)
        x, masks = cut_sys_encoded_contexts(
            gadgets, words_per_label, self._config.hyper_parameters.seq_len,
            self._negative_value)

        lengths_per_label = [
            min(self._config.hyper_parameters.seq_len, word_per_label.item())
            for word_per_label in words_per_label
        ]
        # accelerating packing
        with torch.no_grad():
            first_pad_pos = torch.from_numpy(numpy.array(lengths_per_label))
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)
            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        x = x[sort_indices]
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              sorted_path_lengths,
                                              batch_first=True)

        # bgru_out: (batch size, seq len, 2 * hidden size), h_n: (num layers * 2, batch size, hidden size)
        bgru_out, h_n = self.bgru_layer(x)
        # (batch size, num layers * 2, hidden size)
        h_n = h_n.permute(1, 0, 2)

        # atten_out = self.att_layer(blstm_out, h_n, masks) # (batch size, hidden size)
        # atten_out = self.att_layer(blstm_out,
        #                            masks)  # (batch size, hidden size)
        atten_out = torch.sum(h_n, dim=1)  # (batch size, hidden size)
        atten_out = self.dropout_rnn(atten_out)[reverse_sort_indices]
        out = self.out_layer(
            self.hidden_layers(atten_out))  # (batch size, output size)
        # out_prob = F.softmax(out.view(batch_size, -1)) # (batch size, output size)
        out_prob = torch.log_softmax(out.view(batch_size, -1),
                                     dim=1)  # (batch size, output size)

        return out_prob
""" 
    def training_step(self, batch: SYSBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch.gadgets, batch.tokens_per_label)
        # loss = F.cross_entropy(logits, batch.labels)
        loss = F.nll_loss(logits, batch.labels)
        log: Dict[str, Union[float, torch.Tensor]] = {"train/loss": loss}

        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_matric = statistic.calculate_metrics(group="train")
            log.update(batch_matric)
            self.log_dict(log)
            self.log("f1",
                     batch_matric["train/f1"],
                     prog_bar=True,
                     logger=False)

        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: SYSBatch, batch_idx: int) -> Dict:
        # (batch size, output size)
        logits = self(batch.gadgets, batch.tokens_per_label)

        loss = F.nll_loss(logits, batch.labels)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )

        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: SYSBatch, batch_idx: int) -> Dict:
        
        logits = self(batch.gadgets, batch.tokens_per_label)

        loss = F.nll_loss(logits, batch.labels)
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            for pred, label, metric in zip(preds, batch.labels, batch.metric_infos):
                if pred == 1 or label == 1:
                    self.pre_union_ground_truth.append({'pred':pred, 'label':label, 'metric':metric})
                    if pred == 1:
                        self.pre_truth.append({'pred':pred, 'label':label, 'metric':metric})
                    if label == 1:
                        self.ground_truth.append({'pred':pred, 'label':label, 'metric':metric})
                    

        return {"loss": loss, "statistic": statistic}
     """
