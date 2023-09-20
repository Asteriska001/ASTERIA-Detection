import torch.nn as nn

class Code2SeqAttn(nn.Module):
    def __init__(self, config, vocabulary):
        super(Code2SeqAttn, self).__init__()
        
        self.encoder = PathEncoder(
            config.encoder,
            config.classifier.classifier_input_size,
            len(vocabulary.token_to_id),
            vocabulary.token_to_id[PAD],
            len(vocabulary.node_to_id),
            vocabulary.node_to_id[PAD],
        )
        self.num_classes = 2
        self.classifier = PathClassifier(config.classifier, self.num_classes)

    def forward(self, samples, paths_for_label):
        return self.classifier(self.encoder(samples), paths_for_label)
