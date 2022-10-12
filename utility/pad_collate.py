from torch.nn.utils.rnn import pad_sequence

import torch


class PadCollate:

    def __init__(self, dim=0, padding_value=0.0, channels=None):
        self.dim = dim
        self.padding_value = padding_value
        self.channels = channels

    def pad_collate(self, batch):
        # batch order: features, label, name, signer, seq_length, coords, padding
        feature_padded = {}
        for _ch in self.channels:
            feature_padded[_ch] = pad_sequence([x[0][_ch] for x in batch], batch_first=True, padding_value=0.0)

        padding_masks = torch.BoolTensor([[1] * x[4] + (feature_padded[self.channels[0]].size(1) - x[4]) * [0] for x in batch]).unsqueeze(1)

        labels = []
        names = []
        signers = []
        seq_lengths = []
        coords = []
        for b in batch:
            labels.append(b[1])
            names.append(b[2])
            signers.append(b[3])
            seq_lengths.append(b[4])
            coords.append(b[5])

        return feature_padded, \
               torch.LongTensor(labels), \
               names, \
               signers, \
               torch.LongTensor(seq_lengths), \
               torch.LongTensor(coords), \
               padding_masks

    def __call__(self, batch):
        return self.pad_collate(batch)