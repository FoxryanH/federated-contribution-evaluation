import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size=50000,
        embed_dim=128,
        num_classes=2,
        kernel_sizes=(3, 4, 5),
        num_channels=100,
        dropout=0.5,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, num_channels, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, inputs):
        embedded = self.embedding(inputs).transpose(1, 2)
        conv_features = [F.relu(conv(embedded)).amax(dim=2) for conv in self.convs]
        features = torch.cat(conv_features, dim=1)
        features = self.dropout(features)
        return self.fc(features)
