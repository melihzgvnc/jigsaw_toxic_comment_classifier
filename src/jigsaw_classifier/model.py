import torch
import torch.nn as nn

class JigsawClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x, seq_len):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.bilstm(x)
        final_hidden_state = torch.cat([h_n[0], h_n[1]], dim=1)
        logits = self.linear(final_hidden_state)
        return logits