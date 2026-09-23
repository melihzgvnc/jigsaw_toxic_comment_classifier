import torch
import torch.nn as nn

class JigsawClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_dim*2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, seq_len):
        #print(x.min().item(), x.max().item(), self.embedding.num_embeddings)
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.bilstm(x)

        # unpack the output
        output, lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # mask out padded timesteps
        batch_size, max_len, _ = output.shape
        mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        output = output.masked_fill(mask.unsqueeze(2), float('-inf'))

        # max-pool over timesteps
        pooled, _ = torch.max(output, dim=1)

        pooled = self.dropout(pooled)
        logits = self.linear(pooled)
        return logits

# class JigsawClassifier(nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=6):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
#         self.bilstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
#         self.linear = nn.Linear(hidden_dim*2, num_classes)

#     def forward(self, x, seq_len):
#         x = self.embedding(x)
#         x = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
#         packed_output, (h_n, c_n) = self.bilstm(x)
#         final_hidden_state = torch.cat([h_n[0], h_n[1]], dim=1)
#         logits = self.linear(final_hidden_state)
#         return logits
    

