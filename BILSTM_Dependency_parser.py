import torch.nn as nn

class BiLSTMDependencyParser(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, dep_vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.word_embeddings = nn.Embedding(word_vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, bidirectional=True)
        self.head_predictor = nn.Linear(hidden_dim * 2, 1)
        self.dep_predictor = nn.Linear(hidden_dim * 2, dep_vocab_size)




    def forward(self, words, pos):
      # Generate embeddings
      word_embeds = self.word_embeddings(words)  # [batch_size, embedding_dim]
      pos_embeds = self.pos_embeddings(pos)      # [batch_size, embedding_dim]
      if word_embeds.dim() == 2:
        word_embeds = word_embeds.unsqueeze(1)  # Add sequence length dimension
      if pos_embeds.dim() == 2:
        pos_embeds = pos_embeds.unsqueeze(1)
      # Concatenate embeddings along the feature dimension (last dimension)
      try:
        embeddings = torch.cat([word_embeds, pos_embeds], dim=2)
      except Exception as e:
        print("Error during concatenation:", e)
        return None, None  # Early exit on error
      # LSTM and predictors
      lstm_out, _ = self.lstm(embeddings)
      head_logits = self.head_predictor(lstm_out.squeeze(1))
      dep_logits = self.dep_predictor(lstm_out.squeeze(1))
      return head_logits, dep_logits
