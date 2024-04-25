import torch
import torch.optim as optim
import torch.nn as nn

# Assume these are the sizes of your vocabularies
word_vocab_size = 10000
pos_vocab_size = 50
dep_vocab_size = 45
embedding_dim = 100
hidden_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMDependencyParser(word_vocab_size, pos_vocab_size, dep_vocab_size, embedding_dim, hidden_dim)
model.to(device)

# Using Mean Squared Error Loss for head prediction and CrossEntropyLoss for dependency prediction
head_loss_function = nn.MSELoss()
dep_loss_function = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
