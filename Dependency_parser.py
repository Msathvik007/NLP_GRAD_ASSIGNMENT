from torch.utils.data import Dataset
class DependencyParsingDataset(Dataset):
    def __init__(self, file_path, pos_vocab, dep_vocab):
        self.pos_vocab = pos_vocab
        self.dep_vocab = dep_vocab
        self.data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith("#"):  # Skip empty lines and comments
                    parts = line.strip().split()
                    if len(parts) > 7:  # Ensure there are enough parts
                        word_index = int(parts[0])  # ID of the word
                        pos_tag = self.pos_vocab[parts[3]]  # Convert POS tag to index
                        head_index = int(parts[6])  # ID of the head word
                        dep_rel = self.dep_vocab[parts[7]]  # Convert dependency label to index
                        self.data.append((word_index, pos_tag, head_index, dep_rel))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      word_index, pos_index, head_index, deprel_index = self.data[idx]
      return (torch.tensor([word_index], dtype=torch.long).unsqueeze(0),  # Adding sequence length dimension
            torch.tensor([pos_index], dtype=torch.long).unsqueeze(0),
            torch.tensor([head_index], dtype=torch.long).unsqueeze(0),
            torch.tensor([deprel_index], dtype=torch.long).unsqueeze(0))

