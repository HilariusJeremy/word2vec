import collections
import numpy as np
import nltk
nltk.download('brown')
from nltk.corpus import brown
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn 
from torch import optim
import os
from tqdm.auto import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

CONTEXT_SIZE = 2
K = 1
EMBEDDING_DIM = 300
BATCH_SIZE = 8192
LR = 0.001
EPOCHS = 20

def get_context_windows(corpus_indices, context_size):
    context_pairs = []
    for i, word_idx in enumerate(corpus_indices):
        for j in range(max(0, i - context_size), min(len(corpus_indices), i + context_size + 1)):
            if i != j:
                context_pairs.append((word_idx, corpus_indices[j]))
    return context_pairs

def generate_examples(context_pairs, unigram_dist, num_negatives):
    positive_examples = []
    negative_examples = []
    labels = []
    
    for target, context in tqdm(context_pairs, desc="Generating examples"):
        positive_examples.append((target, context))
        labels.append(1)
        
        for _ in range(num_negatives):
            negative_context = np.random.choice(vocab_size, p=unigram_dist)
            negative_examples.append((target, negative_context))
            labels.append(0)
    
    examples = positive_examples + negative_examples
    return examples, labels

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # Define x to be have a shape of [batch_size, 2]
    def forward(self, x):
        embed_input = self.input_embedding(x.T[0])
        embed_output = self.output_embedding(x.T[1])
        return (embed_input*embed_output).sum(dim=1)

class CustomTextDataset(Dataset):

    def __init__(self, context_size, num_negatives, pkl_path='./examples_labels.pkl'):
        self.pkl_path = pkl_path
        self.context_size = context_size
        self.num_negatives = num_negatives
        
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                examples, labels = pickle.load(f)
        else:
            corpus_indices = [word_to_idx[word] for word in filtered_corpus] 
            context_pairs = get_context_windows(corpus_indices, context_size)
            word_counts_filtered = collections.Counter(corpus_indices)
            total_count = sum(word_counts_filtered.values())
            unigram_dist = np.array([word_counts_filtered[i] for i in range(vocab_size)])
            unigram_dist = unigram_dist / total_count
            unigram_dist = unigram_dist ** (3/4)
            unigram_dist = unigram_dist / np.sum(unigram_dist)
            examples, labels = generate_examples(context_pairs, unigram_dist, num_negatives)
            with open(pkl_path, 'wb') as f:
                pickle.dump((examples, labels), f)
        
        self.examples_tensor = torch.tensor(examples)
        self.labels_tensor = torch.tensor(labels)

    def __len__(self):
        return len(self.labels_tensor)
    
    def __getitem__(self, idx):
        return self.examples_tensor[idx], self.labels_tensor[idx]
    

def loop_fn(dataset, dataloader, model, criterion, optimizer, device):
    model.train()
    cost = 0
    for feature, target in tqdm(dataloader):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output.float(), target.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
    cost = cost / len(dataset)
    return cost

if __name__ == "__main__":
    sentences = brown.sents()
    corpus = [word.lower() for sentence in sentences for word in sentence]

    # Calculate word frequencies
    word_counts = collections.Counter(corpus)

    # Remove words with frequency less than 5
    filtered_corpus = [word for word in corpus if word_counts[word] >= 5]

    # Create a vocabulary and word-to-index mapping
    vocab = set(filtered_corpus)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    vocab_size = len(vocab)

    train_set = CustomTextDataset(pkl_path='./examples_labels.pkl', context_size=CONTEXT_SIZE, num_negatives=K)
    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    epochs = EPOCHS
    train_cost = []
    for i in range(epochs):
        cost = loop_fn(train_set, trainloader, model, criterion, optimizer, device)
        train_cost.append(cost)
    
        print(f"\rEpoch: {i+1}/{epochs} | train_cost: {train_cost[-1]: 4f} | ", end=" ")
    
    os.makedirs('model', exist_ok=True)

    # Save Model
    torch.save(model.state_dict(), './model/weights.pth')
    # Load Model
    model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
    weights = torch.load('./model/weights.pth', map_location='cpu')
    model.load_state_dict(weights)
    model = model.to(device)



