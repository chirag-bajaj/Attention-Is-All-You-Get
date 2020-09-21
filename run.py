import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, num_heads, num_dim):
        super().__init__()

        self.num_heads = num_heads
        self.num_dim = num_dim

        self.toqueries = nn.Linear(self.num_dim, self.num_heads*self.num_dim, bias=False)
        self.tokeys = nn.Linear(self.num_dim, self.num_heads*self.num_dim, bias=False)
        self.tovalues = nn.Linear(self.num_dim, self.num_heads*self.num_dim, bias=False)
        self.unify_heads = nn.Linear(self.num_heads*self.num_dim, self.num_dim)

    def forward(self, x):
        num_batch, num_words, num_dim = x.shape
        num_heads = self.num_heads

        queries = self.toqueries(x).view(num_batch, num_words, num_heads, num_dim)
        keys = self.tokeys(x).view(num_batch, num_words, num_heads, num_dim)
        values = self.tovalues(x).view(num_batch, num_words, num_heads, num_dim)

        queries = queries.transpose(1, 2).contiguous().view(num_batch*num_heads, num_words, num_dim)
        keys = keys.transpose(1, 2).contiguous().view(num_batch*num_heads, num_words, num_dim)
        values = values.transpose(1, 2).contiguous().view(num_batch*num_heads, num_words, num_dim)

        queries = queries/(num_dim**(1/4))
        keys = keys/(num_dim**(1/4))

        raw_weights = torch.bmm(queries, keys.transpose(1, 2))
        weights = torch.softmax(raw_weights, dim=2)

        out = torch.bmm(weights, values).view(num_batch, num_heads, num_words, num_dim)
        out = out.transpose(1, 2).contiguous().view(num_batch, num_words, num_heads*num_dim)
        out = self.unify_heads(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_dim = num_dim
        self.sa = SelfAttention(self.num_heads, self.num_dim)
        self.norm1 = nn.LayerNorm(self.num_dim)
        self.mlp = nn.Sequential(
                        nn.Linear(self.num_dim, 4*self.num_dim),
                        nn.ReLU(),
                        nn.Linear(4*self.num_dim, self.num_dim)
                        )
        self.norm2 = nn.LayerNorm(self.num_dim)

    def forward(self, x):
        out1 = self.sa(x)
        out2 = self.norm1(out1+x)
        out3 = self.mlp(out2)
        final = self.norm2(out3+out2)
        return final

class Embedding(nn.Module):
    def __init__(self, embed, num_dim, device, max_len=10000):
        super().__init__()
        self.embed = embed
        self.device = device
        pe = torch.zeros(max_len, num_dim)
        for pos in range(max_len):
            for i in range(0, num_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/num_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/num_dim)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        num_words = x.shape[1]
        word_embedding = self.embed(x).to(self.device)
        postional_encoding = self.pe[ :, :num_words]
        postional_encoding.requires_grad = False
        postional_encoding = postional_encoding.to(self.device)
        return word_embedding + postional_encoding

class ClassificationTransformer(nn.Module):
    def __init__(self, embed, device, num_blocks=1, num_dim=50, num_heads=8, num_classes=2):
        super().__init__()
        self.embbeding = Embedding(embed, num_dim, device)
        blocks = []
        for i in range(num_blocks):
            blocks.append(Transformer(num_dim, num_heads))
        self.transformer_blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(num_dim, num_classes)
    def forward(self, x):
        #print("Before embedding", torch.cuda.memory_allocated())
        embedded_input = self.embbeding(x)
        #print("After Embedding", torch.cuda.memory_allocated())
        #print(embedded_input.get_device())
        output = self.transformer_blocks(embedded_input)
        #print("After Blocks", torch.cuda.memory_allocated())
        output = self.out(output.mean(dim=1))
        return F.log_softmax(output, dim=1)

import spacy
spacy_en = spacy.load('en')
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

from torchtext.data import Field, TabularDataset, BucketIterator
review = Field(sequential=True, tokenize=tokenize, use_vocab=True, lower=True, batch_first=True, fix_length=1024)
sentiment = Field(sequential=False, use_vocab=False)

fields = {"review": ("review", review), "sentiment": ("sentiment", sentiment)}
train_data, test_data = TabularDataset.splits(
    path='/home/chirag_17bit012/Attention-Is-All-You-Get/data',
    format='csv',
    train='train.csv',
    test='test.csv',
    fields=[('review', review), ('sentiment', sentiment)],
    skip_header=True
)

review.build_vocab(train_data, vectors="glove.6B.50d")
batch_size = 8

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort=False
)

cnt = 0
for b in train_iterator:
    cnt+=1
    print(b.review.size())
    if cnt==10:
        break

vocab = review.vocab
embed = nn.Embedding(len(vocab), 50)
embed.weight.data.copy_(vocab.vectors)

import tqdm
import torch.optim as optim

num_heads = 8
num_classes = 2
num_blocks = 10
#print(torch.cuda.memory_allocated())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
embed = embed
#print(torch.cuda.memory_allocated())
model = ClassificationTransformer(embed, device, num_blocks=num_blocks, num_heads=num_heads, num_classes=num_classes).to(device)
#print(torch.cuda.memory_allocated())
opt = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.NLLLoss()

epochs = 80

for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    model.train()
    #try:
    for train_obj in tqdm.tqdm(train_iterator):
        inp = train_obj.review.to(device)
        tar = train_obj.sentiment.to(device)
        #print(x.get_device())
        opt.zero_grad()

        preds = model(inp)
        loss = criterion(preds, tar)
        loss.backward()
        opt.step()

        running_loss += loss.item() * batch_size
        #gc.collect()
    epoch_loss = running_loss / len(train_iterator)

    val_loss = 0.0
    with torch.no_grad(): 
        for test_obj in test_iterator:
            x = test_obj.review.to(device)
            y = test_obj.sentiment.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            val_loss += loss.item() * batch_size


    val_loss /= len(test_iterator)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    #except Exception as e:
    #    print(e)
    #    continue


print("Done")