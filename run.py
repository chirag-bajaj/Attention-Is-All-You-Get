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

        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        out1 = self.sa(x)
        out2 = self.norm1(out1+x)
        out2 = self.drop(out2)
        out3 = self.mlp(out2)
        final = self.norm2(out3+out2)
        final = self.drop(final)
        return final

class Embedding(nn.Module):
    def __init__(self, embed, num_dim, device, max_len=10000):
        super().__init__()
        self.embed = embed
        self.num_dim = num_dim
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
        word_embedding = word_embedding * math.sqrt(self.num_dim)
        postional_encoding = self.pe[ :, :num_words]
        postional_encoding.requires_grad = False
        postional_encoding = postional_encoding.to(self.device)
        return word_embedding
        #return word_embedding + postional_encoding

class ClassificationTransformer(nn.Module):
    def __init__(self, embed, device, num_blocks=1, num_dim=100, num_heads=8, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.embbeding = Embedding(embed, num_dim, device)
        blocks = []
        for i in range(num_blocks):
            blocks.append(Transformer(num_dim, num_heads))
        self.transformer_blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(num_dim, num_classes)
    def forward(self, x):
        embedded_input = self.embbeding(x)
        num_batch, num_reviews, num_dim = embedded_input.shape
        output = self.transformer_blocks(embedded_input)
        output = self.out(output.mean(dim=1))
        return F.log_softmax(output, dim=1) 

import spacy
spacy_en = spacy.load('en')
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext import datasets
'''
review = Field(sequential=True, lower=True, tokenize=tokenize, batch_first=True, fix_length=512)
sentiment = Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

fields = {"review": ("review", review), "sentiment": ("sentiment", sentiment)}
train_data, test_data = TabularDataset.splits(
    path='/home/chirag_17bit012/Attention-Is-All-You-Get/data',
    format='csv',
    train='train.csv',
    test='test.csv',
    fields=[('review', review), ('sentiment', sentiment)],
    skip_header=True
)
'''
review = Field(sequential=True, lower=True, batch_first=True, fix_length=512)
sentiment = Field(sequential=False, pad_token=None, unk_token=None)
train_data, test_data = datasets.IMDB.splits(review,sentiment)

#ex = train_data[0]
#print(ex.sentiment)
#print(ex.review)

review.build_vocab(train_data, vectors="glove.6B.100d")
sentiment.build_vocab(train_data)
batch_size = 32

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort=False,
    shuffle=True
)
'''
cnt = 0
for b in train_iterator:
    cnt+=1
    print(b.review.size())
    if cnt==10:
        break
'''
vocab = review.vocab
embed = nn.Embedding(len(vocab), 100)
embed.weight.data.copy_(vocab.vectors)

import tqdm
import torch.optim as optim

num_heads = 8
num_classes = 2
num_blocks = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed = embed
model = ClassificationTransformer(embed, device, num_blocks=num_blocks, num_heads=num_heads, num_classes=num_classes).to(device)
print(model)
"""
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
"""
opt = optim.Adam(model.parameters(), lr=0.001)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (10000 / batch_size), 1.0))
criterion = nn.NLLLoss()

epochs = 300
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    running_corrects = 0
    tot = 0
    model.train()
    #try:
    for train_obj in tqdm.tqdm(train_iterator):
        inp = train_obj.text.to(device)
        tar = train_obj.label.to(device)
        #print(x.get_device())
        opt.zero_grad()

        preds = model(inp)
        loss = criterion(preds, tar)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1)

        opt.step()
        #sch.step()
        tot += float(inp.size(0))
        running_corrects += float((tar == preds.argmax(1)).sum().item())
        running_loss += loss.item() * batch_size
        
    epoch_acc = running_corrects / tot
    epoch_loss = running_loss / tot

    val_loss = 0.0
    val_corrects = 0
    tot = 0
    with torch.no_grad(): 
        model.eval()
        for test_obj in test_iterator:
            inp = test_obj.text.to(device)
            tar = test_obj.label.to(device)
            preds = model(inp)
            loss = criterion(preds, tar)
            tot += float(inp.size(0))   
            val_corrects += float((tar == preds.argmax(1)).sum().item())
            val_loss += loss.item() * batch_size

    val_acc = val_corrects / tot
    val_loss /=tot
    print('Epoch: {}, Training Acc: {:.4f}, Training Loss: {:.4f}, Validation Acc: {:.4f}, Validation Loss: {:.4f}'.format(epoch, 100*epoch_acc, epoch_loss, 100*val_acc, val_loss))
    #except Exception as e:
    #    print(e)
    #    continue


print("Done")
