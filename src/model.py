"""BiLSTM-CRF sequence tagger with character CNN.
Architecture (per token):
    word embedding (100d)  +  char-CNN over the word's characters (100d)
        -> 2 stacked BiLSTM layers (200 per direction, dropout between)
        -> one Linear + CRF head PER TASK
Tasks are passed as {"ner": 7} today; adding {"pos": 17} later turns the same
encoder into the joint POS+NER model (shared BiLSTM, separate heads).
The char-CNN matters for Gujarati: words carry suffixes (-માં, -નો, -ને ...) and many
names are OOV, so the model must read spelling, not only look words up.
"""
import torch
import torch.nn as nn
from torchcrf import CRF
#PyTorch libraries for building the deep learning model and CRF for predicting the best sequence of tags.

class CharCNN(nn.Module): # handles char level info and CNN layer to extract features from the spelling of each word
    def __init__(self, n_chars, dim=50, out=100, kernel=3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, dim, padding_idx=0)
        self.conv = nn.Conv1d(dim, out, kernel, padding=kernel // 2)

    def forward(self, chars):  #function takes characters of each word, converts them into numerical values and apply relu  
        B, T, L = chars.shape
        flat = chars.view(B * T, L)
        x = torch.relu(self.conv(self.emb(flat).transpose(1, 2)))   # (B*T) x out x L
        x = x.masked_fill((flat == 0).unsqueeze(1), float("-inf")).max(dim=2).values
        x = x.masked_fill(torch.isinf(x), 0.0)        # padding positions (all-pad words)
        return x.view(B, T, -1)

 
class BiLSTMTagger(nn.Module): # main function create word embbedding  , create 2 BiLSTML layers 
    def __init__(self, n_words, n_chars, tasks, word_dim=100, char_dim=50, char_out=100,
                 hidden=200, layers=2, dropout=0.5):
        super().__init__()
        self.config = dict(n_words=n_words, n_chars=n_chars, tasks=dict(tasks), word_dim=word_dim,
                           char_dim=char_dim, char_out=char_out, hidden=hidden, layers=layers,
                           dropout=dropout)
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        self.char_cnn = CharCNN(n_chars, char_dim, char_out)
        self.drop = nn.Dropout(dropout)


        # Stacked 1-layer BiLSTMs with our own dropout between them, instead of
        # nn.LSTM(num_layers=2, dropout=...): on Windows + cuDNN 9.19 the built-in inter-layer
        # dropout makes the process crash at exit (0xC0000409) after any backward pass.
        # Same computation, keeps cuDNN speed.


#
        self.lstms = nn.ModuleList(nn.LSTM(word_dim + char_out if i == 0 else 2 * hidden, hidden, bidirectional=True, batch_first=True)

            for i in range(layers))   #picks the best tag sequence for the whole sentence
        
        self.heads = nn.ModuleDict({t: nn.Linear(2 * hidden, n) for t, n in tasks.items()}) #gives a score to every tag for each word
        self.crfs = nn.ModuleDict({t: CRF(n, batch_first=True) for t, n in tasks.items()})#picks the best tag sequence for the whole sentence

    
    
    def encode(self, words, chars, mask):#Combines word and character numbers, runs them through the BiLSTM layers and return give us context 
        x = torch.cat([self.word_emb(words), self.char_cnn(chars)], dim=-1)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(x), lengths, batch_first=True,
                                                   enforce_sorted=False)
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                packed = packed._replace(data=self.drop(packed.data))
            packed, _ = lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=words.size(1))
        return self.drop(h)

    def emissions(self, words, chars, mask):#Turns those into a score for each tag
        h = self.encode(words, chars, mask)
        return {t: head(h).float() for t, head in self.heads.items()}

    def loss(self, words, chars, mask, targets):
        #Used in training. Measures how wrong the model is compared to the correct tags.
        """targets: {task: B x T LongTensor}. Returns summed mean negative log-likelihood."""
        em = self.emissions(words, chars, mask)
        return sum(-self.crfs[t](em[t], y, mask=mask, reduction="mean") for t, y in targets.items())

    @torch.no_grad()
    def decode(self, words, chars, mask): #Used in prediction. The CRF picks the final tags.
        em = self.emissions(words, chars, mask)
        return {t: self.crfs[t].decode(e, mask=mask) for t, e in em.items()}

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)
