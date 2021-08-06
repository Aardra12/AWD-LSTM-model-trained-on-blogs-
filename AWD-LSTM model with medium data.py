#!/usr/bin/env python
# coding: utf-8

# In[3]:


#hide

#!sudo apt install git && sudo dpkg -i gh_*_linux_amd64.deb
get_ipython().system('pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()

from fastbook import *
from fastai.vision.widgets import *


# In[25]:


$ source activate fastai2
(fastai2)$ ipython kernel install --name "fastai2-kernel" --user


# In[21]:


get_ipython().system(' git clone --recurse-submodules https://github.com/fastai/fastai2')


# In[23]:


get_ipython().system('pip install fastai2 --quiet')


# In[19]:


get_ipython().system('brew install gh')


# In[20]:


get_ipython().system('git@github.com:fastai/fastai_dev.git')


# In[17]:


get_ipython().system('gh repo clone fastai/fastai_dev')


# In[ ]:


import fastbook
fastbook.setup_book()

#hide
from fastbook import *


# In[13]:


import fastai; fastai.__version__


# In[4]:


#export
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.text.core import *


# In[5]:


from fastai.text.all import *


# In[44]:


#export
def _get_tokenizer(ds):
    tok = getattr(ds, 'tokenizer', None)
    if isinstance(tok, Tokenizer): return tok
    if isinstance(tok, (list,L)):
        for t in tok:
            if isinstance(t, Tokenizer): return t


# In[40]:


#export
def _maybe_first(o): return o[0] if isinstance(o, tuple) else o


# In[42]:


#export
def _get_lengths(ds):
    tok = _get_tokenizer(ds)
    if tok is None: return
    return tok.get_lengths(ds.items)


# In[38]:


#export
class TensorText(TensorBase):   pass
class LMTensorText(TensorText): pass

TensorText.__doc__ = "Semantic type for a tensor representing text"
LMTensorText.__doc__ = "Semantic type for a tensor representing text in language modeling"


# In[36]:


#export
def make_vocab(count, min_freq=3, max_vocab=60000, special_toks=None):
    "Create a vocab of `max_vocab` size from `Counter` `count` with items present more than `min_freq`"
    vocab = [o for o,c in count.most_common(max_vocab) if c >= min_freq]
    special_toks = ifnone(special_toks, defaults.text_spec_tok)
    for o in reversed(special_toks): #Make sure all special tokens are in the vocab
        if o in vocab: vocab.remove(o)
        vocab.insert(0, o)
    vocab = vocab[:max_vocab]
    return vocab + [f'xxfake' for i in range(0, 8-len(vocab)%8)]


# In[29]:


#export
class Numericalize(Transform):
    "Reversible transform of tokenized texts to numericalized ids"
    def __init__(self, vocab=None, min_freq=3, max_vocab=60000, special_toks=None):
        store_attr('vocab,min_freq,max_vocab,special_toks')
        self.o2i = None if vocab is None else defaultdict(int, {v:k for k,v in enumerate(vocab)})

    def setups(self, dsets):
        if dsets is None: return
        if self.vocab is None:
            count = dsets.counter if getattr(dsets, 'counter', None) is not None else Counter(p for o in dsets for p in o)
            if self.special_toks is None and hasattr(dsets, 'special_toks'):
                self.special_toks = dsets.special_toks
            self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab, special_toks=self.special_toks)
            self.o2i = defaultdict(int, {v:k for k,v in enumerate(self.vocab) if v != 'xxfake'})

    def encodes(self, o): return TensorText(tensor([self.o2i  [o_] for o_ in o]))
    def decodes(self, o): return L(self.vocab[o_] for o_ in o)


# In[30]:


#export
class TextBlock(TransformBlock):
    "A `TransformBlock` for texts"
    @delegates(Numericalize.__init__)
    def __init__(self, tok_tfm, vocab=None, is_lm=False, seq_len=72, backwards=False, **kwargs):
        type_tfms = [tok_tfm, Numericalize(vocab, **kwargs)]
        if backwards: type_tfms += [reverse_text]
        return super().__init__(type_tfms=type_tfms,
                                dl_type=LMDataLoader if is_lm else SortedDL,
                                dls_kwargs={'seq_len': seq_len} if is_lm else {'before_batch': Pad_Chunk(seq_len=seq_len)})

    @classmethod
    @delegates(Tokenizer.from_df, keep=True)
    def from_df(cls, text_cols, vocab=None, is_lm=False, seq_len=72, backwards=False, min_freq=3, max_vocab=60000, **kwargs):
        "Build a `TextBlock` from a dataframe using `text_cols`"
        return cls(Tokenizer.from_df(text_cols, **kwargs), vocab=vocab, is_lm=is_lm, seq_len=seq_len,
                   backwards=backwards, min_freq=min_freq, max_vocab=max_vocab)

    @classmethod
    @delegates(Tokenizer.from_folder, keep=True)
    def from_folder(cls, path, vocab=None, is_lm=False, seq_len=72, backwards=False, min_freq=3, max_vocab=60000, **kwargs):
        "Build a `TextBlock` from a `path`"
        return cls(Tokenizer.from_folder(path, **kwargs), vocab=vocab, is_lm=is_lm, seq_len=seq_len,
                   backwards=backwards, min_freq=min_freq, max_vocab=max_vocab)


# In[32]:


#export
#TODO: add backward
@delegates()
class LMDataLoader(TfmdDL):
    "A `DataLoader` suitable for language modeling"
    def __init__(self, dataset, lens=None, cache=2, bs=64, seq_len=72, num_workers=0, **kwargs):
        self.items = ReindexCollection(dataset, cache=cache, tfm=_maybe_first)
        self.seq_len = seq_len
        if lens is None: lens = _get_lengths(dataset)
        if lens is None: lens = [len(o) for o in self.items]
        self.lens = ReindexCollection(lens, idxs=self.items.idxs)
        # The "-1" is to allow for final label, we throw away the end that's less than bs
        corpus = round_multiple(sum(lens)-1, bs, round_down=True)
        self.bl = corpus//bs #bl stands for batch length
        self.n_batches = self.bl//(seq_len) + int(self.bl%seq_len!=0)
        self.last_len = self.bl - (self.n_batches-1)*seq_len
        self.make_chunks()
        super().__init__(dataset=dataset, bs=bs, num_workers=num_workers, **kwargs)
        self.n = self.n_batches*bs

    def make_chunks(self): self.chunks = Chunks(self.items, self.lens)
    def shuffle_fn(self,idxs):
        self.items.shuffle()
        self.make_chunks()
        return idxs

    def create_item(self, seq):
        if seq>=self.n: raise IndexError
        sl = self.last_len if seq//self.bs==self.n_batches-1 else self.seq_len
        st = (seq%self.bs)*self.bl + (seq//self.bs)*self.seq_len
        txt = self.chunks[st : st+sl+1]
        return LMTensorText(txt[:-1]),txt[1:]

    @delegates(TfmdDL.new)
    def new(self, dataset=None, seq_len=None, **kwargs):
        lens = self.lens.coll if dataset is None else None
        seq_len = self.seq_len if seq_len is None else seq_len
        return super().new(dataset=dataset, lens=lens, seq_len=seq_len, **kwargs)


# In[6]:


path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path/'texts.csv')
df.head()


# In[7]:


imdb_lm = DataBlock(blocks=TextBlock.from_df('text', is_lm=True), get_x=ColReader('text'), splitter=ColSplitter())


# In[8]:


dls = imdb_lm.dataloaders(df, bs=64)
dls.show_batch(max_n=2)


# In[9]:


#startup text
path_train ="/storage/data/samplestartuptext/train_data.csv"
path_test="/storage/data/samplestartuptext/test_data_csv.csv"
df_test = pd.read_csv(path_test)
df_train =pd.read_csv(path_train)


# In[10]:


df_test.columns =['date','title','subtitle','claps','responses','author_url','story_url','reading_time (mins)','number_sections','section_titles','number_paragraphs','paragraphs']
df_train.columns =['date','title','subtitle','claps','responses','author_url','story_url','reading_time (mins)','number_sections','section_titles','number_paragraphs','paragraphs']


# In[11]:


df_all = pd.concat([df_train,df_test])


# In[14]:


print(df_all.head())


# In[12]:


startup_lm = DataBlock(blocks=TextBlock.from_df('paragraphs',is_lm=True),get_x=ColReader('paragraphs'),splitter=ColSplitter())


# In[15]:


#another approach
tfms = [attrgetter("paragraphs"), Tokenizer.from_df(0), Numericalize]
dsets = Datasets(df_all, [tfms],dl_type=LMDataLoader)


# In[16]:


bs,sl = 104,72
dls = dsets.dataloaders(bs=bs, seq_len=sl)


# In[17]:


dls.show_batch()


# In[18]:


config = awd_lstm_lm_config.copy()
config.update({'input_p': 0.6, 'output_p': 0.4, 'weight_p': 0.5, 'embed_p': 0.1, 'hidden_p': 0.2})
model = get_language_model(AWD_LSTM, len(dls.vocab), config=config)


# In[21]:


opt_func = partial(Adam, wd=0.1, eps=1e-7)
cbs = [MixedPrecision(clip=0.1), ModelResetter, RNNRegularizer(alpha=2, beta=1)]


# In[22]:


learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), opt_func=opt_func, cbs=cbs, metrics=[accuracy, Perplexity()])


# In[ ]:


learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), opt_func=opt_func, cbs=cbs, metrics=[accuracy, Perplexity()])


# In[23]:


learn.fit_one_cycle(1, 5e-3, moms=(0.8,0.7,0.8), div=10)


# In[20]:


#startup_dls =startup_lm.dataloaders(df_all, bs=64)


# In[ ]:





# In[ ]:




