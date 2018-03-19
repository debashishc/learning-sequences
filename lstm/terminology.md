##

## S

softmax function: function maps arbitrary values $x_i$ to a probability distribution $p_i$. max because amplifies probability of largest $x_i$ and soft becasuse still assigns some probability to smaller $x_i$

$$softmax(x_i) = \frac{exp(x_i)}{\sum_{j=1}^n exp(x_j)} = p_i$$


## W

word vectors: dense vector for each word, chosen so that it is similar to vectors of words that appear in similar contexts. alias: word embeddings, word representations
