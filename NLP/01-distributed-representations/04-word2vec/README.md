# Word2Vec: Learning Distributed Word Representations

## Abstract
Representing words as numerical vectors is the first step of almost every NLP system. This notebook places **Word2Vec** (Mikolov et al., 2013) in its historical and theoretical context. We first review two earlier paradigms, **n-gram (bigram) language models** and **Latent Semantic Analysis (LSA)**, and show the specific limitations that motivated Word2Vec. We then state the aims of Word2Vec and introduce its two architectures, **CBOW** and **Skip-gram**, which are studied in two dedicated notebooks.

---

## 1. Theoretical foundation: the distributional hypothesis

All the methods in this notebook rest on one linguistic idea:

> *"Words that occur in similar contexts tend to have similar meanings."* (Harris, 1954)
>
> *"You shall know a word by the company it keeps."* (Firth, 1957)

The methods differ in **how** they turn this hypothesis into a computational model: by **counting** (n-grams, LSA) or by **predicting** (neural models, Word2Vec).

---

## 2. Bigram / n-gram language models

### 2.1 Formulation
A language model assigns a probability to a sequence of words $w_1, \dots, w_T$. By the chain rule:

$$P(w_1,\dots,w_T)=\prod_{t=1}^{T} P(w_t \mid w_1,\dots,w_{t-1})$$

The **Markov assumption** limits the history to the previous $n-1$ words. For a **bigram** model ($n=2$):

$$P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-1})$$

The parameters are estimated by **maximum likelihood** from corpus counts $C(\cdot)$:

$$\hat{P}(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$$

### 2.2 Limitations
| Problem | Explanation |
|---|---|
| **Data sparsity** | The number of possible n-grams grows as $\lvert V\rvert^{n}$. Most valid sequences never occur in the training corpus and get probability 0. Smoothing (Laplace, back-off, Kneser-Ney) is only a patch. |
| **Curse of dimensionality** | Words are discrete symbols (one-hot vectors of dimension $\lvert V\rvert$). Each new word or context adds parameters. |
| **No notion of similarity** | One-hot vectors are orthogonal: $\text{sim}(\text{cat}, \text{dog}) = \text{sim}(\text{cat}, \text{economy}) = 0$. Observing "the cat sleeps" teaches nothing about "the dog sleeps". |
| **Short context** | Increasing $n$ to capture more context makes sparsity explode. |

**Key takeaway:** n-gram models have **no generalisation mechanism** across similar words, because the representation itself carries no semantics.

---

## 3. Latent Semantic Analysis (LSA)

### 3.1 Formulation (Deerwester et al., 1990)
LSA builds a **word-document** (or word-word) co-occurrence matrix $X \in \mathbb{R}^{\lvert V\rvert \times D}$, usually weighted with TF-IDF or PMI, and applies a **truncated Singular Value Decomposition**:

$$X \approx U_k \Sigma_k V_k^{\top}$$

The rows of $U_k\Sigma_k$ are **dense $k$-dimensional word vectors** ($k \ll \lvert V\rvert$, typically 100 to 500). The truncation keeps the $k$ strongest latent dimensions and discards noise, which lets words that never co-occur directly still end up close if they share contexts.

### 3.2 Contributions over n-grams
- Dense, low-dimensional representations.
- Semantic similarity measurable with cosine similarity.
- Some robustness to synonymy.

### 3.3 Limitations
| Problem | Explanation |
|---|---|
| **Computational cost** | The SVD of a large matrix costs roughly $O(\min(\lvert V\rvert D^2,\ \lvert V\rvert^2 D))$ and does not scale to billion-word corpora. |
| **Static and batch** | Adding new documents or words requires recomputing the decomposition. |
| **Count bias** | Raw counts are dominated by frequent words, so weighting heuristics are needed. |
| **Bag-of-words** | Word order and local syntax are ignored when documents are the context. |
| **Weak linear structure** | Similarity is captured, but **relations** between words (e.g. gender, tense) are not reliably encoded as consistent directions. |

**Key takeaway:** LSA solves *representation* but not *scalability*, and it is an **unsupervised matrix factorisation**, not a model trained with a predictive objective.

---

## 4. The bridge: neural language models

Bengio et al. (2003) proposed learning word vectors **jointly** with a language model. Each word is mapped to a dense vector by a shared embedding matrix, the vectors of the previous $n-1$ words are concatenated, passed through a non-linear hidden layer, and a softmax over $V$ predicts the next word.

This brought two important ideas:
1. **Embeddings are learned** as parameters through a predictive objective.
2. **Generalisation** across similar words, since similar words receive similar vectors.

But the training cost per example is dominated by the hidden layer and the output softmax:

$$Q_{\text{NNLM}} = N\times D + N\times D\times H + H\times V$$

where $N$ is the context size, $D$ the embedding dimension, $H$ the hidden size and $V$ the vocabulary size. This made training on very large corpora impractical.

---

## 5. Word2Vec (Mikolov et al., 2013)

### 5.1 Historical position
Word2Vec was introduced by Mikolov, Chen, Corrado and Dean at Google in two papers:
- *Efficient Estimation of Word Representations in Vector Space* (2013a): the two architectures.
- *Distributed Representations of Words and Phrases and their Compositionality* (2013b): negative sampling, subsampling of frequent words, phrase detection.

It sits at the intersection of the previous approaches: it keeps the **predictive, learned** nature of neural language models while dropping everything that made them slow.

### 5.2 Aims
1. **Quality:** learn dense vectors (typically 100 to 300 dimensions) that encode both **semantic** and **syntactic** regularities.
2. **Efficiency:** train on corpora of **billions of words** in hours, by
   - removing the non-linear hidden layer,
   - approximating the softmax (hierarchical softmax, negative sampling),
   - subsampling very frequent words.
3. **Linear structure:** make relations appear as **vector offsets**, so that analogies can be solved by vector arithmetic:

$$\vec{v}(\text{king}) - \vec{v}(\text{man}) + \vec{v}(\text{woman}) \approx \vec{v}(\text{queen})$$

4. **Reusability:** produce general-purpose embeddings that can be transferred to downstream tasks.

### 5.3 The two architectures
Word2Vec defines two shallow models that differ in the **prediction task** they use to learn the embeddings:

| Architecture | Dedicated notebook |
|---|---|
| **CBOW** (Continuous Bag-of-Words) | Notebook 1 |
| **Skip-gram** | Notebook 2 |

Their mechanics, objective functions and training tricks are developed in the two notebooks.

### 5.4 Evaluation protocol
Embedding quality is usually assessed by:
- **Word analogy task** (Mikolov's semantic-syntactic test set, about 19.5k questions).
- **Word similarity benchmarks** (e.g. WordSim-353, SimLex-999), comparing cosine similarity with human judgements.
- **Extrinsic evaluation** on downstream tasks.

---

## 6. Synthesis

| Criterion | Bigram / n-gram | LSA | Word2Vec |
|---|---|---|---|
| **Paradigm** | Counting | Counting + matrix factorisation | Prediction (learned) |
| **Representation** | Sparse, one-hot | Dense, from SVD | Dense, learned by gradient descent |
| **Semantic similarity** | No | Yes (topical) | Yes (semantic + syntactic) |
| **Relational structure (analogies)** | No | Weak | Strong |
| **Local context / word order** | Yes, very short | No (document level) | Yes (context window) |
| **Scalability** | Limited by sparsity | Limited by SVD | Very high |

### A theoretical link
Levy and Goldberg (2014) showed that Skip-gram with negative sampling implicitly factorises a **shifted PMI word-context matrix**. The "prediction vs counting" opposition is therefore less sharp than it first appears: Word2Vec is closely related to LSA-style factorisation, but it reaches it through an efficient, online, stochastic optimisation.

### Limitations that motivated later work
- **One vector per word type** (no polysemy): motivates contextual embeddings (ELMo, BERT).
- **Out-of-vocabulary words:** motivates subword models (fastText).
- **Only local context** is used: GloVe (Pennington et al., 2014) combines it with global co-occurrence statistics.

---

## References
1. Harris, Z. (1954). *Distributional structure*. Word, 10(2-3).
2. Firth, J. R. (1957). *A synopsis of linguistic theory 1930-1955*.
3. Deerwester, S. et al. (1990). *Indexing by Latent Semantic Analysis*. JASIS, 41(6).
4. Bengio, Y. et al. (2003). *A Neural Probabilistic Language Model*. JMLR, 3.
5. Mikolov, T. et al. (2013a). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781.
6. Mikolov, T. et al. (2013b). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
7. Levy, O. & Goldberg, Y. (2014). *Neural Word Embedding as Implicit Matrix Factorization*. NeurIPS.
8. Pennington, J., Socher, R. & Manning, C. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP.