# Distributed Representations of Language

This chapter explores how words can be represented as dense numerical vectors that capture semantic and syntactic relationships.

Starting from the distributional hypothesis, we progressively move from statistical representations of word context to learned word embeddings and neural language models.

The objective is to understand not only how these representations work, but also how they are constructed, trained, evaluated, and visualized.

## Learning Objectives

By the end of this chapter, you should be able to:

* Explain the distributional hypothesis and its role in NLP.
* Construct and analyze word co-occurrence matrices.
* Understand Latent Semantic Analysis and dimensionality reduction using SVD.
* Explain how neural language models learn word embeddings.
* Implement and compare Word2Vec (CBOW and Skip-gram) and GloVe.
* Evaluate and visualize word embeddings.

## Contents

| Notebook                       | Description                                                  | Status      |
| ------------------------------ | ------------------------------------------------------------ | ----------- |
| 01 - Distributional Hypothesis | Word context and co-occurrence statistics                    | In progress |
| 02 - LSA and SVD               | Latent semantic representations through matrix factorization | Planned     |
| 03 - Neural Bigram Embeddings  | Learning embeddings through next-word prediction             | Planned     |
| 04 - Word2Vec                  | CBOW, Skip-gram and negative sampling                        | Planned     |
| 05 - GloVe                     | Global co-occurrence statistics and embedding learning       | Planned     |
| 06 - Embedding Evaluation      | Similarity, analogies and visualization                      | Planned     |

## Prerequisites

Basic linear algebra, probability, Python, NumPy, and introductory NLP concepts such as tokenization and Bag of Words.

## References

* Harris (1954), Distributional Structure.
* Deerwester et al. (1990), Indexing by Latent Semantic Analysis.
* Mikolov et al. (2013), Efficient Estimation of Word Representations in Vector Space.
* Pennington et al. (2014), GloVe: Global Vectors for Word Representation.

This chapter is part of Eruditio, an open-source educational NLP repository exploring language representations, neural architectures, and language models from first principles.
