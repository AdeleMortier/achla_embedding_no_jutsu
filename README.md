# Achla embedding no jutsu

Project testing differences in L1 *vs* L2 morphology in various word embeddings.

## Pipeline

The project is organized in notebooks, which should be executed in the following order:

1. `wiki_corpus_generator.ipynb` : generates a corpus from Wikipedia dumps, in order to train models. Corpora are stored in `./corpora/`
2. `training.ipynb` : trains Word2Vec or GloVe on the specified corpus. The word embeddings that result from training are stored in `./models/raws/`
3. `hebrew_dataset.ipynb` : generates a dataset of Hebrew words to test. Datasets are stored in `./hebrew dataset/`
4. `embedding_reduction.ipynb` : reduces the dimension and vocabulary size of a given embedding, given a specific dataset. Reduced embeddings are stored in `./models/reduced/`
5. `testing.ipynb` : performs statistical tests testing a specific hypothesis, on a specific dataset, and model. Test reports (including *p*-values and effect sizes) are stored in `./reports/`.

Step 1. and 2. may be skipped if the model to be tested is pre-trained. Pre-trained models can be loaded and reduced directly form the `embedding_reduction` notebook.

## Background

Morphologists distinguish between level-1 (L1) and level-2 (L2) operations. L1 operations are supposed to happen "closer" to the root than L2 operations. In the distributed morphology framework, this corresponds to the claim that L1 operations amount to the merger of the first functional head, which sets the general semantic and phonological properties of the newly created word; L2 operation then correspond to the merger of higher heads. The L1 *vs* L2 distinction is motivated by various observations:

- **Semantic contrast** : L1 operations usually lead to more idiosyncratic meanings than L2 operations. In other words, L2 operations are supposed to lead to more systematic semantic effects.
- **Phonological contrast** : L1 operations can modify the phonological features of the word, for instance, stress assignment. L2 operations cannot.

## Objectives

The project is intended to test for differences in the semantic encoding of L1 *vs* L2 morphology in various test cases, including:

* **Hebrew** : testing if denominal verbs (verbs derived from a noun *via* an L2 operation) are semantically closer to their base noun than verbs derived from the same root as the noun (L1 operation) are to the noun.
* **English** : testing if *-ness* affixation (L2 operation) has less variable semantic effects than *-ity* affixation (L1 operation). Same research question for *-less* (L2) and *-al* (L1) affixation.
* **Japanese** : upcoming!