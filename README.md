# Frugal Ai Challenge Participation
This repository contains the code of EURECOM submission to the [Frugal Ai Challenge](https://frugalaichallenge.org/)

# Approach

Our model uses the Sentence-transformers python library to embed the sentences into vectors of 768 dimensions, using the [sentence-transformers/sentence-t5-large](https://huggingface.co/sentence-transformers/sentence-t5-large) model.

We then train a very lightweight Multi Layer Perceptron classifier, with the following layers:

`768 → 100 → 100 → 100 → 50 → 8`
 

We use the AdamW optimizer with a learning rate of 5e-4, and a weighted CrossEntropy loss function, with weights inversely proportional to the number of sample of the given class.

Our goal was to have a model that can compute inference fast to reduce emissions, as the MLP only has 64k parameters. This also means that this model is very fast to train, therefore also reducing the emissions during training compared to larger models (BERT-based, etc.).

Further work could be done exploring even lighter sentence-bert models that perform well on the mteb leaderboard, and search for better hyper-parameters.

# Other experiments
During this challenge, we have also experimented using larger, more accurate models, but also more energy-consuming. We have tested:
* [Modern-BERT-large](https://huggingface.co/answerdotai/ModernBERT-large) (395M parameters)
* [Modern-BERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M parameters)
* [gte-large](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) (434M parameters)
* [gte-base](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) (136M parameters)
* [covid-twitter-bert](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert) (336M parameters)

As an alternative to light-weight models, we have also tried using SVM from sklearn library instead of MLP.

# Results
We train each model on the same 80% of the data and test them on the same 20%




