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

| Model | Size (M) | Accuracy | F1 macro | MCC average | energy_consumed_wh | emissions_gco2eq | 0_not_relevant | 1_not_happening | 2_not_human | 3_not_bad | 4_solutions_harmful_unnecessary | 5_science_is_unreliable | 6_proponents_biased | 7_fossil_fuels_needed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Modern-BERT-large |	395	| **0.758**	| **0.744**	| **0.716**	| 2.78716	| 0.15619	| 0.782	| 0.812	| 0.803	| 0.732	| **0.744**	| 0.725	| 0.734	| 0.631 |
| CT-BERT | 336 | 0.753 | 0.741 | 0.711 | 1.9166 | 0.1074 | 0.746 | 0.805 | **0.832** | 0.742 | **0.744** | **0.731** | 0.741 | 0.615 |
| gte-large | 434 | 0.747 | 0.737 | 0.704 | 2.39456 | 0.13419 | 0.743 | **0.818** | 0.796 | **0.784** | 0.719 | 0.669 | **0.763** | **0.662** |
| gte-base | 136 | 0.726 | 0.714 | 0.68 | 0.74074 | 0.04151 | 0.723 | 0.812 | 0.745 | 0.732 | 0.75 | 0.719 | 0.64 | 0.631 |
| Modern-BERT-base | 149 | 0.716 | 0.702 | 0.667 | 1.493 | 0.08367 | 0.765 | 0.792 | 0.715 | 0.639 | 0.738 | 0.669 | 0.676 | 0.569 |
| Sbert + SVM | X | 0.713 | 0.699 | 0.661 | 0.18236 | 0.01022 | **0.788** | 0.792 | 0.701 | 0.629 | 0.662 | 0.65 | 0.748 | 0.523 |
| Sbert + MLP | 0.065 | 0.705 | 0.689 | 0.655 | **0.01268** | **0.00071** | 0.72 | **0.818** | 0.686 | 0.649 | 0.656 | 0.706 | 0.698 | 0.615 |


# Hugging Face models
Our trained models are shared on HuggingFace:
* [Sbert + MLP](https://huggingface.co/ypesk/frugal-ai-EURECOM-mlp-768) / [Sbert + MLP (full data)](https://huggingface.co/ypesk/frugal-ai-EURECOM-mlp-768-fullset)


