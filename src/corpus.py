"""A 30-passage hand-curated corpus for tiny RAG.

Each passage is a short factual paragraph about a topic that downstream questions
will probe. Keeping the corpus small and hand-written removes any dataset-download
dependency and makes the eval reproducible across machines.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str

    @property
    def joined(self) -> str:
        return f"{self.title}. {self.text}"


PASSAGES: list[Passage] = [
    Passage("p01", "Python history",
            "Python was created by Guido van Rossum and first released in 1991. "
            "Its design philosophy emphasizes code readability and a syntax that allows "
            "programmers to express concepts in fewer lines of code."),
    Passage("p02", "Transformer architecture",
            "The Transformer architecture was introduced by Vaswani et al. in the 2017 paper "
            "Attention Is All You Need. It replaces recurrence with self-attention and parallel "
            "computation, which is why it powers modern large language models."),
    Passage("p03", "BERT model",
            "BERT (Bidirectional Encoder Representations from Transformers) was proposed by "
            "Devlin et al. at Google in 2018. It is pre-trained with a masked language modeling "
            "objective and a next-sentence-prediction task."),
    Passage("p04", "ResNet architecture",
            "ResNet, introduced by He et al. at Microsoft Research in 2015, popularized residual "
            "connections that let very deep networks train without vanishing gradients. The original "
            "ResNet-152 reached 3.57% top-5 error on ImageNet."),
    Passage("p05", "ImageNet dataset",
            "ImageNet is a large visual database designed for visual object recognition research, "
            "released by Fei-Fei Li and collaborators in 2009. The classic ImageNet-1k subset "
            "contains over 1.2 million labeled training images across 1,000 classes."),
    Passage("p06", "BM25 ranking",
            "BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks documents by "
            "term frequency and inverse document frequency, with length normalization. It was "
            "developed by Stephen Robertson and Karen Sparck Jones in the 1990s."),
    Passage("p07", "TF-IDF",
            "TF-IDF stands for term frequency times inverse document frequency. It weights words "
            "by how often they appear in a document (TF) and how rare they are across the corpus (IDF). "
            "It is a foundation of classical information retrieval."),
    Passage("p08", "Word2Vec",
            "Word2Vec is a family of shallow neural network models trained to learn dense word "
            "representations from large text corpora. It was introduced by Mikolov et al. at Google "
            "in 2013 and comes in two flavors: skip-gram and continuous bag-of-words."),
    Passage("p09", "ReLU activation",
            "The Rectified Linear Unit (ReLU) outputs max(0, x). It was popularized for deep neural "
            "networks by Nair and Hinton in 2010 and is now the default hidden-layer activation in "
            "most feed-forward and convolutional architectures."),
    Passage("p10", "Adam optimizer",
            "Adam is an adaptive learning-rate optimization algorithm for training deep neural networks. "
            "It was proposed by Kingma and Ba in 2014 and combines momentum and RMSProp ideas. The "
            "default hyperparameters are beta1=0.9, beta2=0.999."),
    Passage("p11", "Dropout regularization",
            "Dropout is a regularization technique that randomly sets a fraction of hidden units to "
            "zero during training. It was introduced by Srivastava et al. in 2014 and helps prevent "
            "overfitting in deep neural networks."),
    Passage("p12", "Convolutional neural networks",
            "Convolutional Neural Networks (CNNs) apply learnable filters across a spatial input. "
            "LeNet-5, designed by Yann LeCun in 1998, was an early CNN used to read handwritten digits "
            "on bank cheques."),
    Passage("p13", "Recurrent neural networks",
            "Recurrent Neural Networks (RNNs) process sequences one step at a time and maintain "
            "a hidden state. The Long Short-Term Memory (LSTM) variant, introduced by Hochreiter and "
            "Schmidhuber in 1997, addresses the vanishing-gradient problem of vanilla RNNs."),
    Passage("p14", "Reinforcement learning",
            "Reinforcement Learning is a paradigm where an agent learns to act in an environment by "
            "maximizing a reward signal. Q-learning is a model-free algorithm introduced by Chris "
            "Watkins in 1989 that estimates the value of state-action pairs."),
    Passage("p15", "AlphaGo program",
            "AlphaGo was developed by DeepMind and became the first computer program to defeat a "
            "professional human Go player without handicaps. It combined deep neural networks with "
            "Monte Carlo Tree Search and beat Lee Sedol 4-1 in March 2016."),
    Passage("p16", "Generative adversarial networks",
            "Generative Adversarial Networks (GANs) consist of a generator and a discriminator trained "
            "in a minimax game. They were introduced by Ian Goodfellow and collaborators in 2014 and "
            "produce realistic synthetic images."),
    Passage("p17", "Diffusion models",
            "Diffusion models generate data by learning to reverse a gradual noising process. The "
            "denoising diffusion probabilistic model formulation was proposed by Ho et al. in 2020 "
            "and underpins systems like Stable Diffusion."),
    Passage("p18", "GPT family",
            "The GPT (Generative Pretrained Transformer) family started with GPT-1 from OpenAI in 2018, "
            "a 117 million parameter decoder-only Transformer pretrained on books. Later versions scaled "
            "the same architecture to billions of parameters."),
    Passage("p19", "Whisper speech model",
            "Whisper is an automatic speech recognition model open-sourced by OpenAI in 2022. It is "
            "trained on 680,000 hours of multilingual and multitask supervised data and uses a "
            "Transformer encoder-decoder architecture."),
    Passage("p20", "CLIP model",
            "CLIP (Contrastive Language-Image Pretraining) was released by OpenAI in 2021. It learns "
            "a shared image-text embedding space by contrastive training over 400 million image-caption "
            "pairs collected from the web."),
    Passage("p21", "Knowledge distillation",
            "Knowledge distillation transfers knowledge from a large teacher network to a smaller "
            "student network. It was popularized by Hinton, Vinyals and Dean in 2015 and uses softened "
            "teacher logits as additional supervision."),
    Passage("p22", "Pruning",
            "Network pruning removes weights, neurons or filters whose contribution is small. "
            "Magnitude pruning was a key technique in Deep Compression, proposed by Han et al. in 2016, "
            "achieving large size reductions with little accuracy drop."),
    Passage("p23", "Quantization",
            "Quantization reduces the bit-width of weights and activations from 32-bit floating point "
            "to lower precision such as 8-bit integers. INT8 inference is a standard deployment "
            "technique on mobile and edge hardware."),
    Passage("p24", "MLOps",
            "MLOps is a set of practices for deploying and maintaining machine learning models in "
            "production. It borrows ideas from DevOps and adds concerns specific to ML such as data "
            "versioning, experiment tracking and model monitoring."),
    Passage("p25", "Hugging Face library",
            "The Hugging Face Transformers library, released in 2018, provides a unified API for "
            "thousands of pretrained models. It supports both PyTorch and TensorFlow backends and "
            "underpins most production NLP pipelines."),
    Passage("p26", "ONNX format",
            "ONNX (Open Neural Network Exchange) is an open format for representing machine learning "
            "models. It was launched by Facebook and Microsoft in 2017 and lets models trained in one "
            "framework be deployed in another."),
    Passage("p27", "Federated learning",
            "Federated learning trains a shared model across devices that keep their data local, sending "
            "only model updates to a central server. The term and FedAvg algorithm were introduced by "
            "McMahan et al. at Google in 2017."),
    Passage("p28", "Self-supervised learning",
            "Self-supervised learning trains models on pretext tasks derived from unlabeled data, such "
            "as predicting masked words or image patches. SimCLR by Chen et al. in 2020 is a popular "
            "contrastive method for images."),
    Passage("p29", "JEPA architecture",
            "Joint-Embedding Predictive Architectures (JEPA) predict in a learned latent space rather "
            "than in pixel space. They were promoted by Yann LeCun starting around 2022 as a path "
            "toward world models that avoid generating fine-grained details."),
    Passage("p30", "Hyperparameter tuning",
            "Hyperparameter tuning searches over learning rate, batch size, regularization and "
            "architectural choices. Bayesian optimization and successive halving are popular search "
            "strategies; the Optuna library, released in 2019, is a common implementation."),
]


def passage_index() -> dict[str, Passage]:
    return {p.pid: p for p in PASSAGES}
