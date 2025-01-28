---
title: Failure of Linear/Hybrid Model in the QMKAR Task 
published: 2025-01-26
tags: [Analysis, Benchmark]
category: Model Arch
draft: false
---

# Background

In the [Test-time regression](https://arxiv.org/abs/2501.12352) paper, the authors analyze sequence modeling with a test-time regression framework using the concept of associative memory. Specifically, given a set of associations $(\mathbf{k}_1, \mathbf{v}_1), ..., (\mathbf{k}_T, \mathbf{v}_T)$,  an associative memory system performs associative recall: return $\mathbf{v}_i$ when given $\mathbf{k}_i$. To construct such a system $m$, we can formulate the problem as a weighted regression problem:
$$
\min_{m \in \mathcal{M}} \frac{1}{2} \sum_{i=1}^{T} \gamma_i \|\mathbf{v}_i - m(\mathbf{k}_i)\|_2^2,
$$
where the importance of each association is controlled by adjusting the relative weights $\{\gamma_i\}_{i=1}^{T}$. When $m$ is flexible enough to interpolate all values, it is a *perfect* associative memory system. However, due to practical constraints, we can only approximate this system using different model architectures. Both linear attention and softmax attention can be viewed as  variants of such a system.  

One impressive aspect of this framework is its coherence with advances in Linear Attention Models:

* Linear attention is (suboptimal) linear least squares regression
* Gated linear attention and state-space models are (suboptimal) weighted linear least squares regression
* Fast weight programmers and online learners are first-order methods for solving streaming least-squares

For further details on this part, please refer to the following sources:
* [Original paper](https://arxiv.org/abs/2501.12352) from Ke Alexander Wang, Jiaxin Shi, Emily B. Fox
* [Twitter explanation](https://x.com/leloykun/status/1883561892926677029) from Franz Louis Cesista
* [Lecture slides](https://sustcsonglin.github.io/assets/pdf/talk_250117.pdf) from Songlin Yang

# Limitation

In the main part of this blog, we challenge the completeness of this framework for certain tasks (it is still an elegant framework 😃), as it assumes that each $\mathbf{k}_i$ should be associated with only one $\mathbf{v}_i$. Theoretically, this assumption poses a problem in cases like the counting task, where a query needs to retrieve and aggregate multiple related keys and values. However, in practice, the errors in counting tasks may arise from missing information or incorrect aggregation, while even correct models may only accumulate such information implicitly during the prefilling phase. To better examine this limitation, we introduce a new synthetic task, Query Multi-Key Associative Recall (QMKAR), to systematically evaluate model capability in such scenarios.

# Query Multi-Key Associative Recall

We begin with the Multi-Query Associative Recall (MQAR) task as introduced in [Based](https://arxiv.org/pdf/2402.18668).

$$
\underbrace{\text{A 4 B 3 C 6 F 1 E 2}}_{\textbf{Key-Value}} \quad \rightarrow \quad \underbrace{\text{A ? C ? F ? E ? B ?}}_{\textbf{Query}}
$$

To convert this into a Query-Merged Key Associative Recall (QMKAR) problem, we modify the key-value sequence to allow repeated keys, we only evaluate one query for simplicity:

$$
\underbrace{\text{A 4 B 3 A 6 F 1 A 2}}_{\textbf{Key-Value}} \quad \rightarrow \quad \underbrace{\text{A ?}}_{\textbf{Query}}
$$

# Practical Implementation

Instead of training small models, we directly evaluate the performance of cutting-edge industry models to ensure they shold be sufficiently powerful to address this simple task. However, since no large-scale pre-trained model is purely linear, we use [Minimax-01](https://www.minimaxi.com/en/news/minimax-01-series-2) as our target model, which is assumed to be stronger than linear models by incorporating a few softmax attention layers. For Transformer-based models, we select [ChatGPT 4o](https://openai.com/index/hello-gpt-4o/), [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet), [Deepseek V3](https://api-docs.deepseek.com/news/news1226), [Gemini 1.5](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/), and [Meta Llama 3.1 405B](https://ai.meta.com/blog/meta-llama-3-1/) for comparison.

Next, we format our task using the following prompt:

```markdown
k: 248, v: 5542; k: 169, v: 2882; k: 191, v: 2724; ...

If k is 230, what are v? Please directly give me all values without other output
```

This prompt format offers us three benefits:
* It prevents the model from finding answers in softmax attention layers using CoT reasoning.
* It allows the model a brief thinking phase after presenting the value of query.
* The hints 'are' and 'all' suggest that the answer includes multiple values.

For each question, we will generate 500 kv pairs, including 3 target ones. In addition to measuring the overall accuracy per question, we provide a more fine-grained evaluation at the KV pair level using the following metrics:
* TP (True Positive): Successfully retrieving a target value.
* FP (False Positive): Incorrectly retrieving a non-target value.
* FN (False Negative): Failing to retrieve a target value.

# Experimental Results

Based on a preliminary evaluation of 20 examples, we assessed the performance of various models on our QMKAR task, leading to several key observations.

### Overall performance comparison

1. Aligning with our assumption, Minimax-01, which utilizes multiple linear attention layers, completely fails on this synthetic dataset with a 15% accuracy. All failure cases involve forgetting some target values. In comparison, Transformers can achieve at least 50% accuracy.
2. Among Transformer-based models, while LLaMA underperforms—likely due to being an earlier open-weight model—we are surprised to find that even the most powerful models make mistakes on this task.

|                   | ✅ True | ❌ False w/ FN | ❌ False w/ FP | ❌ False w/ Both |
|:------------------:|:---------:|:--------------------:|:--------------------:|:--------------:|
| **ChatGPT 4o**        | 18  | 1  | 1  | 0  |
| **Claude 3.5 Sonnet** | 14   | 0  | 6  | 0  |
| **LLaMA 3.1 405B**    | 10   | 8  | 1  | 1  |
| **Minimax-01**        | 3   | 11  | 0  | 5  |
| **Deepseek V3**       | 13  | 5  | 1  | 1  |
| **Gemini 1.5**        | 18  | 1  | 1  | 0  |

### Fine-grained performance comparison

1. The Hybrid model exhibits a significantly higher rate of missing associative values for a given query compared to Transformer-based models. 

2. Both Claude 3.5 Sonnet and Minimax-01 struggle with incorrectly retrieving non-target values. In the hybrid model, the primary failure stems from retrieving values associated with similar keys (e.g., 150 and 151).

|                   |  TP  |  FP  |  FN  | FN-Pos0 | FN-Pos1 | FN-Pos2 |
| ----------------- | :--: | :--: | :--: | :-----: | :-----: | :-----: |
| **ChatGPT 4o**         |  59  |  1   |  1   |    1    |    0    |    0    |
| **Claude 3.5 Sonnet** |  60  |  10   |  0   |    0    |    0    |    0    |
| **LLaMA 3.1 405B**            |  50  |  3   |  10   |    8    |    0    |    2    |
| **Minimax-01**             |  36  |  10   |  24  |    2    |    7    |   15    |
| **Deepseek V3**       |  55  |  2   |  5   |    4    |    1    |    1    |
| **Gemini 1.5**        |  59  |  2   |  1   |    0    |    0    |    1    |

### The surprising 'reverse' phenomenon in the hybrid model


Another crucial observation is that Minimax-01 tends to miss more values from nearby positions, which appears abnormal. Ideally, one would expect a more significant decay in attention to earlier information, making this discrepancy noteworthy. We hypothesize that the hybrid architecture may compel the softmax attention layers to prioritize distant tokens, leading to an unexpected 'reverse' phenomenon. However, we just provide this hypothesis as quantifying this effect rigorously remains challenging without large-scale linear models.

### Can more advanced linear model architecture address this problem?

Given recent advances in linear model architectures, particularly [DeltaNet](https://arxiv.org/abs/2406.06484), [GatedDeltaNet](https://arxiv.org/abs/2412.06464), and [TTT](https://arxiv.org/abs/2407.04620), one would ask whether these methods can address this problem. HHowever, since these approaches follow the test-time regression framework and optimize the weighted regression problem, they cannot address this challenge theoretically. An intuitive understanding of these methods, as outlined in [Songlin's Blog](https://sustcsonglin.github.io/blog/2024/deltanet-1/), is that their update rules first remove information in the past memory that is related to the current key, which contradict the QMKAR task.

$$
\mathbf{v}_t^{\text{new}} = (1 - \beta_t) \mathbf{v}_t^{\text{old}} + \beta_t \mathbf{v}_t,
$$

$$
\mathbf{S}_t = \mathbf{S}_{t-1} - \underbrace{\mathbf{v}_t^{\text{old}} \mathbf{k}_t^\top}_{\text{erase}} + \underbrace{\mathbf{v}_t^{\text{new}} \mathbf{k}_t^\top}_{\text{write}},
$$

where $\mathbf{v}_t^{\text{new}}$ is a learned combination of the old and current values, controlled by a dynamic $\beta_t \in (0,1)$.

# Ablation Study

While the above experiments show the ineffectiveness of linear/hybrid models on our task, we need to further classify the source of this problem do not come from the limited memorization ability of linear attention.

### MQAR

First, we use the MQAR task to demonstrate that all models can recall associative values from three distinct keys. We remove duplicate keys from the input in this task and refine the prompt as follows:

```markdown
k: 248, v: 5542; k: 169, v: 2882; k: 191, v: 2724; ...

If k is one of 129, 175, 342, what are possible v? Please directly give me all values without other output
```

We observe that Minimax-01 performs good in this task, indicating it should have the ability to memoerize 500 kv pairs. Additionally, the two failure cases further support the idea that it may struggle to distinguish between similar keys.

|                   | ✅ True | ❌ False w/ FN | ❌ False w/ FP | ❌ False w/ Both |
|:------------------:|:---------:|:--------------------:|:--------------------:|:--------------:|
| **ChatGPT 4o**        | 19  | 0  | 0  | 1  |
| **Claude 3.5 Sonnet** | 20   | 0  | 0  | 0  |
| **LLaMA 3.1 405B**    | 16   | 1  | 0  | 3  |
| **Minimax-01**        | 18   | 0  | 2  | 0  |
| **Deepseek V3**       | 19  | 0  | 0  | 1  |
| **Gemini 1.5**        | 20  | 0  | 0  | 0  |

### QMKAR with More Values Related to the Target Key

In the second ablation study, we increased the frequency of target keys from 3 to 5 occurrences across all kv pairs. Our results show that Minimax-01 exhibited increased forgetting of values associated with these target keys. This supports our assumption that the model tends to retain fewer values per key.

```markdown
4527, 6383 # Answer for 3 target keys

6179, 4528, 7902 # Answer for 5 target keys
```

# Counting Problem

Finally, we go back to the counting problem by randomly generating 500 values in a certain range (i.e., 0-50). After that, we answer the models the occurence of a certain value. The prompt format is following:

```markdown
37, 21, 20, 20, 15, 35, 10, 24, 2, 9, 44, ...

What is the count of values equal to 44? Please answer this question without using any external tool
```

Even when LLMs are allowed to generate intermediate steps to tackle this challenge, this counting problem remains highly challenging for LLMs,  with only models specifically trained for reasoning tasks (such as R1 and o1) performing well by thinking (actually scanning the list again) before generation. The best non-reason models achieves only 20% accuracy. Here, we highlight some intriguing observations regarding this counting task:
* Most correct answers are obtained when the model directly scans the input values again. However, despite this intensive process, the model still makes numerous errors.
* The model occasionally misidentifies a significantly larger number of values as the target.
* R1 and o1 can address these challenges, but in an inefficient way by scanning and thinking over the whole context.

| **ChatGPT 4o** | **Claude 3.5 Sonnet** | **LLaMA 3.1 405B**    | **Minimax-01**        | **Deepseek V3**       | **Gemini 1.5**        | **Deepseek R1**       |**o1**       |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| 4/20  | 3/20  | 1/20 | 2/20 | 4/20 | 3/20| 20/20 | 20/20 |


# Reproducible Code

We provide the data generation process of QMKAR, MQAR, and counting problems at. Currently, we assess these problems by manually interacting with the models via the website. An automated version with API support will be released in a future update.