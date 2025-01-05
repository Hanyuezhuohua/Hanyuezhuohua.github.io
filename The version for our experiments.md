The version for our experiments:

$\begin{align*}
    O' &= \frac{[A_{P}, A'_{C_1}, \ldots, A'_{C_N}, A_Q]}{\sum_{j=1}^{l_P}a_{P, j} + (\sum_{i=1}^{N}\sum_{j=1}^{l_{C_i}}a'_{C_i, j})^{S}+ \sum_{j=1}^{l_Q}a_{j}}\times [V_P, V_{C_1}, \ldots, V_{C_{N}}, V_Q], \\
    &\text{where }A'_{C_i} = [ \exp \frac{Qk_{C_i, 1}^\top}{T\sqrt{d}}, \ldots,  \exp \frac{Qk_{C_i, l_{C_i}}^\top}{T\sqrt{d}}] \cdot \frac{1}{(\sum_{i=1}^{N}\sum_{j=1}^{l_{C_i}}a'_{C_i, j})^{S-1}}\text{ and }a'_{C_{i}, j} = \exp\frac{Qk_{C_i, j}^\top}{T\sqrt{d}}.\notag
\end{align*}$

This version works for all experiments in our paper. Our current explanation of this method is that it can reduce more for the attention scores from the context for longer context. However, it will increase the attention score when $\sum_{i=1}^{N}\sum_{j=1}^{l_{C_i}}a'_{C_i, j}$ is smaller than 1. Therefore, it seems more like an adaptive way to adjust the importance of the context instead of naively scaling down.

INSCIT: 19.75



The verison in the original paper:

$\begin{align*}
    O' &= \frac{[A_{P}, A'_{C_1}, \ldots, A'_{C_N}, A_Q]}{\sum_{j=1}^{l_P}a_{P, j} + (\sum_{i=1}^{N}\sum_{j=1}^{l_{C_i}}a'_{C_i, j})^{T}+ \sum_{j=1}^{l_Q}a_{j}}\times [V_P, V_{C_1}, \ldots, V_{C_{N}}, V_Q], \\
    &\text{where }A'_{C_i} = [ S \cdot \exp \frac{Qk_{C_i, 1}^\top}{T\sqrt{d}}, \ldots,   S \cdot \exp \frac{Qk_{C_i, l_{C_i}}^\top}{T\sqrt{d}}] \cdot \frac{1}{(\sum_{i=1}^{N}\sum_{j=1}^{l_{C_i}}a'_{C_i, j})^{T-1}}\text{ and }a'_{C_{i}, j} = S \cdot \exp\frac{Qk_{C_i, j}^\top}{T\sqrt{d}}.\notag
\end{align*}$

I have tested this version on the original ICL tasks, which can achieve a good performance. However, it will result in more performance drop for tasks with various number of contexts. I have tried it on RAG setting and it fails. After that, I give up this setting. I will not use this setting again as mixing s and T is confusing. 



Another version I am testing now is:

$\begin{align*}
    O' &= \frac{[A_{P}, A'_{C_1}, \ldots, A'_{C_N}, A_Q]}{\sum_{j=1}^{l_P}a_{P, j} + (\sum_{i=1}^{N}\sum_{j=1}^{l_{C_i}}a'_{C_i, j})^{T}+ \sum_{j=1}^{l_Q}a_{j}}\times [V_P, V_{C_1}, \ldots, V_{C_{N}}, V_Q], \\
    &\text{where }A'_{C_i} = [S \cdot \exp \frac{Qk_{C_i, 1}^\top}{T\sqrt{d}}, \ldots, S \cdot \exp \frac{Qk_{C_i, l_{C_i}}^\top}{T\sqrt{d}}] \text{ and }a'_{C_{i}, j} = S \cdot\exp\frac{Qk_{C_i, j}^\top}{T\sqrt{d}}.\notag
\end{align*}$

This one will always scaling down the context attention weight for a fixed ratio. But seems not as good as the first one. 

INSCIT: 19.13 
