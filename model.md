请解释一下下面的要设计的模型和实验是在做什么：核心模型
构建一个异构证据图：

- 节点：文档、chunk 或原子 claim

- support 边：两个证据相互支持或共同完成一个推理步骤

- bridge 边：共享实体、时间、因果关系，负责多跳连接

- contradict 边：两个 claim 相互矛盾

- dependency 边：转载、引用、同一原始来源、近重复改写

  检索流程：

  用户问题     -> hybrid retrieval 得到 100~300 个候选证据     -> 识别来源依赖/转载关系并收缩 provenance group     -> 构造 query-conditioned signed evidence graph     -> 挖掘预算约束的独立证据稠密子图     -> 按推理拓扑序列化给 LLM     -> 回答、引用、置信度或拒答

  关键不是最大化普通的 |E(S)| / |V(S)|，而是最大化“每个 token 中包含多少独立、互补的证据”：

$$
\max_{S:\operatorname{tokens}(S)\le B}   \frac{   \sum_{v\in S}r_q(v)   +\alpha,\mathrm{Support}(S)   +\beta,\mathrm{ReasoningCoverage}(S)   }{   \operatorname{tokens}(S)   }   -\gamma,\mathrm{Dependency}(S)   -\delta,\mathrm{Redundancy}(S)
$$

其中同一 provenance group 的十篇转载不能贡献十次支持分数，可以使用 max、饱和函数，或者先把来源簇收缩成 super-node。

## 真正能写进论文的贡献

1. 发现新失败模式：Correlated Evidence Amplification

   在语义图中，大量近重复文档形成的区域通常最稠密。错误新闻被转载 20 次后，普通 densest-subgraph retriever 可能比唯一 的一手正确来源更偏向错误答案。

2. 提出 Independent Evidence Density

   密度不再由边数量决定，而由独立来源覆盖、推理槽位覆盖和跨来源支持决定。这个目标比“相关性 + 连通性”更有论文辨识度。

3. 提供 duplication-invariance 性质

   如果新增文档只是已有 provenance group 的完全重复或改写，收缩后的检索图不变，因此检索结果不应改变。这个性质可以形式 化为定理，而多数 Top-K、PPR 和普通 dense subgraph 方法都不具备。

4. 构造 CopyBurst 压力测试

   给 HotpotQA、2WikiMultiHopQA 或 MuSiQue 中的 distractor 做 1/5/10/20 次复制和释义，分别复制正确证据、错误证据和无关 证据，测答案随重复次数的稳定性。

这是很好的 benchmark contribution，因为它直接验证方法的核心假设，而不只是再跑一次 QA accuracy。

## 冲突感知扩展

可以进一步挖掘两个 signed dense coalitions：

- (S^+)：支持某个候选结论的独立证据群

- (S^-)：反对该结论的独立证据群

- coalition 内部以支持边为主

- coalition 之间以矛盾边为主

  如果两个 coalition   的可靠性接近，系统不应该强行选择答案，而应该输出冲突说明或拒答。现有研究表明，多跳场景下模型很难定位冲突来源，MAGIC,   EMNLP Findings 2025   (https://aclanthology.org/2025.findings-emnlp.466/)；多文档数量增加即便总上下文长度不变，也可能使性能下降最多约   20%，说明“选择什么结构”确实比单纯增加上下文重要。EMNLP Findings 2025   (https://aclanthology.org/2025.findings-emnlp.1064/)

  不过我建议把 signed coalition 作为第二阶段扩展。第一版论文先聚焦“重复证据放大 + provenance-aware density”，问题会更集   中。

## Subgraph Matching 怎么加入

可以让问题先生成一个轻量 reasoning motif：

- chain：A -> B -> answer

- comparison：两个实体分别连接到可比较属性

- intersection：两个条件共同指向答案

- temporal：事件与时间约束

- support/refute：claim 与证据立场

  然后做：
  $$
  \text{MotifMatch}(M_q,S)   +\lambda\text{IndependentDensity}(S)
  $$
  但不能把 motif matching 本身作为主创新，因为 SimGRAG 已经做了 query-to-pattern 和 pattern-to-subgraph。你的区别必须   是：

> 匹配的是“推理角色与独立证据结构”，而不只是实体关系模式。

## 实验设计

数据集建议分三组：

- 多跳 QA：HotpotQA、2WikiMultiHopQA、MuSiQue

- KGQA：WebQSP、ComplexWebQuestions

- 冲突与鲁棒性：MAGIC、RAMDocs，加自建 CopyBurst

  基线至少包括：

- BM25、Contriever、ColBERT

- HippoRAG、KG²RAG

- G-Retriever、GRAG

- GNN-RAG、SubgraphRAG、SimGRAG

- 普通 densest subgraph、Steiner Tree、PPR

  除 EM/F1 外，必须报告：

- supporting-fact Recall/F1

- 每 1K token 的有效证据量

- provenance-group coverage

- 重复次数增长时的性能下降曲线

- conflict detection、ECE 或 selective risk

- 检索延迟和子图规模