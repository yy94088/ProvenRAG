# ProvenRAG MVP

这个目录实现了论文设想中最小、最关键的可证伪实验：**Correlated Evidence
Amplification** 与 provenance 收缩带来的 **duplication-invariance**。

当前 MVP 包含：

- 基于 canonical provenance metadata 与文本近重复度的来源分组；
- 将同源证据收缩成 super-node，组相关性采用 `max` 聚合；
- token 预算约束下的 Independent Evidence Density 贪心选择器；
- 会重复计算近重复支持边的普通 density 基线；
- 错误证据复制 1/5/10/20 次的 CopyBurst 合成压力测试；
- provenance 分组、基线退化和 duplication-invariance 的单元测试。

## 运行

项目没有第三方运行时依赖，Python 3.10+ 即可：

```powershell
python -m unittest discover -s tests -v
python -m provenrag
python -m provenrag --json
```

预期现象：普通 density 基线在错误证据出现 5 个副本后，30-token 预算被三个副本占满；
provenance-aware 方法在 1/5/10/20 个副本下选择相同的三个来源组，并保持 100% 的
supporting-fact recall。这里的 `proxy_correct` 是按被选中的独立正确/错误来源多数计算的检索代理指标，
它不是最终 QA 的 EM/F1。

## 代码边界

这是机制可行性验证，不是完整 RAG。下一阶段需要把 `Evidence` 接到 HotpotQA/2WikiMultiHopQA
数据加载器与真实 BM25/dense retrieval，再用 MinHash/embedding/NLI 替换当前的保守词法近重复度，
最后接入回答模型测 EM/F1 与引用正确率。
