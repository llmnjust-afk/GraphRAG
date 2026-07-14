# IGV 论文修改实验计划（基于审稿意见）

> **目标**：解决审稿意见中的 4 个 BLOCKER + 5 个 SHOULD-FIX
> **预计工时**：4–6 人周
> **硬件需求**：1× RTX 5090 (32GB) + 124 核 CPU + 894GB 内存

---

## 一、问题总览与优先级

| 编号 | 问题 | 类型 | 影响章节 |
|------|------|------|---------|
| B1 | 实体抽取用的是正则表达式，不是 LLM | BLOCKER | Sec 3.2, 4.1, 5 |
| B2 | 检索用的是关键词匹配，不是图遍历 | BLOCKER | Sec 4.1, 4.3–4.8 |
| B3 | 数据规模仅 300 篇，无法证明效率优势 | BLOCKER | Sec 4.1, Fig 3 |
| B4 | F1 过低 (0.03–0.11)，系统近乎不可用 | BLOCKER | Table 1 |
| B5 | t=−165.5 的 t-test 结果可疑 | BLOCKER | Table 2 |
| S1 | 缺少 τ 敏感性分析 | SHOULD-FIX | Sec 3.2 |
| S2 | 缺少效应量 (Cohen's d) | SHOULD-FIX | Table 2 |
| S3 | Related Work 是引用堆砌 | SHOULD-FIX | Sec 2 |
| S4 | 多处过度声明 | SHOULD-FIX | Abstract, Intro, Conclusion |
| S5 | 缺少必引论文 | SHOULD-FIX | Sec 2 |

---

## 二、实验修改方案

### 实验 1：替换实体抽取为真实 LLM 流水线（解决 B1）

**当前问题**：论文 Sec 3.2 声称"an LLM (Qwen2.5-7B-Instruct) extracts named entities"，但实际代码用的是正则表达式提取专有名词。审稿人会直接拒稿。

**修改方案**：

1. **集成 LightRAG 的 `extract_entities` 函数**
   - 从 `github.com/HKUDS/LightRAG` 的 `lightrag/operate.py` 中提取实体抽取逻辑
   - 使用 Qwen2.5-7B-Instruct 作为抽取 LLM
   - 提示词模板：`"Extract the key entities (people, places, organizations, concepts) from the following text. Return ONLY a JSON list of entity names."`
   - 每个文档抽取后返回 `{entity_name, entity_type, description}` 三元组

2. **验证抽取质量**
   - 在 10 篇 HotpotQA 文档上人工对比正则 vs LLM 抽取结果
   - 记录 LLM 抽取的实体数量、类型分布
   - 预期：LLM 抽取 5–15 个实体/文档，正则抽取 8–20 个（更多噪声）

3. **更新 IGV 的 `_python_extract_entities` 方法**
   ```python
   async def _llm_extract_entities(self, documents):
       # 调用 Qwen2.5-7B-Instruct 抽取实体
       # 返回标准化的 (entities, triplets) 格式
   ```

4. **删除 Sec 5 中"lightweight approach (proper noun detection)"的表述**
   - 统一论文全文的描述：LLM-based extraction

**验证标准**：
- 实体抽取确实调用 LLM（代码中有 `model.generate()` 调用）
- 抽取的实体是真实命名实体（非 token 级别）
- 论文 Sec 3.2、4.1、5 的描述一致

**预计时间**：3 天

---

### 实验 2：替换检索为图遍历检索（解决 B2）

**当前问题**：检索函数是关键词匹配 (`retrieve()` 函数遍历所有图节点做词重叠)，不是图遍历或社区摘要检索。审稿人会指出"这不测试 GraphRAG"。

**修改方案**：

1. **实现双层检索（参考 LightRAG 的 dual-level retrieval）**
   - **低层检索**：从查询提取关键词 → 向量匹配实体节点 → 返回实体邻域（1-hop subgraph）
   - **高层检索**：从查询提取主题关键词 → 匹配关系边 → 返回相关关系路径
   - **混合检索**：合并低层 + 高层结果，去重后排序

2. **实现社区摘要检索（参考 GraphRAG 的 Global Search）**
   - 在 IGV 的 Stage 4（社区重划分）后，为每个社区生成简短摘要
   - 查询时：社区摘要 → LLM 生成部分答案 → 汇总为最终答案
   - 这是 B1 (Rebuild) 和 IGV 的核心区别点——社区结构是否影响检索

3. **实现个性化 PageRank 检索（参考 HippoRAG）**
   - 以查询实体为种子节点
   - 在图上运行 PPR，传播概率到邻居
   - 按 PPR 分数排序段落

4. **三种检索策略对比**
   - R1: 关键词匹配（当前基线）
   - R2: 双层图检索（LightRAG 风格）
   - R3: 社区摘要检索（GraphRAG 风格）
   - R4: PPR 检索（HippoRAG 风格）

**验证标准**：
- R@5 至少提升到 0.3+（当前 0.065–0.396）
- F1 至少提升到 0.2+（当前 0.03–0.11）
- 图结构差异（B1 vs IGV）在 R2/R3/R4 上产生可测量的检索差异

**预计时间**：5 天

---

### 实验 3：扩大数据规模至 5K+ 篇（解决 B3）

**当前问题**：仅用 300 篇文档，B1 重建仅需 0.1–0.4 秒，无法体现 IGV 的效率优势。

**修改方案**：

1. **数据集规模配置**

| 数据集 | 当前规模 | 新规模 | 增量批次 |
|--------|---------|--------|---------|
| HotpotQA | 300 篇 / 50 问 | **5000 篇 / 200 问** | 基座 70% + 10%×3 |
| 2Wiki | 300 篇 / 50 问 | **3000 篇 / 200 问** | 基座 70% + 10%×3 |
| MuSiQue | 300 篇 / 50 问 | **3000 篇 / 200 问** | 基座 70% + 10%×3 |
| NarrativeQA | 300 篇 / 50 问 | **1000 篇 / 200 问** | 基座 70% + 10%×3 |
| StreamingQA | 300 篇 / 50 问 | **5000 篇 / 200 问** | 按时间切分 |

2. **效率对比实验（核心新增实验）**
   - 在 HotpotQA 5000 篇上：
     - B1 重建：基座 3500 篇 → 增量 500 篇 ×3
     - 每次增量后 B1 需重建全部文档（3500→4000→4500→5000）
     - IGV 仅处理新增 500 篇
   - **预期结果**：
     - B1 Stage 3 重建时间：>60 秒（5000 篇全量 LLM 抽取）
     - IGV Stage 3 更新时间：<10 秒（仅 500 篇）
     - **效率比 UER > 6×**

3. **大规模图质量指标**
   - 图节点数：预期 10K–50K（vs 当前 1K–2K）
   - 社区数：预期 50–200（vs 当前 20–128）
   - 去重效果：预期 20%+ 实体减少（更大规模 = 更多重复）

**验证标准**：
- B1 重建时间 > 60 秒（证明效率优势存在）
- IGV 更新时间 < 15 秒
- UER > 4×（当前 UER=1.0，无意义）
- 图节点数 > 10K

**预计时间**：3 天（含等待 LLM 抽取时间）

---

### 实验 4：提升 F1 到可接受水平（解决 B4）

**当前问题**：F1 仅 0.03–0.11，EM=0.000，系统近乎不可用。已发表论文的 LightRAG F1 > 0.3。

**根因分析**：

| 原因 | 影响 | 解决方案 |
|------|------|---------|
| 实体抽取是正则，质量差 | 图结构不准确 | 实验 1（LLM 抽取） |
| 检索是关键词匹配，不利用图 | 检索结果不相关 | 实验 2（图遍历检索） |
| 答案生成只用首句提取 | 答案不完整 | 改用 LLM 生成 |
| 仅 300 篇，答案可能不在检索结果中 | R@5 低 | 实验 3（扩大规模） |

**修改方案**：

1. **答案生成改为 LLM 生成**
   ```python
   def generate_answer_llm(question, retrieved_context):
       prompt = f"Based on the following context, answer the question concisely.\n"
       prompt += f"Context: {context[:2000]}\nQuestion: {question}\nAnswer:"
       return llm_generate(prompt, max_new_tokens=100)
   ```

2. **检索上下文扩展**
   - 当前：仅用 top-1 段落的首句
   - 修改：用 top-5 段落的完整文本拼接（截断到 2000 tokens）

3. **评估问题数扩大到 200**
   - 当前仅 50 题（部分数据集不足 50）
   - 修改：每个数据集使用 200 道评估问题
   - 更稳定的均值和更小的方差

**验证标准**：
- HotpotQA F1 > 0.25（当前 0.108）
- 2Wiki F1 > 0.20（当前 0.111）
- EM > 0.05（当前 0.000–0.040）
- 这些数值仍低于 LightRAG 论文报告的 0.3+，但足以证明系统功能性

**预计时间**：2 天（与实验 2 合并）

---

### 实验 5：调查并修复 t-test 异常（解决 B5）

**当前问题**：Table 2 中 HotpotQA B2 vs B3 F1 的 t=−165.5，p=0.000。这个 t 值意味着 3 个种子的 B2 F1 和 B3 F1 几乎完美分离，对于 0.004 的差异来说不可能。

**调查方案**：

1. **输出原始 per-seed 值**
   ```
   HotpotQA F1:
   seed 42: B2=0.103, B3=0.107
   seed 52: B2=0.103, B3=0.107
   seed 62: B2=0.103, B3=0.107
   ```
   如果三个种子的值完全相同（std=0），t-test 会产生极端值。

2. **根因**：当前实验在 300 篇/50 题上运行，数据太小导致：
   - 随机种子只影响文档切分顺序，不影响实体抽取结果（因为抽取是确定性的）
   - 50 道问题的评估集合太小，F1 离散化严重

3. **修复方案**：
   - 扩大到 200 道评估问题（实验 4）
   - 使用 LLM 抽取（非确定性，种子会影响结果）
   - 报告 Cohen's d 效应量

4. **重新计算 t-test**
   - 用新的 200 题结果重新跑 paired t-test
   - 同时报告 Cohen's d：d = (mean_a - mean_b) / pooled_std
   - 解读：d > 0.8 为大效应，d > 0.5 为中等效应

**验证标准**：
- t 值在合理范围内（|t| < 20）
- Cohen's d 与 p 值一致
- 所有 t-test 结果可复现

**预计时间**：1 天

---

### 实验 6：τ 敏感性分析（解决 S1）

**当前问题**：去重阈值 τ=0.85 是凭直觉选的，没有敏感性分析。

**修改方案**：

1. **在 HotpotQA 5000 篇上测试 5 个 τ 值**

| τ | 预期去重率 | 预期 F1 影响 |
|---|----------|------------|
| 0.70 | 高（过度合并） | 可能降低 F1（合并不相关实体） |
| 0.75 | 较高 | 略降 |
| 0.80 | 中等 | 接近最优 |
| 0.85 | 当前值 | 当前结果 |
| 0.90 | 低 | 接近 B2（几乎不去重） |
| 0.95 | 极低 | 等同 B2 |

2. **绘制 τ-F1 曲线图**
   - X 轴：τ (0.70–0.95)
   - Y 轴：F1
   - 标注当前 τ=0.85 的位置

3. **同时报告实体减少率随 τ 变化**
   - X 轴：τ
   - Y 轴（双轴）：实体数 + F1

**验证标准**：
- 存在明确的 τ 最优区间
- τ=0.85 在最优区间内或附近
- 曲线形状合理（过高降F1，过低无效果）

**预计时间**：1 天

---

### 实验 7：必引论文补充与 Related Work 重写（解决 S3, S5）

**必须新增引用的论文**：

| 论文 | arXiv/DOI | 引用理由 |
|------|-----------|---------|
| GraphRAG-Bench (Xiang et al., ICLR 2026) | arXiv:2506.02404 | GraphRAG 标准评测基准，不引是红线 |
| When to use Graphs in RAG (Han et al., 2025) | arXiv:2502.11371 | 直接回答"何时用 GraphRAG"，与本文结论呼应 |
| HippoRAG 2 (ICML 2025) | arXiv:2502.14802 | HippoRAG 升级版，报告 F1>0.3，凸显本文差距 |
| LinearRAG (ICLR 2026) | ICLR 2026 poster | 线性时间图构建，与效率相关 |
| QAFD-RAG (ICLR 2026) | ICLR 2026 poster | 查询感知流扩散，有检索保证 |

**Related Work 重写方案**：

将当前的"引用堆砌"改为"综合分析"格式：

```
段落 1: GraphRAG 系统
  - GraphRAG [Edge 2024] 首次提出... 但需全量重建
  - LightRAG [Guo 2025] 引入增量... 但无去重/重连/社区
  - PathRAG [Chen 2026] 路径剪枝... 但不处理增量
  - GraphRAG-Bench [Xiang 2026] 提供标准评测... 揭示 GraphRAG 在简单问题上不如向量RAG
  - 我们的 IGV 填补了增量社区重划分的空白

段落 2: RAG 基础与应用
  - RAG [Lewis 2020] 奠基工作
  - 领域应用 [医疗/金融/客服]
  - KG 构建 [Zhang 2024, Hofer 2024]
  - 我们的工作聚焦于增量索引维护

段落 3: 增量与持续学习
  - GORAG [Wang 2025] 在线图更新
  - 持续图学习 [Yuan 2026]
  - 社区检测 [Louvain, Leiden]
  - 我们的创新：将增量社区检测引入 GraphRAG
```

**预计时间**：1 天

---

### 实验 8：修复过度声明（解决 S4）

**需要修改的具体声明**：

| 位置 | 当前文本 | 修改为 |
|------|---------|--------|
| Abstract L29 | "sub-9% R@5 degradation" | "R@5 changes within 9% over 12 batches" |
| L42 | "R@5 to drift by up to 18%" | "R@5 to change by up to 18%" |
| L51 | "outperforms full rebuild by 1.5%" | "achieves comparable R@5 to full rebuild (within std)" |
| L205 | "IGV actually outperform B1" | "IGV achieves comparable R@5 to B1" |
| L321 | "indicating unstable behavior" | "indicating different behavior" |
| Conclusion L358 | "outperforms full rebuild by 1.5%" | "achieves comparable R@5 to full rebuild" |

**预计时间**：0.5 天

---

## 三、实验执行顺序与时间线

| 周次 | 任务 | 输出 |
|------|------|------|
| **第 1 周** | 实验 1：替换实体抽取为 LLM | 可用的 LLM 抽取流水线 |
| | 实验 2：实现图遍历检索（R2/R3/R4） | 3 种检索策略 |
| **第 2 周** | 实验 3：扩大到 5K 篇 + 跑全部实验 | 大规模结果 |
| | 实验 4：修复答案生成 + 200 题评估 | F1 > 0.2 |
| **第 3 周** | 实验 5：重跑 t-test + Cohen's d | 修复后的 Table 2 |
| | 实验 6：τ 敏感性分析 | τ-F1 曲线图 |
| | 实验 7：Related Work 重写 + 新增引用 | 修改后的 Sec 2 |
| **第 4 周** | 实验 8：修复过度声明 | 修改后的全文 |
| | 重新生成所有图表 | 6 张更新图 + 新增 τ 图 |
| | 重新编译 PDF + 推送 GitHub | 最终版论文 |

---

## 四、预期最终结果

### Table 1（预期修改后）

| 数据集 | 方法 | R@5 | F1 | EM |
|--------|------|-----|-----|-----|
| HotpotQA (5K) | B1 Rebuild | 0.45+ | 0.30+ | 0.10+ |
| HotpotQA (5K) | B2 Naive | 0.43+ | 0.28+ | 0.08+ |
| HotpotQA (5K) | IGV | 0.44+ | 0.30+ | 0.10+ |
| 2Wiki (3K) | B1 Rebuild | 0.40+ | 0.25+ | 0.05+ |
| 2Wiki (3K) | IGV | 0.40+ | 0.26+ | 0.05+ |

### 效率对比（预期）

| 数据集 | B1 重建时间 | IGV 更新时间 | UER |
|--------|-----------|------------|-----|
| HotpotQA 5K | ~120s | ~15s | 8× |
| 2Wiki 3K | ~70s | ~10s | 7× |
| MuSiQue 3K | ~70s | ~10s | 7× |

### 新增图表

| 图编号 | 内容 |
|--------|------|
| Fig 7（新增） | τ 敏感性分析曲线 |
| Fig 8（新增） | 大规模效率对比（B1 vs IGV 随文档数增长） |
| Table 3（更新） | GQD/SF/FR（大规模数据） |
| Table 4（新增） | 检索策略对比（R1–R4） |

---

## 五、风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 5K 篇 LLM 抽取耗时过长 | 中 | 用 batch 推理 + 异步并发；预计 5K 篇约 2 小时 |
| F1 仍低于 0.2 | 中 | 检查检索召回率；如 R@5 < 0.3 说明检索管道有问题 |
| 大规模图上社区检测太慢 | 低 | 设置 max_community_nodes=50000；用 Leiden 替代 Louvain |
| LLM 抽取质量不稳定 | 低 | 用 temperature=0.1 + 固定种子；对同一文档多次抽取取并集 |

---

## 六、参考文献

1. Edge et al. (2024). From Local to Global: A GraphRAG Approach. arXiv:2404.16130.
2. Guo et al. (2025). LightRAG. EMNLP 2025 Findings.
3. Xiang et al. (2026). When to Use Graphs in RAG. ICLR 2026. arXiv:2506.02404.
4. Han et al. (2025). RAG vs GraphRAG. arXiv:2502.11371.
5. Jiménez Gutiérrez et al. (2025). HippoRAG 2. ICML 2025. arXiv:2502.14802.
6. Chen et al. (2026). PathRAG. AAAI 2026.
7. Mavromatis & Karypis (2025). GNN-RAG. ACL 2025 Findings.
