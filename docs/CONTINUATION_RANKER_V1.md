# 语料接续学习 v1：学习评分，不预填内容

本轮在 `induced.html` 的“种子与研究设置 → 跨句评分”增加 **语料学得接续（实验）**。原来的固定关系评分仍是默认值；新模型没有被当作文学生成已经晋级。模式改变不会把旧诗稿伪称为新模式结果。手机的折叠设置、长证据换行、触控布局与近期避重保留。

## 方法

内容词段、接续图与候选句仍来自原语料自组织模型，不读取人工题材词表、主谓宾表、事件脚本或必选句。此次不改变声律、字音候选、词表、候选池、原语料或测试分割。

在8个通用统计量上学习线性评分：同诗关联、正向下一句关联、中联跨度关联、反向关联、分段跨度吻合、相邻句字符重合、多字词段重合、接续边覆盖。统计量的选择是工程假设；它们不是人工定义的战争/爱情/山水标签。参数用真实语料顺序与跨诗错配对拟合，不提供“应当先写战争再写离愁”的路线。

使用成对logistic排序损失和L2正则，600次固定全批次更新。方法与[成对排序学习](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)同属排序目标思路；本版只有线性统计评分，不是RankNet神经网络复现。已有[TopWORDS-Poetry](https://aclanthology.org/2023.emnlp-main.205/)说明无预给内容词表的古诗词发现是已有研究方向；本项目没有复现或借用其质量结论。

运行时对线性分数做有界变换，作为整诗搜索中的一项偏好，不解释为“好诗概率”。模型文件约1.3KB，仅含数值、特征名称、来源身份与训练设置，不存诗句。模型与当前语料身份不符时拒绝运行；缺失资源不偷偷退回旧模式。

## 数据与评测

从既有 `train_candidate` 按固定散列取五言、七言各1,024首，一诗族最多一首。2,048首产生8,192个上下文、32,768对排序比较。目标位置为各联下句（0基序号1/3/5/7）。每个训练目标配4个不同诗族的干扰句。

开发诊断使用全部1,464首 `dev_reserve`，共5,856题；每题真实下一句加15个干扰句，优先匹配句长、目标句位与可识别韵部。1005个开发查询的相应韵部池不足，补用相同句长和句位的干扰句；这一回退不隐藏。测试保留集、隔离集在JSON解码前跳过，未用于拟合。源码顺序是自动监督信号，不是文学好坏标签。

准确率按诗族等权计算，分数相同用均匀随机打破平局的期望值计分，不能因原文放在数组第0位而得分。

| 16选1接续识别 | 五言 | 七言 |
|---|---:|---:|
| 现有3项固定关系评分 | 16.8853% | 19.4355% |
| 8项统计量的学习评分 | 18.1786% | 21.5728% |
| 候选不变，只把前文换成另一诗族 | 6.2051% | 6.3756% |
| 全候选同分对照 | 6.2500% | 6.2500% |

联合诗族等权提升1.6483个百分点，2000次配对诗族重采样95%区间为[1.0542,2.1859]个百分点。这是开发集证据，不是独立终测。改进同时包含新增统计量和学习权重，尚未分离两者贡献。上下文干预说明该任务中的输入依赖，不证明现实因果或文学叙事能力。

## 接入生成后的完整结果

固定种子 `[0,1,2,3,42,43,44,45]`，五言/七言 × 原关系评分/学习评分，共32次；variation=0.8，beam=16，不带近期历史。32次均成篇；16组配对的单句候选池指纹全部相同。五言仅2/8组改变输出，七言也只有2/8，合计4/16；没有提高权重或挑换种子来扩大这个数字。

学习评分每种诗体8个种子得到8首不同诗稿；这只是样本描述，不保证任意种子不重复。每组末次结果精确复现。单独Python检查器核对32首、1536个字位、128条学习评分记录，以及词项身份、平仄/韵脚/重复限制和整诗累计分数，全部通过。七言初次本地批跑受到执行时限中断，保留7条完成记录后按原设置续跑；未改变模型、参数或种子。CI从零重跑全部32次。

**接续识别改善没有自动转化成明显的整诗变化，更不等于诗更好。** 当前候选生成仍主要看句内接续，句间信号仅在后面的整诗选择中起作用。下一步应检验上下文参与候选构造，而非加手写故事框架。

## 验证与边界

7项Python训练/分割/平局测试、8项Node特征/参数/生成/复现/界面事件测试通过。64个固定词段跨度样本由Python与JS两种实现核对，误差低于1e-12；重新训练得到完全相同的浏览器模型字节。跨平台重建比较系数时容许1e-9数值误差，不重新改写冻结权重。既有175项Python回归在本地重新通过；其他现有网页测试以本次Pages CI为准。

界面事件测试不等于真实浏览器排版测试。本轮保持已有移动样式，没有新增iOS Safari真机认证；上线核验覆盖真实HTTP资源的散列及下载引擎执行。

仍有明确限制：
- 原始文本仅通过诗体形状筛选；没有独立文学评分或专家工对金标。
- 错配句可能是合理的另一种续法，原文恢复不是文学质量定义。
- 训练图包含拟合排序器所用的训练诗，训练损失不是留出效果；报告只用开发集衡量区分能力。
- 训练目标是各联下句；应用于其他句位以及程序生成的非原诗句，是外推试验，尚未验证同等效果。
- 原诗可重复词段，而生成器限制重复；这种分布差异可能削弱模型在整诗搜索中的作用。
- 历史读音仍保留候选，不能因评分改善就宣称语境读音已消歧。

## 重建

先按现有README构建bootstrap和 `web/induced-data.js`。然后：

```sh
OPENBLAS_NUM_THREADS=1 python scripts/train_continuation_ranker.py \
  --corpus output/phrase-input/run/candidate_poems.jsonl --output output/continuation-v1
python scripts/check_continuation_context.py \
  --corpus output/phrase-input/run/candidate_poems.jsonl \
  --experiment output/continuation-v1 --output output/continuation-v1/context-control.json
python -m unittest discover -s tests_continuation -v
CONTINUATION_FIXTURES=output/continuation-v1/feature-fixtures.json \
  node --test tests_continuation/*.test.cjs
node scripts/evaluate_continuation.cjs output/continuation-five 5
node scripts/evaluate_continuation.cjs output/continuation-seven 7
python scripts/audit_continuation_generation.py --ranker output/continuation-v1/ranker.json \
  --input output/continuation-five/outputs.jsonl output/continuation-seven/outputs.jsonl \
  --output output/continuation-v1/generation-audit.json
```

开发逐题候选ID、原句散列、各方法分数、全部生成输出和失败记录保留在独立Actions构件；不得只展示佳作。新增训练/检查脚本不写回旧语料、韵表、已冻结实验或默认评分。
