# VerseMatrix Lab v0.5

**可运行的上下文匹配器、补句命令、已训练权重和实测记录已经交付。新匹配模型未达到开发资源门，统计基线继续保留。**

主模型的开发集诗族等权 R@16 为 **1.058%**，固定统计基线为 **4.425%**；差 **-3.367 个百分点**。这是已经用于检查点选择的开发结果，不是新的独立确认，不代表文学质量。全部原始结果和负结果均保留。

## 立即运行补句

在解压后的 `wordconv-lab` 目录运行（Python 3.10+，依赖 NumPy）：

```bash
python3 scripts/complete_span_v05.py --upper '空山新雨後' --suffix '晚來秋'
```

默认并列显示固定统计基线和实验匹配模型各自的前16个候选成句。无需重新训练。五言填5字上句和3字后缀；七言填7字上句和5字后缀，不带标点。文字保留原字形，不自动转换简繁或异体。

```bash
# 只使用已有统计方法；这条功能示例的首选是“天氣晚來秋”
python3 scripts/complete_span_v05.py --upper '空山新雨後' --suffix '晚來秋' --method statistical --limit 5

# 明确查看实验匹配模型，并输出JSON
python3 scripts/complete_span_v05.py --upper '江雨催寒色' --suffix '過遠村' --method matching --json
```

也可执行 `python3 -m pip install -e .`，然后使用 `wordconv-complete`。如果通过非可编辑方式安装包，需用 `--project-dir /path/to/wordconv-lab` 指明随包的数据和权重目录。默认实验权重为训练前已指定的主模型 seed0，加载三个种子共同选中的第800步；不选择最好种子，不等于三模型集成。输出的候选和分数用于来源原文匹配，不能直接当作合律认证或优美诗句评分。

固定4个展示输入及其全部输出：`configs/completion_examples_v0.5.json`、`results/completion_examples_v05.json`。首个五言示例与一条训练题的可见上下文完全相同；其余三个未发现相同训练题面。四例都只是功能展示，不是独立效果评测，训练重合情况逐条记录。

## 本轮算法

查询只含上句和下句可见后缀。共享12维字符向量经水平卷积与固定位置投影，得到16维查询向量；两个候选字按左右次序编码为16维候选向量。二者采用平滑 L2 归一化，固定温度0.1；把温度吸收到查询矩阵后：

**S = Q Eᵀ**

评分对固定句长词表一次覆盖全部候选。没有候选ID向量或独立候选输出偏置。每个训练查询使用全候选 softmax 交叉熵；所有原文不在词表的开发题仍保留并计零。`score_factors()` 提供 Q、E 与零偏置，可直接重算矩阵乘法。

已有声律矩阵和有限局部可加目标的精确动态规划继续保留。新的整体上下文检索分数并没有自动获得原DP的可加性保证。声律可行性、候选匹配、全篇组织和文学评价分别需要证据。

## 实际训练和开发结果

候选W固定：五言10,375项，七言1,504项；每次仍输出至多16项。统计资源来自原train的2,502首诗。梯度训练只使用W内的5,951道题，涉及2,375首诗、2,247个诗族。

正式完成4组×3种子，共 **12次训练、14,400次参数更新、460,800次题目呈现**。每模型1,200步、batch32；同seed各组共享同一任务安排。第400/800/1,200步都保存不可覆盖检查点。共有36份正式检查点，选择只引用其路径和SHA，不复制覆盖成一个通用“selected”文件。

主指标为所有2,188道开发题的三种子逐题命中均值，再在诗族内平均，最后对508个诗族等权。每组据此共同选一个训练步数，同分选早。

| 模型 | 共同选中步数 | seed0 | seed1 | seed2 | 三种子均值 |
|---|---:|---:|---:|---:|---:|
| 完整上下文，卷积宽度3（主模型） | 800 | 1.230% | 0.763% | 1.181% | 1.058% |
| 完整上下文，卷积宽度1 | 1200 | 1.329% | 1.230% | 0.861% | 1.140% |
| 只看上句，卷积宽度3 | 1200 | 1.206% | 1.206% | 1.230% | 1.214% |
| 只看后缀，卷积宽度3 | 400 | 0.960% | 0.591% | 0.640% | 0.730% |

固定对照：统计基线 **4.425%**；所有位置二字频次排序 **1.599%**；训练下句首二字频次排序 **2.067%**。所有神经模型本轮的开发观察值均低于这两个无上下文频次对照，不能报告为检索升级。

主模型使用正常上下文为 **1.058%**，换成其他同句长诗族的上下文为 **0.139%**，全部遮蔽为 **0.492%**。正确上下文相对错配的差为 **+0.919个百分点**；这个开发观察值提示模型捕捉到部分与任务相关的上下文信息，但仍低于预定1个百分点资源门，且远未赶上统计基线。这不是独立显著性或因果归因结论。

资源门三个条件均未通过：平均比统计基线高至少2个百分点；三个种子各自均高于基线；正常上下文比错配高至少1个百分点。本轮决定为 `DO_NOT_ALLOCATE_CONFIRMATION_FROM_THIS_DEV_RUN`，没有运行新的独立确认，也没有沿用旧确认集调参。

## 数据覆盖限制

开发集2,188题中，647题的原文曾作为训练下句前缀正例；250题的原文在W内但从未作为此前缀正例；1,291题在W外。W有8,401个候选从未作为该恢复任务的正例（五言7,233，七言1,168）。它们仍参与softmax分母，字符共享允许其他正例影响其表示，但没有相应完整片段的正例监督。这个分布限制是后续机制假设，不能未经新实验便称作失败的唯一原因。

## 核验与复现

```bash
python3 scripts/run_tests.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_matching_v05.py prepare --out results/matching_v05_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_matching_v05.py run --out results/matching_v05_replay
```

原正式结果目录不可重跑覆盖。新结果必须使用新的目录。精确浮点和SHA复现应使用记录的Python 3.12.14、NumPy 2.3.5与单线程设置。

核心作者之外的独立检查包含标量卷积/候选编码/平滑归一化/full-softmax公式及108个坐标中心差分。选中权重另有192个整词表矩阵、单句/批次、保存加载和部署案例核验。独立数据账本由Counter重建全部词面、计数与来源；实际成绩从逐题输出重算。最终回执见 `results/test_receipt.json`、`results/matching_v05/independent_review.json`、`results/matching_v05/data_schedule_review.json` 和 `results/matching_v05_engineering/numeric_audit/report.json`。

工程吞吐预检另有20次更新/640次呈现，seed172905；它不计入正式12模型，不参与选型。原始预检驱动、冻结文件、采样和20步权重保留在 `results/matching_v05_engineering/preflight/`。

## 文件入口

| 内容 | 路径 |
|---|---|
| 中文报告 | `docs/VerseMatrix_Progress.pdf` |
| 可训练矩阵匹配器 | `src/wordconv_lab/matching.py` |
| 用户补句API | `src/wordconv_lab/completion_v05.py` |
| 训练/开发执行器 | `scripts/run_matching_v05.py` |
| 配置与协议 | `configs/matching_v0.5.json`、`docs/MATCHING_v0.5_PROTOCOL.md` |
| 所有逐题输出及检查点 | `results/matching_v05/` |
| 方法独立评审 | `docs/MATCHING_v0.5_REVIEW.md` |
| 历史与交付完整性 | `results/history_preservation_v05.json`、`docs/DELIVERY_INTEGRITY_v0.4.md` |

## 既有证据和能力边界

v0.2声律文法、有限局部及全诗可加目标求解和机制实验保留；v0.3六次训练与未晋级判断、检查点异常原件及精确恢复记录保留；v0.4独立确认的未晋级结果及配额开发负结果保留。本轮开始前267个v0.4清单项目已逐项匹配，新交付采用先闭合、检查全部条目再原子发布文件名的封包方式。

本项目的源诗只经过八句×五/七字形态筛选，不是严格律诗金标；二字相邻片段也不等于经过人工分词的词。历史韵部参考仍需语境读音审核。原文恢复成功不等于写出了更好的诗，本轮没有文学盲评。

来源：原语料 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，固定提交 `b8594f81a89752241442f2ce267d6f66f96704ee`，相应来源清单与许可随数据保存。模型设计参照 [Kim 2014](https://aclanthology.org/D14-1181/) 的局部文本卷积和 [Karpukhin et al. 2020](https://aclanthology.org/2020.emnlp-main.550/) 的双编码器点积检索；没有复制其预训练模型或效果结论。
