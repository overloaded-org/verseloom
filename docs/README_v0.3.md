# VerseMatrix Lab / 诗矩验证

**版本0.3.0，2026-10-29。律诗矩阵组织算法与局部词卷积的可验证研究。**

算法主干已经实现：基本声律文法、有限模板/韵部枚举、局部可加评分与精确动态规划。本轮新增真实来源的二字片段复原、NumPy训练、同池比较，以及保留完整片段成员关系的DP接口。

**本轮结论：当前小型卷积没有通过原文复原先导门。** 主测试501个诗族、2120题，诗族等权top1为统计基线0.749%、宽3卷积三种子均值0.399%、宽1消融0.432%。这不构成文学质量评价；具体模型与训练配置的结果不能推广为对卷积的一般否定。

## 实际成果

| 项目 | 证据 |
|---|---|
| 声律与完整矩阵 | v0.2八个基本模板、全篇有限目标穷举核对保留 |
| 真实来源数据 | 10008条源记录→3579首八句等长文本→3382族；先分族再切分 |
| 实际训练 | 3种子×2结构；每次400个optimizer updates、12800次任务曝光 |
| 目标可达性 | test原文418/2120在训练片段表，74/2120进入同一top16池 |
| 训练后精确求解 | 144次片段约束DP与原始候选池枚举最优一致 |
| 读音参考关联 | 1414个训练片段关联106部原字形参考，全部仍需语境审核 |
| DEV候选诊断 | 训练全句相邻二字min2：dev词表覆盖19.47%→41.00%，尚非检索/测试改善 |
| 风险测试 | 116项通过，见实际回执 |

当前报告是 `docs/VerseMatrix_Progress.pdf`。详细协议见 [docs/RESTORATION_v0.3_PROTOCOL.md](docs/RESTORATION_v0.3_PROTOCOL.md)，结果与决策见 [docs/STATUS.md](docs/STATUS.md)。形式化与复杂度见 [docs/ALGORITHM_v0.2.md](docs/ALGORITHM_v0.2.md)。

## 复现

运行核心仅需Python 3.10+与NumPy。使用新输出目录保护既有证据。

```bash
python -m pip install numpy
python scripts/run_tests.py
python scripts/build_corpus_v03.py --check
python scripts/audit_candidate_readings_v03.py --check
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_restoration_v03.py --out results/restoration_replay
```

Windows可先 `python -m pip install -e .`，再按环境设置单线程BLAS并省略 `PYTHONPATH=src`。实际环境为Python3.12.14、NumPy2.3.5。PDF构建另需ReportLab与Noto Sans SC静态字体。

## 三种证据要区分

- **求解正确性**：在声明域与目标内，DP是否达到枚举最优。已经验证，新增二字片段约束不扩大原池。
- **原文复原能力**：是否找回固定来源中的原二字片段。本轮已训练和留出评价，当前卷积未过门。
- **文学质量**：搭配、对仗、篇章与意境是否更好。尚未开展独立人工盲评。

八句等长形状没有逐首认证为律诗；固定二字跨度不是已经完成的古汉语分词；真实原文也不是其他措辞劣于它的金标。

## 本轮实验契约

字符表、候选片段和统计模型只来自train。模型仅见上句、下句可见后缀、句长与mask。作者、标题、target及完整原始下句不进入候选生成或特征。dev/test答案不注入候选池，未覆盖题保留并计0。

主池由双向trigram及同列统计提议前16个，所有系统重排同一池；结果条件于该提议机制。宽1是横向局部性消融，仍保留上下句交互，不当作同感受野注意力强对照。

每种子两结构共享训练候选与schedule。checkpoint只用全dev诗族等权top1选择，同分留较早版本。三种子先平均，再按族等权配对bootstrap；区间条件于这三个种子。正式H-C人工偏好门及同感受野注意力比较尚未完成。

## 代码入口

| 文件 | 职责 |
|---|---|
| `prosody.py` | 四基句、完整模板、窄规格声调审核 |
| `matrix.py` | 历史FIFO/priority通用搜索与审计 |
| `factor_dp.py` | 宽3可加局部因子与精确链式DP |
| `verse_solver.py` | 严格基本模板、已消歧读音和四联可加目标的八句求解 |
| `corpus.py` | 源数据、表面诗族、稳定切分和训练词表 |
| `learned_conv.py` | 批量前向、手工反传、Adam、checkpoint保存恢复 |
| `restoration.py` | train-only统计提议、允许列表题面和配对指标 |
| `span_solver.py` | 首二字片段成员约束，禁止池外交叉组合 |

以上模块均在 `src/wordconv_lab/`。历史matrix.organize默认签名仍为priority以保留冻结实验身份；通用对照应显式传strategy。priority在v0.2未过晋级门。

## 证据目录

- `results/restoration_v03/`：配置冻结、候选池、实际训练曝光、模型权重、dev选择、正式test预测与144项DP核对。
- `results/restoration_v03_recovery/`：一次检查点完整性异常的原件及按冻结schedule恢复的证据。
- `results/dev_candidate_diagnostic_v03/`：四个train候选来源策略的DEV-only覆盖诊断，未回改正式实验。
- `data/corpus_v03/`：固定源、诗族、任务、候选词表与许可。
- `data/reading_candidates_v03/`：训练片段到音韵参考的候选关联，不作为语境金标。
- `data/reference_v02/`：106部固定参考、校勘议题及原字形索引。
- `results/mechanism_v02/`、`results/discovery_v02_reviewed/`、`results/full_verse_v02/`：历史机制与精确性证据。

完整性异常已被SHA检查拦截：一个文件为step100而冻结选择step200。按原始schedule重放后恢复了完全相同的预期文件SHA，再继续评测，配置和选择没有改变；根因未确定。详见 [docs/INTEGRITY_v0.3.md](docs/INTEGRITY_v0.3.md)。

## 当前决策

保留精确组织算法；统计基线继续作为客观复原比较参照，当前CNN不升为主评分器。下一轮候选来源优先从训练文本全部行内相邻二字取min2，但词表扩大约8.4倍，须在dev另查检索召回及预算，不能把41%的词表覆盖当作准确率。本轮test已暴露，后续确认使用新的独立留出。

真实读音需要具体语境依据。所有新关联仍为requires_context_review；当前完整八句接口的verified是调用者断言，不是程序自动校勘结论。

第三方文本与现代标注保留各自署名和许可；项目代码未替用户决定最终开源许可。卷积、动态规划和自动诗歌生成均有研究先例，本项目不作方法首创或诗人认知机制声明。
