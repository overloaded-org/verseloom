# VerseMatrix Lab / 诗矩验证

**版本 0.2.0，2026-10-29。主项目：律诗的矩阵组织算法；局部评分模块：词卷积。**

本轮找到并实现的算法主干是：**基本声律文法 → 明确读音与有限候选 → 外层模板/韵部枚举 → 内层局部因子动态规划**。它在明确分解条件下求内部数学目标的全局最优；尚未训练诗歌质量模型，尚未开展真实诗歌盲评。

## 直接看结果

| 项目 | 实际证据 | 决策 |
|---|---|---|
| 基本声律文法 | 五言、七言各4种基本整首模板；独立模板测试 | 保留，明确变格覆盖不足 |
| 局部词卷积精确求解 | 5/7列×2种子，与原模型1024/16384种组合穷举最优一致 | 可分解任务使用DP |
| 八句完整矩阵 | 有限模板×韵部；四联可加目标；全篇独立穷举核对 | 条件成立时使用精确组合 |
| 得分优先搜索 | 240测试题主预算下差距下降11.45%，局部陷阱退步0.05169 | 未过20%收益与0.05退步门槛，不晋级 |
| 韵部参考 | 新固定来源106部；7663字；1183字跨部；573字跨平仄 | 候选参考，不能自动充当语境金标 |
| 文学质量 | 未训练；未盲评 | 不作效果提升结论 |

算法定义及证明见 [docs/ALGORITHM_v0.2.md](docs/ALGORITHM_v0.2.md)，当前状态见 [docs/STATUS.md](docs/STATUS.md)。v0.1的设计协议 [docs/PROTOCOL.md](docs/PROTOCOL.md) 保留为历史设计，已执行项以v0.2状态和回执为准。

## 算法范围

`prosody.py` 根据第二字平仄、句末平仄及粘对关系生成基本型。`common_prefix`是一个有出处但不完整的放宽规格；不匹配只能说明范围外，不能宣布古诗失律。输出只检查声调层，不认证韵部、对仗或意义。

`factor_dp.py` 将宽3局部卷积和同列上下句组合评分分解成三列局部因子。特征必须是字、位置、固定任务条件的函数；不能随待生成整句改变。DP保留最后两列的成对候选，求给定有限域最优。Python浮点结果按报告中的数值容差核对。

`verse_solver.py` 在调用者给定的完整基本模板与韵部集合上枚举，严格检查候选读音，再求四联分数之和的最优。它没有实现跨联语义、全篇禁重字或任意词跨度约束。这些限制若未纳入求解，不能在事后检查失败后仍声称已找到相应约束下的最优。

令D为一列上下句候选的笛卡尔积大小，局部DP转移量为O(LD³)，前沿最多O(D²)。当前代码还缓存所有因子、保存完整路径；总内存并非O(D²)。如果上下各k个候选，D=k²。

## 快速复现

需要Python 3.10+和NumPy；交付运行环境见 `results/test_receipt.json`。PDF构建另需ReportLab及Noto Sans SC静态字体。

```bash
python -m pip install numpy
python scripts/run_tests.py
python scripts/build_rhyme_reference.py --check

# 以下使用新目录，保护已经保留的证据。
PYTHONPATH=src python scripts/run_discovery_v02.py --out results/discovery_replay
PYTHONPATH=src python scripts/run_full_verse_v02.py --out results/full_verse_replay
PYTHONPATH=src python scripts/run_mechanism_suite.py --out results/mechanism_replay
```

Windows可先 `python -m pip install -e .`，随后省略 `PYTHONPATH=src`。以上实验中的评分权重均未训练，候选输入均明确标记为合成数据。

## 保留的v0.1工程能力

- 同一全矩阵状态图上的单格和显式联合替换、FIFO/得分优先搜索。
- 尝试与评分双预算、最佳合法结果保留、事件哈希链和确定性重新执行。
- 与受限搜索隔离的穷举oracle；全域最优与起点可达最优分开报告。
- 同一记录中的声调/韵部、未知与多音待复核；候选池与约束词典身份校验。
- 按给定源诗/重复族的分割审计；匿名评审表的内容身份绑定与配对bootstrap。
- 1000首诗库分片与旧105部表的冻结快照及可复算数据审计。

历史 `matrix.organize()` 的默认签名仍为priority，保持冻结机制实验的源码身份。**调用通用比较时显式传 `strategy="fifo"`**；本轮选择的精确求解走新接口，不依赖旧默认值。当前配置说明见 `configs/algorithm_v0.2.json`。

## 数据与证据

| 位置 | 内容 |
|---|---|
| `results/discovery_v02_reviewed/` | 数值边界修复后的4组局部精确核对 |
| `results/discovery_v02/` | 第一次运行保留，见其POST_REVIEW说明 |
| `results/full_verse_v02/` | 八句全矩阵实验、冻结配置和原模型穷举 |
| `results/mechanism_v02/` | 300题、2400次运行、可达性oracle和压缩轨迹 |
| `data/reference_v02/` | 固定106部来源、索引、补充层、校勘议题与许可 |
| `data/worked_examples_v02.json` | 6首文献例诗，仅2首有公开逐字声调标注 |
| `results/test_receipt.json` | 实际风险测试数、环境和源码校验值 |
| `docs/` | 当前报告、算法说明及历史协议 |

106部主表仍有明确疑点，全部记录为 `source_candidate_needs_context`。不自动合并繁简、异体，不把多音集合按格律挑读；没有读音证据就保留未知。2首声调标注诗只是讲解用例，不是独立测试集。

## 研究纪律

H-M比较组织策略，固定目标与搜索条件；H-C比较评分器，固定候选池与外部评价；H-I用同一外部评价比较二者交互。内部高分不能作为文学质量的循环论证。

机制测试配置在运行前本地冻结，不是外部预注册；测试题现已暴露。改过策略后须冻结新的独立确认集。现有bootstrap不提供对所有诗歌或所有任务分布的普遍结论。

第三方快照和现代标注保留各自来源与许可。项目代码及新增研究文档为本任务创建，未替用户决定最终开源许可证。本项目不声明动态规划、卷积或矩阵存储为新发明，也不把模型等同于古人脑内机制。
