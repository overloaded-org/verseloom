# 律织 · VerseLoom

**把声律表示为矩阵，让程序自行组织诗句，并保留可复核的生成过程。**

项目原名 **VerseMatrix Lab**，现以 **律织 · VerseLoom** 为项目名称。Python 包名 `versematrix-lab`、模块目录 `wordconv_lab`、`wordconv-*` 命令及历史报告文件名沿用既有命名，以保持源码、数据和冻结实验记录可复核。

当前可直接运行的最新入口是 **v0.6 自由生成试验**：不接收主题、上句、下句后缀或人工候选，从训练语料自动抽取短片段，按声律约束和固定统计分数组成八句五言。此前的矩阵求解、卷积与二字补句实验也保留在仓库中。

## 第一首程序原始输出

以下四十字直接取自 [首轮原始结果](results/free_generation_v06_run001/result.json)，没有人工改字：

> 日暮春風起，青山美少年。  
> 誰知何處去，獨有白雲天。  
> 萬里無人到，還家越水邊。  
> 江南歸路遠，草木落花前。  

程序自动选中仄起、首句不入韵的基本模板，韵脚为“年、天、邊、前”。参考表的候选平仄检查通过；“青山美少年”“草木落花前”的语义衔接仍生硬，对仗和文学质量尚未得到认证。

本次由**统计片段自动组合与约束搜索**生成。v0.5 卷积模型没有参与这首诗的生成，因此它不构成“卷积模型已学会写整诗”的证据。

## 快速开始

推荐 Python 3.12；首次运行记录使用 Python 3.12.14、NumPy 2.3.5。不需要外部模型、API 密钥或重新训练神经模型。请在仓库根目录执行：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps

# 无内容输入：程序自行生成八句。输出目录必须尚不存在。
python3 scripts/generate_free_poem_v06.py --output results/free_generation_replay
```

当前生成器是确定性算法；相同源文件、配置与数值环境下复跑应得到同样的选择。`seed: 0` 仅是试验记录标签，不表示随机采样。第一次结果保存在 `results/free_generation_v06_run001/`，程序拒绝覆盖已有输出目录。

可用以下命令核对**既有首轮结果**，不会重新生成诗：

```bash
python3 scripts/audit_free_poem_v06.py \
  --run results/free_generation_v06_run001 \
  --output results/ci_free_poem_audit.json
```

审计输出文件也只能新建；再次执行时应使用另一个输出文件名。

## 生成算法

1. 只使用原 `train` 分割中的 **2,203 首五言诗**，不从开发集或测试集取材。
2. 自动抽取各行前二字与后三字，两个片段均须出现至少两次；未知、跨平仄候选和已标记校勘问题字被排除。
3. 组合成五字句，并筛除完整训练原句、句内重字以及不符合四种基本句型的组合。韵脚必须在参考表中对应唯一平声韵部。
4. 用双向插值三元字符统计评分；程序按不同韵脚的覆盖度与训练频数自动选韵部。
5. 比较四种五律基本模板，通过宽度 32 的 beam 搜索选出全诗；评分加入训练诗字符共现，扣除重字惩罚。全诗每字最多出现两次，韵脚不重，中间两联对应位置不同字。

配置见 [free_generation_v0.6.json](configs/free_generation_v0.6.json)。本轮共评分 **322,600 个合法候选句**，保留 2,392 个候选进入有限搜索。它是有限候选池上的启发式搜索，不保证全局最优。唯一候选平仄也不等于已经确认语境读音。

“不是训练集完整原句”只是在本次五言训练集范围内的检查，不是全古典诗库原创性查重。固定跨度片段也不等于语言学上的分词。

## 运行记录与核验

| 文件 | 内容 |
|---|---|
| [freeze.json](results/free_generation_v06_run001/freeze.json) | 生成前固定的代码、配置与输入 SHA |
| [resources.json](results/free_generation_v06_run001/resources.json) | 全部可用片段、计数与来源位置 |
| [candidate_pools.json](results/free_generation_v06_run001/candidate_pools.json) | 保留候选及统计分数 |
| [rhyme_selection.json](results/free_generation_v06_run001/rhyme_selection.json) | 程序选韵依据 |
| [search.json](results/free_generation_v06_run001/search.json) | 四模板搜索计数与最终路径 |
| [result.json](results/free_generation_v06_run001/result.json) | 原始八句、逐句来源及分数分解 |
| [independent_review.json](results/free_generation_v06_engineering/independent_review.json) | 独立来源与最终路径审计 |

独立审计重建了 2,709 个片段记录和 8,035 次来源，复算全部 2,392 个保留候选分数与四模板最终路径，共 2,589 个数值项，最大差为 0.0。它没有独立重放每个被剪枝状态，也不认证文学质量。

完整单元测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

本次仓库整理后的本地验证为 **175 项全部通过**，迁移后的首诗独立审计也通过。对应记录在 [GitHub 导入验证](results/github_import_validation/)。GitHub Actions 使用同一套单元测试和只读首诗审计，不自动重新生成或改写冻结样本；远端 CI 状态以实际 Actions 运行记录为准。

## 既有二字补句功能

保留了 v0.5 主模型三个种子共同选中的第 800 步权重。默认 seed 0 不是最好种子，也不是集成。无需再训练即可运行：

```bash
python3 scripts/complete_span_v05.py \
  --upper '江雨催寒色' --suffix '過遠村' --method compare --limit 5
```

这个接口要求完整上句及下句后三字／后五字，只选择下句前两个字；它与上面的无内容输入整诗生成入口不同。统计分数与匹配模型分数都不是文学质量分数。原字形保留，不自动简繁转换。

## 历史实验与负结果

| 阶段 | 已得到的证据 | 尚未得到的结论 |
|---|---|---|
| v0.2 声律矩阵与动态规划 | 基本平仄文法、有限局部可加目标的精确求解 | 任意全局语义目标的精确求解或文学质量保证 |
| v0.3 卷积复原 | 六次正式训练、原文恢复及求解核验 | CNN 未通过晋级标准 |
| v0.4 扩展候选与关联检索 | 新独立确认及完整统计记录 | 未通过晋级标准；扩词表不等于复原质量提升 |
| v0.5 上下文矩阵匹配 | 12 次训练、14,400 次参数更新；开发诗族等权 R@16 主模型 1.058%，统计基线 4.425% | 三个开发资源门均未通过；没有运行新的独立确认 |
| v0.6 无填句整诗试验 | 自动选择全部 40 字，固定规则产生首轮完整结果 | 对仗、语义连贯性、文学质量与普遍生成效果待验证 |

历史协议、报告和原 README 均保留在 [docs](docs/)，其中 [README_v0.5.md](docs/README_v0.5.md) 是此前完整实验包的说明。基础 Python 包版本暂保留 `0.5.0`，自由生成模块单独以试验版本 `v0.6` 标记，避免改写旧实验源码身份。

## 仓库收录范围

主仓库收录全部源码、脚本、测试、配置、合成 fixture、语料与韵表数据、历史 JSON 报告／冻结／审计、首首完整生成轨迹，以及三个已选主模型权重。原始复制文件保持字节不变。

体积较大的历史逐题预测 JSONL、未选权重及工程回读副本没有放入主仓库。共 131 个省略历史文件的路径、字节数和 SHA，以及 353 个原始复制文件的 SHA，见 [GITHUB_IMPORT_SCOPE.json](docs/GITHUB_IMPORT_SCOPE.json)。旧 manifest 是历史完整包的清单，不是本仓库全部文件均存在的声明；需要重算历史完整逐题报告时，应使用原完整实验包或按原协议输出到新的结果目录。最新整诗生成、首诗审计、二字补句 CLI 和单元测试均具备本地依赖。

## 来源与许可

唐诗数字语料来自 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，固定提交 `b8594f81a89752241442f2ce267d6f66f96704ee`；完整原始来源、作者信息、SHA 与许可通知随 [data/corpus_v03](data/corpus_v03/) 和 [data/confirmation_v04](data/confirmation_v04/) 保存。源诗仅做八行×五／七字的形状筛选，并非已认证严格律诗金标。

历史韵表来自中文维基文库固定修订，字形和已知校勘问题均保留。录文、派生参考数据的署名与 CC BY-SA 4.0 通知见 [data/reference_v02/LICENSE_NOTICE.md](data/reference_v02/LICENSE_NOTICE.md)。其他第三方源码或资料沿用各自随附许可。本次整理没有添加覆盖整个项目原始代码的开源许可证，也没有把第三方数据静默改授其他许可。
