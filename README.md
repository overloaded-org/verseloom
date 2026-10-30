# 律织 · VerseLoom

**让程序组织诗句，而不是替程序写诗。**

用声律矩阵约束位置，用带出处的词组表达意义，让上下句在句法与音韵关系中共同成篇。支持五言、七言；不预写首句，不指定中间句，不用存档诗稿填补失败。

## 直接体验

### [打开「词组共构 v2」在线生成台 →](https://overloaded-org.github.io/verseloom/composition.html)

[单文件离线版](https://overloaded-org.github.io/verseloom/composition-offline.html) · [原候选句生成台](https://overloaded-org.github.io/verseloom/) · [早期词组实验台](https://overloaded-org.github.io/verseloom/quality.html) · [发布与线上验证](https://github.com/overloaded-org/verseloom/actions/workflows/pages.yml)

打开页面，选择 **五言／七言**，点击 **开始生成**。默认的“春／薄暮”是场景条件，不是固定诗句；无需填写提示词、上句或下句，也不需要 API 密钥。

每次生成默认自动换种子。关闭该选项可以手动输入种子；**按原参数复现**会使用当次的全部参数和历史记录。点选一句，可展开词组出处、句法角色、反切与释义参照；也可复制诗文或下载完整证据 JSON。

所有生成计算在浏览器内执行。自动模式记录的近期诗稿仅在当前页面会话中使用，不上传。相同数据、引擎版本、种子、参数和历史可以复现；不同种子不保证永不重复。

## 三个入口，各自做什么

| 入口 | 生成单位与用途 |
| --- | --- |
| **词组共构 v2 · `composition.html`** | 现场组合一字、二字词项；中间两联联合构造，接入义项、角色和历史读音证据。本次上线的主体验入口。 |
| 自由组合基线 · `index.html` | 在冻结的五言／七言候选句池上搜索；保留早期结果、随机种子和对照功能。 |
| 早期词组实验台 · `quality.html` | 仓库已存在的另一套词级实验实现，保留供比较；不是 `composition.html` 的别名。 |

“主体验入口”不表示已经证明文学效果优于其他模式。旧生成器、冻结数据和实验记录均保留，没有被新版覆盖。

## 词组如何成篇

```text
固定来源的训练诗与历史字书
             ↓
有训练出处的词项 + 拟用义项 + 读音证据
             ↓
季节／时段筛选 + 句法角色与动词搭配条件
             ↓
五言／七言声律矩阵 + 中间两联联合构造
             ↓
多分支搜索 → 完整八句选择 → 原样输出与证据记录
```

**声音与意义使用同一条证据。** 先依据拟用义项和字书记录确定词项是否准入，再接受格律检查；不为满足某个平仄位置临时切换读法。释义的“上同／下同”可以追溯，但不会把参照条目的读音移到原字上。

**对仗参与构造，而不只是事后打分。** 中间两联同时匹配词组跨度、词类、句法角色和声明的语义类别，并限制重复词组、重复事件及韵脚重复。具体字词仍由程序选择。

**无整句输出库。** v2 数据包含词项、语言统计、模板和读音依据，不包含可供直接抽取的完整输出句。训练原句以哈希用于排除，不作为生成候选。采用的是有限词汇、有限句法框架，不是无限制的自然语言写作。

实现入口：[词项与读音编译](scripts/build_composition_v2.py) · [浏览器生成器](web/composition-engine.js) · [独立约束核验](scripts/audit_composition_v2.py) · [完整说明](docs/COMPOSITION_V2.md)。

## 当前能力与边界

| 已实现的工程能力 | 不能由此推断的结论 |
| --- | --- |
| 八句五言／七言，基本声律及声明的窄范围变式 | 覆盖全部传统变格、拗救或诗体情况 |
| 中间联的词组、角色和语义槽位对应 | 专家已认定工对、无合掌、用典准确 |
| 有出处的历史读音、拟用义项与韵部匹配记录 | 所有生成语境读音均已得到独立审定 |
| 季节、时段和搭配条件，搜索与回放 | 文学质量、篇章意蕴或全局最优已获保证 |
| 排除来源训练集的完整原句 | 全古典诗库原创性认证 |

本版词项设计使用旧 `train` 与新增 `train_candidate` 的 **13,810 首形状候选文本**，不是已认证的严格律诗金标。当前设计的 564 个词项中，408 个通过声明的来源／读音准入检查，156 个排除；运行时还会按场景筛选。计数及排除理由由构建报告重新计算，不作为质量评分。

词义、语义类别与语法策略是可审查的助手设计，并非独立人工金标。未找到满足全部约束的诗稿时，界面明确报告失败，不自动放宽规则或返回旧诗。七言的部分季节／时段组合仍可能无解。

## 本地运行

运行网页不需要 Python 或 Node。以下命令用于从源码重新构建整个站点，使用 **Python 3.12**；测试及发布使用 **Node.js 24**。首次构建需要下载固定版本的公开语料与《广韵》，不执行第三方下载脚本。

```bash
git clone https://github.com/overloaded-org/verseloom.git
cd verseloom
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 保留的基线生成器
python scripts/build_web_data.py
python scripts/build_web_seven.py

# 一次性建立有版本与散列的来源缓存；目录必须尚不存在
python scripts/bootstrap_quality_data.py --download \
  --cache output/phrase-input/cache --output output/phrase-input/run

# 保留仓库已有词级实验台
python scripts/build_quality_v2.py \
  --bootstrap output/phrase-input/run --output output/quality-v2
python scripts/build_quality_offline.py

# 编译本次词组共构 v2
python scripts/build_composition_v2.py --bootstrap output/phrase-input
python scripts/build_composition_v2.py --bootstrap output/phrase-input --check
mkdir -p web/licenses
cp output/phrase-input/cache/licenses/tshet-uinh-CC0.txt web/licenses/
python scripts/build_composition_demo.py --output web/composition-offline.html

# 打包并加入首页入口、发布身份及文件清单
python scripts/build_web_demo.py --output _site
python scripts/build_composition_site.py --site _site
python -m http.server 8000 --directory _site
```

访问 `http://localhost:8000/composition.html`。离线使用可直接打开 `_site/composition-offline.html`。它内联程序、词项数据和来源许可，不依赖 CDN 或模型接口。

构建器会拒绝覆盖已有的原始缓存、实验输出和离线 HTML。重建时使用新的相应输出路径，或在确认内容不再需要后清理自己的构建产物；不要改写冻结实验目录。`web/*-data.js` 等派生大文件由构建流程生成，不提交到版本库。

## 测试与复现

在完成上面的数据构建后：

```bash
# 历史 Python 回归与新增证据／站点检查
PYTHONPATH=src python -m unittest discover -s tests -v
python -m unittest discover -s tests_composition -v

# 固定五言／七言配置，各 16 个种子；写入新的结果目录
node scripts/evaluate_composition_v2.cjs output/phrase-results 16
VERSELOOM_COMPOSITION_REPORT=output/phrase-results/report.json \
  node --test --test-isolation=none --test-reporter=spec \
  tests_composition/engine.test.cjs tests_composition/ui.test.cjs

# 用另一实现核验原样输出、词项身份、平仄和对齐记录
python scripts/audit_composition_v2.py \
  --input output/phrase-results/outputs.jsonl \
  --output output/phrase-results/independent_audit.json
```

这些测试检查实现与声明规则，不给文学质量打合格证。界面事件合约测试也不等于 Safari／Chrome／手机的完整渲染兼容测试。原样样本、参数、失败项和负结果应同时保留；不以人工改字替换程序输出。

## Pages 发布

[`Browser demo and GitHub Pages`](.github/workflows/pages.yml) 在相关 `main` 提交后构建三个入口，执行生成与证据检查，发布站点，再下载正式网址的资源进行同构建 SHA-256 校验及五言／七言执行、复现检查。`release.json` 记录实际部署提交；绿色构建不替代部署和 `verify-live` 成功。

[查看工作流](https://github.com/overloaded-org/verseloom/actions/workflows/pages.yml) · [线上版本身份](https://overloaded-org.github.io/verseloom/release.json)

Fork 部署时，在仓库 Settings → Pages 中选择 GitHub Actions；源码中的项目外链需要相应调整。流程不修改仓库可见性、计费配置或其他项目构件。

## 研究记录

本项目从“声律矩阵＋词卷积”验证开始。早期卷积／矩阵匹配模型在相应复原实验中没有超过指定统计基线，**当前网页整诗生成不是这些神经模型的成果**。它使用词项、规则和统计约束搜索，不调用大语言模型 API。

历史实现与负结果见 [docs/](docs/)、[首轮程序原始输出](results/free_generation_v06_run001/result.json)、[七言支持说明](docs/SEVEN_SUPPORT.md)、[质量研究 v1](docs/QUALITY_REVIEW_V1.md)、[数据扩充结果](docs/DATA_BOOTSTRAP_V1_RESULT.md)。本次改写前的 README 原文保存在 [README_BEFORE_PAGES_V2.md](docs/README_BEFORE_PAGES_V2.md)。Python 包 `versematrix-lab`、模块 `wordconv_lab` 继续沿用历史命名，以保持实验身份。

## 来源与许可

原诗数字语料来自 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，固定提交 `b8594f81a89752241442f2ce267d6f66f96704ee`，保留 MIT 许可与来源信息。《广韵》数字数据来自 [nk2028/tshet-uinh-data](https://github.com/nk2028/tshet-uinh-data)，固定提交 `21585e22c8a730ca2fd175112f4d18e16d5ce578`，保留 CC0-1.0 声明。平水韵录文与派生数据沿用项目 [CC BY-SA 4.0 来源通知](data/reference_v02/LICENSE_NOTICE.md)。

第三方材料分别保留其权利声明，不替整个仓库改授许可。本项目原始代码目前未新增统一开源许可证。详见 [来源登记](configs/data_source_registry_v1.json)、[网页 NOTICE](web/NOTICE.md) 和各数据目录的通知。
