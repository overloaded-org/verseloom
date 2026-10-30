# 质量数据启动 v1

本轮落实原诗扩容、音韵来源接入和开发标注种子；不是训练上线，也不是完成200首人工金标。所有输出句位仍由现有程序自由选择。本提交不修改 `web/`、原生成器、冻结数据或既有实验分割。

## 可重复的运行

```sh
python -m unittest discover -s tests_data -v
python scripts/bootstrap_quality_data.py --download \
  --cache output/data_bootstrap/cache --output output/data_bootstrap/run
# 不再联网；输出目录必须是新目录
python scripts/bootstrap_quality_data.py \
  --cache output/data_bootstrap/cache --output output/data_bootstrap_replay
cmp output/data_bootstrap/run/manifest.json output/data_bootstrap_replay/manifest.json
```

Actions 工作流 `Quality data bootstrap` 执行相同检查，保存 `quality-data-bootstrap` 构件。所有外部来源固定提交，逐文件验证 Git blob SHA、SHA-256和字节数。仅下载数据，不执行第三方脚本。原文不做简繁转换、补字或改写。

## 原诗筛选与泄漏隔离

读取同一固定 chinese-poetry 提交中 offset>=20000 的剩余唐诗分片。只把八句、每句五/七字且正文为汉字的文本列为**形状候选**，不自动认定律诗或优质诗。

旧 `corpus_v03/poems.jsonl` 的全部 train/dev/test，以及 `confirmation_v04/new_unique_poems.jsonl` 的全部文本（包括未选入确认集的材料），只用于排除关系。新旧文本联合建立诗族，关系包含：完整句相同、NFKC匹配键的句内二元字串集合Jaccard>=0.8、来源ID相同、作者与题名相同。关系闭包接触旧材料的新增诗全部隔离；同文不同题署的来源记录不得丢弃。NFKC仅用于匹配，输出保留原字形。词表/评分器不读取保留分割。

合资格新诗族按固定SHA规则80/10/10划为 train_candidate/dev_reserve/test_reserve。比例是哈希桶规则而非保证精确数量。候选和保留集仍需版本、诗体与标签复核；不把这批数据宣称为完成的独立盲测。古籍异体、繁简和未记录的版本关系仍可能漏检，同作者同题可能保守地过度隔离。

## 200首与400组任务

只从新 train_candidate 中选择五言/七言各100首，一诗族最多一首，优先分散作者，每种诗体每位作者最多5首。每首两组中间联，共400组任务。每条带八句上下文、原诗ID、分片/记录序号、字位、来源版本以及待填写的词边界、义项、句法角色、对应关系和语境读音字段。

**任务生成数不等于标注完成数。** 初始标注为 pending，标签为空，严格对仗为unknown，is_gold=false。后续助手草案单独保存，不能冒充独立人工金标；没有身份与出处不能升格为reviewed。原诗进入种子集不等于今后生成必须输出它的任何一句。

## 历史音韵证据

从 nk2028/tshet-uinh-data 固定提交接入《广韵》CSV：保留字头、校勘说明、反切、音韵地位、释义、释义参照及记录位置。带编辑符号或空字头保留为原记录，不猜测其字形。相同字所有候选并列；不根据模板挑选读音，不把《广韵》韵目当作平水韵，也不以唯一条目视为语境已确认。任务中的平水韵候选沿用冻结参考表，仅并列展示，映射和语境选择仍为空。

## 来源状态与许可

- 原诗数字语料：`chinese-poetry/chinese-poetry`，提交 b8594f81a89752241442f2ce267d6f66f96704ee，MIT，保留版权与许可全文。
- 《广韵》数字数据：`nk2028/tshet-uinh-data`，提交21585e22c8a730ca2fd175112f4d18e16d5ce578，CC0-1.0，保留原表与许可全文。
- 任务中平水韵派生部分沿用项目 CC BY-SA 4.0 署名/相同方式共享要求；产物附 `PINGSHUI_NOTICE.md`。不同来源的权利声明分别保留，不替整个仓库改授许可。
- 2022词边界论文的32,399首及2024对仗论文的6,125组：本轮未确立数据文件与数据许可，不导入。后者来自六朝五言联句。作者后续 parallelism-benchmark 的银标流水线不冒称同一人工金标。

来源登记见 `configs/data_source_registry_v1.json`。检查范围：论文、ACL页、作者公开项目/发表页及相关仓库README；没有声称网上不存在其他发布。

## 本轮验收

17项合成边界测试：包括相似度连接与穷举对照、传递泄漏、异题署同文、分割与顺序不变性、种子诗体/作者/诗族限制、字书候选与校勘保留、禁止假金标、读音不自动消歧、禁止覆盖和来源URL限制。CI另做完整数据离线重建并比较全部产物清单。实际规模和结果见运行产物 `run/report.json`，不得将预期计数写成实测。
