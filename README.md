# 律织 · VerseLoom

**让算法组织诗歌，不替算法预填词句和故事。**

把声律表示为约束矩阵，从训练语料学习词段、接续和跨句关系，在浏览器里搜索五言、七言八句。每次保留原样输出、参数和来源，不以人工改写或存档诗稿掩盖失败。

## 在线体验

### [打开「语料自组织」生成台 →](https://overloaded-org.github.io/verseloom/induced.html)

选择五言或七言，点击**开始生成**。没有主题、季节、人物和故事路线输入，也不读取人工内容词表或句法框架。初始诗稿为空；计算在浏览器 Worker 内运行，不需要模型 API 或密钥。

“种子与研究设置”中可关闭自动换种子，或切换句内统计／跨句关联作比较。自动模式可降低近期词段重复；手动种子不暗带历史。**原参数复现**使用原种子、参数和历史，不把复现再次记入历史。点击一句查看算法依据，导出 JSON 保存完整结果。五言、七言的近期记录分开，关闭页面后不保留，也不上传。

首次加载约17.3 MiB统计资产；这不是云端模型。宽度和设备不同，计算耗时会变化，可随时停止。未找到完整结果会明确报告，不放宽规则或填入备用诗稿。

## 各入口的区别

| 页面 | 内容由什么决定 |
|---|---|
| **[语料自组织](https://overloaded-org.github.io/verseloom/induced.html)** | 从语料自动发现词段，学习接续、句位和跨句关联。当前“算法主导、没有人工内容预填”的研究主线。 |
| [词组共构 v2](https://overloaded-org.github.io/verseloom/composition.html) | 有来源的词项与人工编写的义项／角色约束。保留作对照，不与自组织模式混称。 |
| [原候选句基线](https://overloaded-org.github.io/verseloom/) | 在冻结的五言／七言候选句池中组合搜索。不是开放词组空间。 |
| [早期词组实验台](https://overloaded-org.github.io/verseloom/quality.html) | 另一套有类型约束的词级实验，保留历史比较用途。 |

所有入口都不预先指定某个首句、中间句或尾句。不同种子不保证永不重复。新主线的对仗是学习得到的软评分，不具有旧手工角色框架的硬对齐保证。

## 算法主线

```text
仅训练诗族 + 冻结音韵参考
             ↓
自动发现一至三字词段、概率分段
             ↓
学习词段接续、句位、句尾及跨句关联
             ↓
声律矩阵中的多分支组合搜索
             ↓
比较完整八句 → 原样诗稿 + 参数 + 来源与分数
```

自组织构建器不读取人工内容词表、语义分类、主谓宾论元表或“家国→离愁”路线。训练素材为13,810首经过形状筛选的文本，不是严格律诗金标；目前准入10,000个统计词段。具体来源与局限见 [CORPUS_INDUCTION_V1.md](docs/CORPUS_INDUCTION_V1.md)。

**没有人工内容预填，不等于没有工程假设。** 语料范围、词段长度、模型、评分权重、搜索预算和声律规则仍由工程设定。共现不是因果；词段不等于已经认证的古汉语词；目前没有可靠的人物共指或事件叙事模型。

历史读音保留候选集合，不为了合律临时选读音。平仄一致不等于语境读音已经消歧。基本声律检查也不能认证严格工对、典故、诗意或全网原创性。

## 手机与重复偏差修复

四个在线入口及已有离线页面使用响应式布局。修复长证据ID撑宽、七言窄屏显示、导航和操作区换行；窄屏输入16px，主要触控控件至少44px。自组织设置折叠，成功后定位诗稿。

旧首页的auto择韵原先总取首个可行韵部，放大了少数高分片段。基线v4改为在探索模式下按种子选择可行韵部；显式韵部和强度0的历史优化不变。完整近期诗稿的二／三字片段作通用软惩罚，**没有针对“美少年”“虎眠”的禁词表**。自组织v1.1同样使用近期偏好，候选词表和单句候选构造不随历史改变。

固定64个五言开发种子的诊断中，可见韵部从1增至12；含“美少年”由30首变为0，含“虎眠”由16首变为0。这不是永不出现保证，其他高频片段仍存在。另一次同预算近期偏好对照中，自组织七言没有观察到重合率改善；负结果保留。

本地Chromium中，四入口×两诗体×五档宽度（320、375、390、430、768px）共40组布局检查通过，并实际生成、展开证据与复现。它是内存加载打包程序的测试，**不是iOS Safari真机或正式URL点击认证**。方法、完整数字及复现步骤见 [MOBILE_DIVERSITY_V1.md](docs/MOBILE_DIVERSITY_V1.md)。

## 从源码构建

运行线上网页无需安装。重新构建使用Python3.12、NumPy版本见requirements.txt；测试及Pages使用Node.js24。首次构建下载固定版本公开数据，不执行第三方下载脚本。以下输出目录应尚不存在。

```sh
git clone https://github.com/overloaded-org/verseloom.git
cd verseloom
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/build_web_data.py
python scripts/build_web_seven.py
python scripts/bootstrap_quality_data.py --download \
  --cache output/phrase-input/cache --output output/phrase-input/run

python scripts/build_quality_v2.py \
  --bootstrap output/phrase-input/run --output output/quality-v2
python scripts/build_quality_offline.py
python scripts/build_composition_v2.py --bootstrap output/phrase-input
mkdir -p web/licenses
cp output/phrase-input/cache/licenses/tshet-uinh-CC0.txt web/licenses/
python scripts/build_composition_demo.py --output web/composition-offline.html

python scripts/build_induced_model.py --bootstrap output/phrase-input
python scripts/build_induced_model.py --bootstrap output/phrase-input --check
python scripts/build_web_demo.py --output _site
python scripts/build_composition_site.py --site _site
python -m http.server 8000 --directory _site
```

访问 `http://localhost:8000/induced.html`。自组织入口需要支持Worker的浏览器与HTTP服务；不要直接以file协议打开。其他模式保留各自单文件离线入口。不要覆盖冻结实验结果；构建缓存和派生大文件不提交进Git。

## 测试与发布

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m unittest discover -s tests_composition -v
python -m unittest discover -s tests_induction -v
node --test --test-isolation=none tests_web/engine.test.cjs
node --test tests_induction/engine.test.cjs
node scripts/probe_repetition.cjs web output/repetition.json 64
node scripts/evaluate_session_diversity.cjs output/session 8
```

移动渲染脚本为 [tests_mobile/check_mobile.py](tests_mobile/check_mobile.py)，需要Playwright1.57.0与Chromium。可在完成站点打包后运行：

```sh
python -m pip install playwright==1.57.0
python -m playwright install chromium
python tests_mobile/check_mobile.py --site _site --output output/mobile
```

[Pages工作流](https://github.com/overloaded-org/verseloom/actions/workflows/pages.yml)重新构建、测试、部署，再下载正式站点资产核对本次构建SHA并执行各生成器。线上资源检查不是手机浏览器测试，CI通过也不是文学质量通过。新增回归检查覆盖种子复现、近期记录、候选池不变、无固定内容和离线移动样式。

## 研究记录与边界

[声律与动态规划](docs/ALGORITHM_v0.2.md) · [七言支持](docs/SEVEN_SUPPORT.md) · [数据补充](docs/DATA_BOOTSTRAP_V1_RESULT.md) · [语料自组织](docs/CORPUS_INDUCTION_V1.md) · [移动与重复诊断](docs/MOBILE_DIVERSITY_V1.md)

早期卷积与匹配实验的负结果、原诗和冻结日志保留；当前自组织生成不是此前卷积模型的成果。独立文学评分、专家工对与完整语境读音认证仍未完成。400组开发标注任务和机器假设不能冒充已批准金标。

## 来源与许可

原诗数字资料来自固定版本的chinese-poetry（MIT）；《广韵》数字资料来自固定版本的nk2028/tshet-uinh-data（CC0-1.0）；平水韵录文与派生数据保留CC BY-SA4.0署名及相同方式共享要求。许可与版本记录见 [web/NOTICE.md](web/NOTICE.md)、[data/reference_v02/LICENSE_NOTICE.md](data/reference_v02/LICENSE_NOTICE.md)和数据来源登记。第三方资料各自保留许可，不自动替整个仓库改授许可证。

原项目名VerseMatrix Lab、Python包名和历史文件名继续保留，避免破坏实验可复现性。历史收录和省略文件范围见 [GITHUB_IMPORT_SCOPE.json](docs/GITHUB_IMPORT_SCOPE.json)。
