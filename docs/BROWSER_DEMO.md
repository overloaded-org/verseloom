# 律织浏览器示例

## 使用

站点入口为 `web/index.html`；从仓库首次使用前需执行下文的数据导出命令，部署产物和单文件版已包含数据。界面初始显示“首轮存档／示例·尚未运行”；点击“开始生成”后才执行算法并展示本机结果。可选择 12 个可行平声韵部、四种基本五律模板与搜索宽度 16、32、64。“换个韵部，再写一首”按可行韵部顺序切换，重新进行搜索；相同设置仍是确定性的。

生成结果支持平仄显示、逐句 2+3 片段和训练出现次数、八项机械约束检查、复制及 JSON 下载。统计搜索分不是文学质量分数。运行中可停止，失败或取消保留上一份诗稿。

网页没有外部字体、第三方脚本或模型 API 依赖。HTTP(S) 模式优先使用 Web Worker；Worker 不可用、文件协议或单文件模式使用同一引擎在主线程逐行异步让出事件循环。无需传输用户输入或设置账户密钥。

## 计算范围与复现边界

浏览器实时计算的是候选约束、训练字符共现项、重字惩罚以及有限 beam 搜索。词块筛选和候选语言分数已经在原 v0.6 试验中计算并冻结；网页不重新训练神经模型，也不重算 322,600 个原始合法候选句。

`scripts/build_web_data.py` 从冻结候选池和原训练数据导出 `web/data.js`：

- 2,392 个候选句及既有语言统计分数；
- 2,203 首五言训练诗的字符共现值；
- 484 个候选字符的对称 Float64 共现表；
- 四模板、韵部覆盖与来源计数；
- 输入文件 SHA-256、首轮基准和许可元数据。

读取语料沿用 `load_five_training`：开发集与测试集行在 JSON 解码前跳过。默认参数的八句、四模板最优路径及逐行搜索计数均与冻结 `search.json` 比较，分数比较容差为 `1e-12`。浏览器按码点实现 Python 字符串词序，并使用补偿浮点求和对应 Python 3.12 的累加行为。

约束核验依赖参考候选读音，不等同于语境读音认证；中间联对应位置异字不等同于对仗认证。“非训练完整原句”也不代表全古典诗库原创性。搜索仍为有限候选池上的启发式 beam，没有全局最优或文学质量保证。

## 开发与检查

使用 Python 3.12、项目固定的 NumPy 依赖和 Node 24：

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_web_data.py
python3 scripts/build_web_data.py --check
node --test --test-isolation=none --test-reporter=spec tests_web/engine.test.cjs
python3 scripts/build_web_demo.py --output _site
```

`web/data.js` 是被 Git 忽略的构建产物；CI 每次从冻结记录导出，避免提交重复大文件。`--check` 再次重建派生数据并比较字节，不修改文件。浏览器测试独立读取原始韵表和训练行，覆盖默认复现、四模板、全部 12 个可行韵部、强制模板、参数变化、非法输入、取消、Worker 协议与源 SHA。

打包器只复制 `web/`，另外生成 `standalone.html` 和 `build-manifest.json`。单文件版内联算法、数据、样式、应用逻辑、图标和来源许可，可直接打开使用。它不依赖旁边的其他文件。打包器拒绝覆盖包含未知文件的非空输出目录。

## GitHub Pages

工作流 `.github/workflows/pages.yml` 在相关文件提交、PR 与手动触发时检查数据、测试引擎并打包 `_site/`。主分支且仓库已启用 Pages 时才部署；PR 不部署。Pages 未启用时，构建与测试依然完成，工作流摘要说明设置缺口；不上传站点构件，`deploy` 作业跳过。只有主分支发布且 Pages 已启用时才上传构件，保留一天。构件上传失败会保留失败状态，不会继续部署。

首次启用：

1. 访问仓库 `Settings → Pages`。
2. 在 `Build and deployment → Source` 选择 `GitHub Actions`。
3. 打开 `Actions → Browser demo and GitHub Pages → Run workflow`。

不需要另建 GitHub 建议的模板。最终站点 URL 以 `deploy-pages` 成功返回的 `page_url` 为准。工作流不会更改仓库可见性、创建个人令牌或自动升级付费计划；私有组织仓库的 Pages 支持取决于组织计划与策略。

官方说明：[配置发布源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)、[自定义 Pages 工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 首次远端执行记录

2026-10-30 的首次网页工作流中，数据导出、9 项引擎测试及网页打包均通过，但 GitHub 返回 `Artifact storage quota has been hit`，站点构件未上传成功。原始失败记录保留在 [首次运行](https://github.com/overloaded-org/verseloom/actions/runs/40986351565)。后续调整为 Pages 未启用时不做无效上传；这不代表存储额度问题已解决。实际通过 Actions 发布前，仍需具备可用构件存储额度。单文件网页的本地运行不依赖该额度。
