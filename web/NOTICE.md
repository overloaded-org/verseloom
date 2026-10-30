# 律织网页示例：数据、来源与范围

网页在浏览器内执行候选约束和有限 beam 搜索。`data.js` 来自已冻结的 v0.6 首轮实验：2,392 个已评分候选句、训练诗字符共现统计、四个基本五律模板与参考韵部。网页不在访问时重新训练模型，也不重新评分全部 322,600 个原始合法候选句。

导出脚本是 `scripts/build_web_data.py`，数据中记录源文件 SHA-256。默认配置用于复现首轮结果；改变韵部、模板或搜索宽度属于网页探索，不改写原始试验记录。格律检查以未确认语境读音的参考表候选为条件，不认证对仗、语义、文学质量、全诗库原创性或全局最优。

## 唐诗数字语料

来源为 [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)，固定提交 `b8594f81a89752241442f2ce267d6f66f96704ee`。保留 Copyright (c) 2016 JackeyGao 与 [完整 MIT 许可证](licenses/poetry-MIT.txt)。网页派生统计只取既有 `train` 分割中的 2,203 首五言诗。

## 平水韵录文及派生参考数据

《平水韻》来源页署名劉淵、王文鬱；电子录文和现代编排贡献者为中文维基文库贡献者。

- 固定录文：[修订 8264804](https://zh.wikisource.org/w/index.php?title=平水韻&oldid=8264804)。
- 贡献者与修订历史：[页面历史](https://zh.wikisource.org/w/index.php?title=平水韻&action=history)。
- 保留 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 通知及 [项目原始来源说明](licenses/rhyme-NOTICE.md)。

网页数据的新增处理包括导出筛选后的候选、候选平仄与韵部、计算训练字符共现值并编码为对称浮点表；没有简繁转换或人工修订诗句。录文及其派生参考部分继续遵循上述署名与相同方式共享条件。本说明不为整个项目代码或其他独立资料新增统一许可证。

项目源码及完整实验范围见 [overloaded-org/verseloom](https://github.com/overloaded-org/verseloom)。
