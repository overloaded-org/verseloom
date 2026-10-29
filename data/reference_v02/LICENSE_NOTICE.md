# 来源与许可说明

## 原作与录文应区分

本目录保存《平水韻》与《重修廣韻（四庫全書本）》的维基文库录文快照，
不是本项目新创作的历史字书。《平水韻》固定修订在源码中标注 `{{PD-old}}`。
原作的公有领域地位不等于可忽略现代录文、编排或校订的署名要求。

维基文库 `siteinfo/rightsinfo` 的实际响应为 **Creative Commons Attribution-Share Alike 4.0**，许可链接：
<https://creativecommons.org/licenses/by-sa/4.0/deed.zh>。原响应保存在 `wikisource_rightsinfo.json`。

本目录对录文及其派生参考数据保留维基文库作者/贡献者署名、固定版本链接与
CC BY-SA 4.0 通知；如现代贡献适用版权，复用或修改这些内容时应遵循该许可。
本通知不将整个项目的代码或其他独立文件重新授权为此许可。

## 署名与固定版本

- 《平水韻》：来源页署名劉淵、王文鬱；电子录文及现代编排贡献者为中文维基文库贡献者。
  固定版本：<https://zh.wikisource.org/w/index.php?title=平水韻&oldid=8264804>。
  修订历史：<https://zh.wikisource.org/w/index.php?title=平水韻&action=history>。
- 《重修廣韻（四庫全書本）/卷1》：陈彭年等撰，中文维基文库贡献者录入。
  固定版本：<https://zh.wikisource.org/w/index.php?oldid=770095>。
- 《重修廣韻（四庫全書本）/卷2》：陈彭年等撰，中文维基文库贡献者录入。
  固定版本：<https://zh.wikisource.org/w/index.php?oldid=770099>。

## 本项目作出的处理

从冻结 API JSON 抽取原始 wikitext，解析 106 部主表及独立补充行；
候选索引按“原字形—韵部”去除同部重复，保留跨部归属；加入来源、审计、
问题标记与构建信息。未作简繁转换、Unicode 规范化、字形合并或正文勘改。
具体字节校验值见 `source_manifest.json`，所有派生处理可离线重建。
