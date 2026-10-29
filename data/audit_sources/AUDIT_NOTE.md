# 审计口径

本目录中的第三方原始快照保持不变，来源与 SHA 见 manifest.json。旧版派生草稿 original_audit_results.json 把同一字在同一部内的重复列出也计入多归属统计；该文件不再作为有效审计结果。

规范的复算脚本是 ../../scripts/audit_sources.py，有效输出是 ../../results/source_audit.json。按不同 (声部, 韵部) 去重后，旧 105 部表有 1,823 个多归属字。原始字表不因此被修订。

v0.2 增加另一独立原字形来源，见 ../reference_v02/；两张表的字数和归属数来自不同文本、不同编码口径，不能把差值当作准确率改善。
