# 同前六句的尾联联合选择：已运行的离线实验

不由人供句。五言、七言各8首（种子0、1、2、3、42、43、44、45），全部来自[同一正式研究构件](https://github.com/overloaded-org/verseloom/actions/runs/41199468905/artifacts/13042330889)的 `mode=learned` 生成记录。本实验读取程序已生成的前六句；只为配对比较临时冻结，与允许用户指定输出句不是一回事。

从纯语料生成引擎回放同种子的候选，16/16逐一通过单句候选池指纹和原始尾联的词段身份、局部分数检查。三种策略每题共用全部合法尾联组合、同一前缀、起式、韵部和确定性并列处理，没有额外抽样或替换诗稿。此研究的第一臂是贪心对照，不是声称等于线上束搜索。第一臂先确定可合法续完的最高分第七句，再择第八句；第二臂比较第七、八句得分之和；第三臂只在第二臂末句得分中加入按前三联位置训练的关联权重。各臂共同验证句内不重字、全诗字数与词段、平仄、韵部与不重韵脚、非训练原句。它们没有验证文学质量。

| 诗体 | 诗稿 | 合法尾联组合数（各题之和） | 逐句→联合：换尾联 | 联合→联合加篇序分：换尾联 |
|---|---:|---:|---:|---:|
| 五言 | 8 | 20,584 | 3 | 1 |
| 七言 | 8 | 59,521 | 3 | 0 |

计数不是80,105首独立新诗，也不表示换掉的尾联更好。16组均保存，相同结果不计为胜。两项比较中输出不同的7组制作成匿名评阅卡，没有独立评分；A/B顺序由独立固定种子决定。不能把差异卡的胜率外推到所有生成。

## 复现输入与命令

先运行 `scripts/probe_closing_order.py`（见 [八句与收尾报告](../../docs/SP03_EIGHT_LINES_AND_CLOSURE.md)）。下列命令从仓库根目录运行，使用项目自身的可信模型与JS文件（不能对不可信文件使用读取私有候选的工具）。`output/continuation-{five,seven}/outputs.jsonl` 各取 mode=learned，按原顺序分别存为 `five-gen.json`、`seven-gen.json` 两个JSON数组；不筛选诗稿质量。

```sh
git show 685e34a19998e0dfaf2ff9ef4fd9b3ea2ad8e023:web/induced-engine.js > pool-engine.js
node research/sp03-endings/extract_end_pools.cjs pool-engine.js web/induced-data.js five-gen.json,seven-gen.json ending-bank.json
PYTHONPATH=scripts OPENBLAS_NUM_THREADS=1 python research/sp03-endings/pair_ending_replay.py --model web/induced-data.js --ranker output/continuation-v1/ranker.json --residual output/continuation-v1/closing-order-residual/report.json --bank ending-bank.json --refs five-gen.json seven-gen.json --out output/ending-pairs
```

检测依据：历史引擎的候选构造、正式研究构件中导出的目标指纹及原尾联路径、`c20372fd…`模型数据、`9b7ee9f2…`权重与本轮新增闭合实验权重。本地已运行，未替换默认引擎。先独立盲评句法与是否承接前六句，再决定上线与否。
