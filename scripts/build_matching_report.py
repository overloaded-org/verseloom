"""Render the completed v0.5 matching experiment and completion-tool report.

Reads actual saved results. It never generates placeholder empirical values.
Report dependencies are optional and separate from the NumPy experiment.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from build_report import Report, TEAL, AMBER, MUTED
from reportlab.lib import colors


class MatchingReport(Report):
    def page_start(self, number, title, subtitle=""):
        super().page_start(number, title, subtitle)
        self.c.saveState()
        self.c.setFillColor(colors.white)
        self.c.rect(self.left, 17, self.content, 15, fill=1, stroke=0)
        self.c.setFillColor(MUTED)
        self.c.setFont("CJK", 8)
        self.c.drawString(self.left, 24, "原文恢复开发验证  |  尚无文学质量结论")
        self.c.drawRightString(self.width - self.right, 24, f"{self.page:02d}")
        self.c.restoreState()


LABELS = {"full_w3": "双方上下文 / 宽度3", "full_w1": "双方上下文 / 宽度1",
          "upper_w3": "只看上句 / 宽度3", "suffix_w3": "只看后缀 / 宽度3"}


def pct(value):
    return f"{100 * value:.3f}%"


def pp(value):
    return f"{100 * value:+.3f}"


def metric(record, name="recall_at_16"):
    return record["metrics"][name]["family_macro"]


def build(root: Path, output: Path, fonts: Path, *, draft=False):
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))
    report = read("results/matching_v05/report.json")
    config = read("configs/matching_v0.5.json")
    selection = read("results/matching_v05/selection_frozen.json")
    tests = read("results/test_receipt.json")
    review_path = root / "results/matching_v05/independent_review.json"
    review = read("results/matching_v05/independent_review.json") if review_path.exists() else None
    if review is None:
        if not draft:
            raise ValueError("final report requires the completed independent review")
        review = {"status": "独立结果复核进行中；本文件为排版工作稿"}
    if not draft and not str(review["status"]).startswith("PASS"):
        raise ValueError("independent review has not passed")
    examples = read("results/completion_examples_v05.json")
    preflight = read("results/matching_v05_preflight.json")
    if report["status"] != "completed_dev_only" or tests["status"] != "passed":
        raise ValueError("completed results and passing acceptance checks required")
    gate, arms = report["resource_gate"], report["arms"]
    ordered_arms = [(key, arms[key]) for key in ("full_w3", "full_w1", "upper_w3", "suffix_w3")]
    primary = arms["full_w3"]
    base = report["baselines"]["strong_statistical"]
    result = primary["seed_average"]
    mismatch = report["diagnostics"]["mismatched_context"]["seed_average"]
    zero = report["diagnostics"]["zero_context"]["seed_average"]
    coverage = report["coverage"]["splits"]
    accounting = report["training_accounting"]
    output.parent.mkdir(parents=True, exist_ok=True)
    r = MatchingReport(output, fonts, version="v0.5")

    r.page_start("01 / 交付", "上下文匹配已经可以运行", "固定词表、固定16候选；本轮结果属于开发实验" + ("；排版工作稿" if draft else ""))
    headline = "通过开发资源门，具备新确认的候选资格" if gate["passed"] else "未通过开发资源门，保留实验工具与完整结果"
    r.box(headline, f"主模型诗族等权R@16为{pct(metric(result))}，固定统计基线为{pct(metric(base))}，差{pp(gate['primary_mean_gain_over_strong_baseline'])}个百分点。这里是选过检查点的开发结果，尚未进行新的独立确认。", TEAL if gate["passed"] else AMBER)
    r.table(["本轮可用产物", "具体内容"], [
        ["可训练匹配器", "上下文和二字片段分别编码，完整评分矩阵为 Q E^T"],
        ["可直接补句的命令", "输入五/七言上句与下句后缀，显示最多16个候选成句"],
        ["已训练权重", "4组模型×3个种子；每次3个不可覆盖检查点"],
        ["完整实验记录", "训练安排、逐题排名、选择、信息对照及独立复核"],
    ], [132, r.content - 132])
    r.h("本轮实际计算")
    r.p(f"完成{accounting['formal_runs']}次正式训练、{accounting['optimizer_updates']:,}次参数更新、{accounting['sample_presentations']:,}次训练题呈现。开发集共{coverage['dev']['tasks']:,}题、{coverage['dev']['families']}个诗族；原文不在词表的题目仍计入分母。重复训练呈现不是新增的独立样本。")
    r.p("补句工具默认并列展示固定统计基线与预先指定的主模型seed0。报告中的三种子平均是对三次运行的指标平均，不是三模型集成检索器。", "small")
    r.h("当前判断")
    r.p("本轮验证的是从可见文字找回语料原文二字片段的能力。候选补句可以供人挑选；严格平仄、语境读音、对仗及全诗文学质量仍是另外的评价问题。")

    r.page_start("02 / 算法", "把词卷积写成可训练矩阵", "候选是原语料中的相邻二字片段，不预设已经完成语言学分词")
    r.box("S = Q E^T", "Q：上句和可见后缀经卷积、固定位置拼接、投影后得到的向量；E：候选二字按左右顺序编码得到的向量。二者平滑归一化，Q再除以固定温度0.1；候选独立输出偏置为0。")
    r.table(["层", "实现"], [
        ["输入", "上句5/7字；下句仅可见后缀3/5字。作者、题名、目标二字不入查询。"],
        ["字符表示", "共享12维训练字符向量；未知字为固定零向量，填充用独立mask。"],
        ["上下文", "两行各7个位置，共享水平tanh卷积；上句7格与后缀5格拼接。"],
        ["候选", "两个字符有序拼接，线性投影+tanh到16维；无候选ID向量。"],
        ["评分与排序", "完整词表矩阵乘法；分数降序，精确同分按候选原字形字典序。"],
    ], [91, r.content - 91])
    r.h("边界清楚的矩阵组织")
    r.p("已有声律模板和有限局部可加目标的动态规划继续保留。新的上下文匹配分数先用于完整候选排序；没有把它直接当作已经证明可按原动态规划分解的整诗目标。")
    r.p("方法借鉴：Kim（2014）的局部文本卷积；Karpukhin等（2020）的双编码器内积检索。本项目采用很小的随机初始化模型和二字恢复任务，没有复用论文中的预训练模型或效果数字。", "small")

    r.page_start("03 / 训练", "固定数据、候选与选择规则", "本轮只训练原train、评估原dev；旧test和v0.4确认集不作新确认")
    r.table(["数据", "训练集", "开发集"], [
        ["源诗 / 诗族", f"{coverage['train']['poems']} / {coverage['train']['families']}", f"{coverage['dev']['poems']} / {coverage['dev']['families']}"],
        ["全部恢复题", str(coverage['train']['tasks']), str(coverage['dev']['tasks'])],
        ["目标在固定词表内", str(coverage['train']['covered_tasks']), str(coverage['dev']['covered_tasks'])],
        ["目标在词表外", str(coverage['train']['tasks']-coverage['train']['covered_tasks']), str(coverage['dev']['tasks']-coverage['dev']['covered_tasks'])],
    ], [180, 155, r.content - 335])
    r.p("固定候选W：五言10,375个，七言1,504个，均来自原train的句内相邻二字、频次至少2。词表外训练目标不注入候选；开发词表外目标记零。")
    r.h("全候选softmax")
    r.p("每个训练查询与对应句长的全部候选竞争：L = -S(q, y) + log Σw exp S(q, w)。只把训练语料的原文二字当作该任务正例；其他可成立的文学写法在这项指标中也会算错。")
    r.table(["预算 / 选择", "冻结设定"], [
        ["训练", "每组seed0/1/2；各1,200步×32题；同seed四组共享完全相同的题目安排。"],
        ["优化器", "Adam；学习率0.003，独立权重衰减0.0001，梯度范数上限5。"],
        ["检查点", "400 / 800 / 1,200步；每组按三种子平均诗族R@16共同选择一步，同分选早。"],
        ["实验约束", "固定主模型full_w3，不混合统计分数，不用控制组结果替换主结论。"],
    ], [110, r.content - 110])
    r.p("开发集已经在前轮使用过，本轮检查点又由它选择；开发数值不能作为独立泛化证明。本地冻结记录也不等同外部预注册。", "small")

    r.page_start("04 / 结果", "公开每个种子的开发成绩", "以下均是诗族等权R@16，原文必须进入前16才算命中")
    r.table(["模型", "选中步数", "seed0", "seed1", "seed2", "平均"], [
        [LABELS[key], str(entry["selected_step"])] + [pct(metric(x["metrics"])) for x in entry["by_seed"]] + [pct(metric(entry["seed_average"]))]
        for key, entry in ordered_arms
    ], [153, 58, 66, 66, 66, r.content - 409])
    r.table(["固定基线", "R@16", "首选命中率", "MRR"], [
        [label, pct(metric(report["baselines"][key])), pct(metric(report["baselines"][key], "top1")), f"{metric(report['baselines'][key], 'mrr'):.4f}"]
        for key, label in [("strong_statistical", "原统计+固定关联"), ("all_position_frequency", "所有位置二字频次"), ("prefix_positive_frequency", "下句开头二字频次")]
    ], [200, 95, 105, r.content - 400])
    r.h("参数规模和归因")
    r.table(["模型", "存储参数", "参与训练的参数"], [
        [LABELS[key], str(entry["by_seed"][0]["parameters"]["stored_parameters"]), str(entry["by_seed"][0]["parameters"]["trainable_parameters"])]
        for key, entry in ordered_arms
    ], [238, 130, r.content - 368])
    r.p("宽度1和宽度3使用接近的参数量，但结构不同。只看上句或后缀的模型从头重新训练，并冻结不使用的查询投影行；它们的活动参数较少。即使匹配模型胜出，也须另看宽度3是否超过宽度1，才能讨论卷积邻域的贡献。", "small")

    r.page_start("05 / 机制", "成绩是否依赖正确的上下文", "训练信息对照与选中权重的推理干预分开报告")
    r.table(["主模型输入", "诗族R@16", "相对正常输入变化"], [
        ["正常上句+后缀", pct(metric(result)), "基准"],
        ["同句长、其他诗族的上下文", pct(metric(mismatch)), pp(metric(mismatch)-metric(result)) + "个百分点"],
        ["遮蔽全部可见上下文", pct(metric(zero)), pp(metric(zero)-metric(result)) + "个百分点"],
    ], [235, 115, r.content - 350])
    r.p("错配映射由任务身份、句长、诗族和可见文本哈希确定，在训练前冻结；是一一置换，保证来自不同诗族且不是同一可见查询。映射不读取目标，也不因为目标碰巧相同就排除。全遮蔽同样只改变查询，不重新训练。")
    r.h("开发资源门")
    checks = gate["checks"]
    r.table(["事先设定条件", "实际", "通过"], [
        ["主模型平均比强基线高至少2个百分点", pp(gate["primary_mean_gain_over_strong_baseline"]) + "pp", "是" if checks["mean_gain_threshold"] else "否"],
        ["三个种子各自都高于强基线", " / ".join(pp(x) for x in gate["seed_gains_over_strong_baseline"]) + "pp", "是" if checks["all_three_seed_gains_positive"] else "否"],
        ["正常输入比错配输入高至少1个百分点", pp(gate["primary_mean_minus_mismatched_context"]) + "pp", "是" if checks["context_mismatch_drop_threshold"] else "否"],
    ], [268, 165, r.content - 433])
    r.p("这三个条件只决定是否值得投入新的独立确认，不是显著性检验，也不是文学质量门槛。确认集尚未运行；不能把开发阶段通过称为最终晋级。", "small")

    r.page_start("06 / 覆盖", "候选在词表内，也可能没有正例", "所有位置的片段词表，大于下句开头二字的监督范围")
    bucket_labels = {"seen_prefix_positive": "在训练前缀任务中作过正例", "in_W_never_prefix_positive": "词表内、未作过前缀正例", "outside_W": "词表外"}
    r.table(["开发目标类别", "题数", "主模型R@16", "统计基线R@16"], [
        [label, str(result["by_prefix_positive_bucket"][key]["tasks"]),
         pct(metric(result["by_prefix_positive_bucket"][key])) if result["by_prefix_positive_bucket"][key]["metrics"] else "无题",
         pct(metric(base["by_prefix_positive_bucket"][key])) if base["by_prefix_positive_bucket"][key]["metrics"] else "无题"]
        for key, label in bucket_labels.items()
    ], [228, 63, 104, r.content - 395])
    r.p("每个切片均对其中包含的诗族等权；不同切片可包含相同诗族，因此不能用切片题数加权还原全体主指标。", "small")
    r.p("片段即使在训练诗的其他位置出现过，也可能从未是下句开头的目标。这类完整片段从未被当作该任务的正确答案，每次只出现在softmax分母的竞争项中；共享字符参数仍可受到其他正例影响。字符组合编码保留了迁移的可能，但不保证迁移成功。")
    r.table(["句长", "开发题数", "主模型R@16", "统计基线R@16"], [
        ["五言" if key == "5" else "七言", str(result["by_length"][key]["tasks"]),
         pct(metric(result["by_length"][key])), pct(metric(base["by_length"][key]))]
        for key in ("5", "7")
    ], [115, 95, 143, r.content - 353])
    r.box("指标的实际含义", "原文恢复要求精确选中来源文本中的二字。别的写法即使通顺或有诗意，也可能被记为失败。因此恢复率、格律合法率、文学偏好不能互相替代。")
    r.p("源语料仍是固定分片的八句五/七字形态筛选，未逐首认证为严格律诗。表面诗族拆分不能穷尽异文或同作者联系；本轮不作作者隔离泛化的宣称。", "small")

    r.page_start("07 / 使用", "输入一句，查看候选成句", "无需重新训练；随包权重可直接加载，默认固定seed0")
    r.p("解压后在wordconv-lab目录运行：", "small")
    r.box("补句命令", "python scripts/complete_span_v05.py --upper '空山新雨後' --suffix '晚來秋'<br/>加 --method matching 只看实验模型；--method statistical 只看统计基线；--json 保存机器可读输出。")
    r.p("输入保留原字形，不自动猜测简繁或异体字。五言上句填5字、后缀填3字；七言上句填7字、后缀填5字，不带标点。默认最多展示16个候选，实际补出完整下句。")
    example_rows = []
    for entry in examples["examples"]:
        data = entry["output"]
        example_rows.append([data["query"]["upper_line"] + " / □□" + data["query"]["lower_suffix"],
            data["methods"]["statistical"]["suggestions"][0]["completed_lower_line"],
            data["methods"]["matching"]["suggestions"][0]["completed_lower_line"]])
    r.table(["固定示例输入", "统计首选", "实验模型首选"], example_rows, [235, 132, r.content - 367])
    r.p("上述4个示例在正式训练前固定，前两个引用古诗公开文本，后两个为构造题面。首个五言示例与一条训练题的可见上下文完全相同；其余3个没有相同训练题面。四例都只用于检查工具接口，不是独立效果样本或精选的优美作品。具体来源重合情况随JSON记录。", "small")
    r.h("为什么默认不挑最好的一次")
    r.p("工具默认seed0在实验开始前已指定，按三种子共同选择的训练步数加载。运行权重、步数和校验值随输出保留；读者可以明确指定seed1/2，但不能把事后挑选的单次成绩当作预定主结果。")

    r.page_start("08 / 核验", "证据与复现入口随包交付", "新文件追加；旧轮实验结论及异常记录继续保留")
    r.table(["核验", "状态"], [
        ["工程测试", f"{tests['tests_run']}项通过；实际运行Python {tests['python']} / NumPy {tests['numpy']}"],
        ["独立结果复算", f"{review.get('full_result_review', {}).get('prediction_rows_checked', '待复核'):,}行逐题记录、{review.get('full_result_review', {}).get('metric_measures_checked', '待复核'):,}项指标通过复算" if 'full_result_review' in review else str(review['status'])],
        ["独立数学检查", "108个坐标的数值梯度；48条固定查询的全部候选重算，完整排名一致"],
        ["部署与边界", "192个已选权重案例；Q E^T、单句/批次、UNK/填充、遮蔽及保存加载检查"],
        ["历史保留", "开始前逐项核对267个v0.4清单文件；历史状态、报告、测试回执另存"],
        ["交付包", "完成后封包、逐项SHA和CRC核验，再发布最终文件"],
    ], [125, r.content - 125])
    r.p(f"训练前另做了有限的train-only工程吞吐检查，成本和正式12次训练分开记录，见results/matching_v05_preflight.json。该检查不是额外独立实验。正式训练耗时累计{accounting['training_seconds']:.1f}秒；含基线与开发评估的主运行阶段{accounting['wall_seconds_including_baselines_and_evaluation']:.1f}秒，不等于每次补句的在线延迟。", "small")
    r.h("复现")
    r.p("python scripts/run_tests.py<br/>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_matching_v05.py prepare --out results/matching_v05_replay<br/>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_matching_v05.py run --out results/matching_v05_replay", "small")
    r.p("源码：src/wordconv_lab/matching.py；选择与结果：results/matching_v05/；补句记录：results/completion_examples_v05.json；协议：docs/MATCHING_v0.5_PROTOCOL.md；审查：docs/MATCHING_v0.5_REVIEW.md。", "small")
    r.h("来源")
    r.p("训练诗歌：<link href='https://github.com/chinese-poetry/chinese-poetry'>chinese-poetry/chinese-poetry</link>，固定提交b8594f81a89752241442f2ce267d6f66f96704ee，来源身份与许可保留。<br/>Kim（2014）：<link href='https://aclanthology.org/D14-1181/'>Convolutional Neural Networks for Sentence Classification</link>。<br/>Karpukhin等（2020）：<link href='https://aclanthology.org/2020.emnlp-main.550/'>Dense Passage Retrieval for Open-Domain Question Answering</link>。", "small")
    r.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--draft", action="store_true", help="permit an explicitly labeled layout draft while the independent review is in progress")
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output, args.fonts, draft=args.draft)
