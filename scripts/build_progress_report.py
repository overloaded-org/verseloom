"""Build the v0.2 Chinese progress report from actual retained run receipts."""
import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from build_report import Report, TEAL, AMBER, INK, MUTED, PALE


def draw_template(r, lines):
    height = 230
    r.ensure(height)
    c, top = r.c, r.y - 8
    cell, x = 24, r.left + 28
    for i, line in enumerate(lines):
        c.setFont("CJK", 8)
        c.setFillColor(MUTED)
        c.drawString(r.left, top - i*cell - 15, str(i+1))
        for j, tone in enumerate(line):
            fill = colors.HexColor("#F7E8CB") if j == len(line)-1 and i%2 else colors.HexColor("#D9EEF0") if j == 1 else PALE
            c.setFillColor(fill)
            c.rect(x+j*cell, top-(i+1)*cell, cell-2, cell-2, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("CJK", 10)
            c.drawCentredString(x+j*cell+11, top-i*cell-16, "平" if tone=="P" else "仄")
    xx, yy = r.left + 185, top - 6
    for title, body in [
        ("五律：仄起、首句不入韵", "这是生成器实际产出的完整基本型。"),
        ("青色：第二字", "联内相反，联间相粘；决定句型的递推状态。"),
        ("金色：偶数句韵脚", "平仄标签还不够，读音记录必须落在统一韵部。"),
        ("其余位置", "本图显示基本型；有限变体审核另有明确范围。")]:
        from reportlab.platypus import Paragraph
        p=Paragraph(f"<b>{title}</b><br/>{body}",r.styles["body"])
        _,h=p.wrap(r.content-185,200)
        p.drawOn(c,xx,yy-h)
        yy-=h+8
    r.y -= height


def build(root, output, fonts):
    def read(path):
        return json.loads((root/path).read_text())
    discovery=read("results/discovery_v02_reviewed/report.json")
    mechanism=read("results/mechanism_v02/report.json")
    reference=read("data/reference_v02/source_audit.json")
    tests=read("results/test_receipt.json")
    full=read("results/full_verse_v02/report.json")
    m=mechanism["primary_test_result"]
    from wordconv_lab.prosody import generate_template
    output.parent.mkdir(parents=True,exist_ok=True)
    r=Report(output,fonts,"v0.2")

    r.page_start("研究结果", "律诗组织算法：可验证的主干", "2026-10-29 / 基本声律文法 + 局部词卷积因子 + 精确动态规划")
    r.p("本轮把项目推进到一个有明确适用范围、可运行且可穷举核对的算法：由声律文法生成完整约束；固定有限候选、语境读音和韵部后，用动态规划组织局部组合。")
    r.box("当前选择", "可分解的局部目标优先使用精确动态规划。此前的得分优先搜索没有达到预设晋级门槛，保留作比较。尚未训练诗歌质量模型，尚无读者盲评结果。")
    r.table(["层次", "实际完成", "结论范围"],[
        ["声律", "五言/七言共8个基本模板", "文献规则的程序化"],
        ["局部组织", "4组原始卷积穷举核对", "有限目标的精确最优"],
        ["完整矩阵", f"{len(full['cases'])}组全篇穷举核对", "四联可加、给定模板/韵部"],
        ["启发式比较", "300题、2400次运行", "优先搜索不升默认"],
        ["数据基础", "106部；保留5个校勘议题", "候选参考，尚非语境金标"],
        ["风险回归", f"{tests['tests_run']}项通过", "工程契约与边界"],
    ],[83,182,r.content-265])
    r.p("本报告所有实验数值来自交付包中的实际回执。常规文法、约束搜索与动态规划是已有方法；这里的贡献是把假设落实为可追溯的算法组合和反例，不作方法首创声明。", "small")

    r.page_start("01 / 声律文法", "用四个句型递推完整矩阵", "先处理近体诗中的五律、七律；古体诗不套用这一固定规格。[S1]")
    r.table(["第二字 s", "句末 e", "五言基本句 F5(s,e)"],[
        ["平", "平", "平平仄仄平"], ["平", "仄", "平平平仄仄"],
        ["仄", "平", "仄仄仄平平"], ["仄", "仄", "仄仄平平仄"],
    ],[92,92,r.content-184])
    r.p("令平=0、仄=1，句号 i 从1开始：<br/><b>s[i] = s[1] XOR (floor(i/2) mod 2)</b><br/>序列为 s、反s、反s、s、s、反s、反s、s。偶数句尾取平；第3、5、7句尾取仄；首句按是否入韵选择。")
    draw_template(r,generate_template(5,"Z",False).core)
    r.p("七言基本型：F7(s,e) = ss + F5(反s,e)。首句入韵只改变首句，不能把首入的四句块直接重复到第五句。common_prefix审核只允许有出处的少数前部变化，未实现完整拗救。", "small")

    r.page_start("02 / 局部评分", "词卷积的可解结构", "优化的是给定内部评分，文学评价需要独立证据。")
    r.p("把一联同列的两个字记为 s[j]=(上句字,下句字)。当字位特征只依赖该字、位置和固定题意，卷积窗口宽度为3，而且输出按位置求和时，当前评分器可以精确分解：")
    r.box("局部因子形式", "Q(s) = Σ[j] f[j](s[j−1], s[j], s[j+1])<br/>句首、句尾使用空边界。每个因子同时保留句内窗口与上下句对齐的非线性；没有用单字分数替换原卷积。")
    r.h("动态规划为何成立")
    r.p("一个窗口的右邻确定后，该窗口分数就能结算。继续向右时，未来只需要知道最后两列；具有相同最后两列的前缀中，只保留总分最高者。被删前缀不可能在同一未来续接下逆转。")
    r.box("核心递推", "V[j](b,c) = max[a] { V[j−1](a,b) + f[j−1](a,b,c) }<br/>起点结算首窗口，终点补计末窗口，再按保留的路径取回候选。显式负无穷表示局部不可行；NaN、正无穷与累计溢出被拒绝。")
    r.table(["原始卷积穷举", "完整候选数", "DP局部因子调用", "最优误差"],[
        [f"{x['length']}字/种子{x['seed']}",str(x['full_combinations']),str(x['factor_calls']),f"{x['optimality_abs_error']:.1e}"] for x in discovery['dp_checks']
    ],[126,95,135,r.content-356])
    r.p("每列有4个上下句组合。所有组合由原始LocalConvScorer直接穷举，独立于因子递推。因子调用和整模型调用计量单位不同，不作调用数速度比；以上只是小规模正确性证据。", "small")

    r.page_start("03 / 完整矩阵", "全篇求解依赖明确的分解假设", "8×5 / 8×7矩阵接口；基本模板、同一韵部、四联局部目标。")
    r.p("外层枚举调用者给定的完整基本模板 T 与韵部 R。每一格按该模板及已消歧读音过滤候选，韵脚同时要求属于 R。四联分别求解，再比较所有外层配置的总分。")
    r.box("全篇优化式", "max[T,R] Σ[k=1..4] max[X_k ∈ D_k(T,R)] Q_k(X_k)<br/>成立条件：固定T与R后，可行域是四联可行域的乘积，总分恰好等于四联局部分数之和。")
    r.table(["真实运行", "完整组合数", "可行组合数", "最优误差"],[
        [f"8×{x['length']}",str(x['full_combinations']),str(x['feasible_combinations']),f"{x['optimality_abs_error']:.1e}"] for x in full['cases']
    ],[92,134,134,r.content-360])
    r.h("读音是输入证据")
    r.p("候选含字、平仄、韵部、读音记录身份及验证状态。未知、未验证或同一字位同一字存在多种未消歧读音，返回待复核。缺候选返回不可行，不先删掉不确定候选再声称搜索了完整范围。")
    r.h("全篇最优的边界")
    r.p("这里的全局最优只覆盖有限候选域及声明目标。跨联主题推进、全篇禁重字、未建模词跨度和篇章语义，不能先在求解后检查，再称为这些约束下的最优解。需要把它们加入状态或交给更一般的求解器。")
    r.p("演示读音与字形均为明确标记的合成输入，不认证任何真实诗作。对仗目前只有对齐评分接口，没有经训练或人工验证的对仗质量判定。", "small")

    r.page_start("04 / 反证实验", "得分优先搜索没有晋级", "本地运行前冻结配置；这不是外部平台的预注册。")
    r.p("300道合成题分为60道开发题、240道测试题；四类结构各占四分之一。两个策略共享起点、候选、动作、约束与评分器，按8/16/32/64次尝试预算比较；主预算32。全部300题核对可达域，2400次运行全部重新执行核对轨迹。")
    r.table(["主预算32 / 测试集", "FIFO", "得分优先"],[
        ["平均归一化最优差距",f"{m['fifo_mean_regret']:.5f}",f"{m['priority_mean_regret']:.5f}"],
        ["可分离结构",f"{m['family_results']['separable']['fifo_mean_regret']:.5f}",f"{m['family_results']['separable']['priority_mean_regret']:.5f}"],
        ["局部相互作用",f"{m['family_results']['local_interaction']['fifo_mean_regret']:.5f}",f"{m['family_results']['local_interaction']['priority_mean_regret']:.5f}"],
        ["耦合硬约束",f"{m['family_results']['coupled_constraint']['fifo_mean_regret']:.5f}",f"{m['family_results']['coupled_constraint']['priority_mean_regret']:.5f}"],
        ["局部高分陷阱",f"{m['family_results']['local_trap']['fifo_mean_regret']:.5f}",f"{m['family_results']['local_trap']['priority_mean_regret']:.5f}"],
    ],[231,134,r.content-365])
    r.p("最优差距越低越好。平均相对下降11.45%，95%配对重采样区间为1.66%～19.51%；未达到事先设定的至少20%。局部陷阱类别的差距增加0.05169，超过0.05退步门槛。")
    r.box("决策：不升为默认策略", "局部优先策略有平均收益，也有结构性反例。保留其运行证据；可分解问题改用精确DP。通用搜索比较显式指定FIFO，历史matrix.py默认签名保持不动以便冻结实验复现。",AMBER)
    r.p("此测试集现已暴露。后续调策略不能把这些结果再次称作独立确认。案例test-local_trap-004为事后解释：FIFO在32次尝试内到达最优，得分优先停在起点，差距0.34507。它不是额外盲测。", "small")

    r.page_start("05 / 数据", "106部是参考层，不是读音金标", "固定原始修订、保留原字形、可离线重建。[S2]")
    counts=reference['counts']
    r.table(["项目", "实际审计", "使用含义"],[
        ["韵部", "106 = 15+15+29+30+17", "结构齐全，内容仍待校勘"],
        ["原字形",f"{counts['main_unique_characters']:,}","不做隐式繁简或异体合并"],
        ["字—部关系",f"{counts['main_unique_character_group_pairs']:,}","同部重复去重、跨部保留"],
        ["跨部 / 跨平仄", "1183字 / 573字", "需要语境读音依据"],
        ["独立补充行", "111行", "默认不并入主索引"],
        ["校勘议题 / 自动改字", "5项 / 0项", "疑点可追溯，不静默修补"],
    ],[115,151,r.content-266])
    r.p("旧来源仅105部，缺上声三讲；新来源补全该部，但仍发现‘先’在所属平声部名单中缺漏、‘啼’未收、‘頹’与‘龍’存在需复核的归属，另有部名惯例议题。新表所有记录均标记source_candidate_needs_context。")
    r.h("真实诗例与规格反例")
    r.p("本轮收集6首来源明确的诗文，其中仅2首附有讲义逐字声调标注，属于讲解用例。按该标注，《曲江》被common_prefix覆盖；《和晋陵陆丞早春游望》的‘云霞出海曙’超出窄规格。程序返回规格范围外，不能据此宣布原诗失律。[S1]")
    r.p("这6首不是独立测试集，也没有被伪装为几百首已标注数据。旧105部表与新106部表来自不同录文和字形口径，不能用收字数差值表示准确率提升。", "small")

    r.page_start("06 / 项目决策", "把尚未证明的部分留给下一道检验", "算法正确、格律覆盖与文学价值各有自己的证据门槛。")
    r.table(["研究项", "状态", "下一项可验收结果"],[
        ["基本声律文法", "已实现，范围明确", "完整变格/拗救规则与独立标注集"],
        ["局部/全篇有限优化", "已实现并穷举核对", "扩大候选后记录时间和内存"],
        ["得分优先策略", "未过门槛", "若改策略，冻结新的独立题集"],
        ["106部候选参考", "结构完整；非金标", "疑点校勘、语境读音及来源记录"],
        ["词卷积质量", "未训练、未评价", "同池训练比较与独立匿名盲评"],
        ["对仗/篇章语义", "接口/设计阶段", "可复核标签和非局部约束建模"],
    ],[119,139,r.content-258])
    r.h("复杂度需要如实计算")
    r.p("令D为同一列上下句候选的组合数，局部DP的因子转移上界为O(LD³)，前沿至多O(D²)。若上下各k个候选，D=k²。当前实现缓存所有因子并保存完整路径，因此总内存不是O(D²)，peak_states也不是进程峰值内存。")
    r.box("下一阶段的主要问题", "内部评分何时代表更好的语言？需在相同候选池和数据分割上，比较卷积、强局部非卷积基线，并让评审在不知道系统身份的情况下评价搭配、对仗、篇章与意境。不能用未训练卷积的自评分给自己作证。",AMBER)
    r.p("项目数据门尚未完整通过。完成有限结构求解不等于找到诗人脑中的算法，也不证明文学表达能够被当前有限特征完整概括。")

    r.page_start("07 / 复核入口", "结果、源码与来源一一对应", "同一项目保留v0.1协议、v0.2证据与明确的变更记录。")
    r.table(["路径", "内容"],[
        ["src/wordconv_lab/prosody.py", "基本模板生成、窄规格审核"],
        ["src/wordconv_lab/factor_dp.py", "局部因子分解与精确DP"],
        ["src/wordconv_lab/verse_solver.py", "八句矩阵与有限全局变量枚举"],
        ["results/discovery_v02_reviewed/", "修复数值边界后的4组原始模型核对"],
        ["results/full_verse_v02/", "全篇合成穷举核对与配置冻结"],
        ["results/mechanism_v02/", "冻结配置、300题、2400次运行与轨迹"],
        ["data/reference_v02/", "固定修订、索引、问题、许可与SHA"],
        ["results/test_receipt.json", f"{tests['tests_run']}项实际测试与代码校验值"],
    ],[252,r.content-252])
    r.h("复算命令")
    r.p("python scripts/run_tests.py<br/>python scripts/build_rhyme_reference.py --check<br/>PYTHONPATH=src python scripts/run_discovery_v02.py --out results/discovery_replay<br/>PYTHONPATH=src python scripts/run_full_verse_v02.py --out results/full_verse_replay", "small")
    r.h("文献及使用边界")
    r.p("[S1] 曾永义，国立台湾大学《韵文学专题（上）》第二讲：近体诗之体制规律；基本句型及变式物理第3—5页，例诗第6—7页。<br/><link href='https://ocw.aca.ntu.edu.tw/uploads/course_item_file/file/684/104S110_CT02L01.pdf'>ocw.aca.ntu.edu.tw / 104S110_CT02L01.pdf</link>", "small")
    r.p("[S2] 维基文库《平水韵》，固定修订8264804。页面未标明底本；只作候选参考。<br/><link href='https://zh.wikisource.org/w/index.php?oldid=8264804'>zh.wikisource.org / oldid=8264804</link><br/>第三方录文及标注保留各自署名与许可，不能静默套用项目代码许可。", "small")
    r.save()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--fonts",type=Path,required=True)
    args=p.parse_args()
    build(Path(__file__).resolve().parents[1],args.output,args.fonts)
