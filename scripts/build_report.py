"""Build the Chinese review report. Optional deps: reportlab and Noto Sans SC.

This is documentation tooling, independent of the NumPy-only experiment runtime.
Pass --fonts with NotoSansSC-Regular.ttf and NotoSansSC-Bold.ttf.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#087F8C")
AMBER = colors.HexColor("#BA7B20")
INK = colors.HexColor("#28394B")
MUTED = colors.HexColor("#637487")
PALE = colors.HexColor("#F2F6FA")
LINE = colors.HexColor("#DAE3EC")


class Report:
    def __init__(self, path: Path, fonts: Path, version: str = "v0.1"):
        pdfmetrics.registerFont(TTFont("CJK", str(fonts / "NotoSansSC-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("CJKBold", str(fonts / "NotoSansSC-Bold.ttf")))
        pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJKBold", italic="CJK", boldItalic="CJKBold")
        self.width, self.height = A4
        self.left, self.right, self.bottom = 48, 48, 53
        self.content = self.width - self.left - self.right
        self.c = canvas.Canvas(str(path), pagesize=A4)
        self.version = version
        self.c.setTitle(f"诗矩验证：矩阵组织算法与词卷积 | {version}")
        self.c.setAuthor("VerseMatrix Lab")
        self.c.setSubject("项目验证协议、可运行工程原型与真实数据源审计")
        self.page = 0
        self.y = 0
        self.styles = {
            "body": ParagraphStyle("body", fontName="CJK", fontSize=10.4, leading=17.2,
                                   textColor=INK, wordWrap="CJK", alignment=TA_LEFT, spaceAfter=8),
            "small": ParagraphStyle("small", fontName="CJK", fontSize=8.5, leading=13.4,
                                    textColor=MUTED, wordWrap="CJK"),
            "head": ParagraphStyle("head", fontName="CJKBold", fontSize=13.2, leading=20,
                                   textColor=NAVY, wordWrap="CJK"),
            "table": ParagraphStyle("table", fontName="CJK", fontSize=9.2, leading=14.2,
                                    textColor=INK, wordWrap="CJK"),
            "tablehead": ParagraphStyle("tablehead", fontName="CJKBold", fontSize=9.2, leading=14,
                                        textColor=colors.white, wordWrap="CJK"),
        }

    def page_start(self, number: str, title: str, subtitle: str = ""):
        if self.page:
            self.c.showPage()
        self.page += 1
        c = self.c
        c.setFillColor(TEAL)
        c.rect(self.left, self.height - 44, 26, 4, fill=1, stroke=0)
        c.setFont("CJK", 8)
        c.setFillColor(MUTED)
        c.drawString(self.left + 36, self.height - 43, f"VERSE MATRIX LAB  /  研究记录 {self.version}")
        c.drawRightString(self.width - self.right, self.height - 43, "2026-10-29")
        c.setFont("CJKBold", 23)
        c.setFillColor(NAVY)
        c.drawString(self.left, self.height - 88, title)
        c.setFont("CJK", 9)
        c.setFillColor(TEAL)
        c.drawRightString(self.width - self.right, self.height - 87, number)
        self.y = self.height - 109
        if subtitle:
            self.p(subtitle, "small", after=14)
        c.setStrokeColor(LINE)
        c.line(self.left, 38, self.width - self.right, 38)
        c.setFillColor(MUTED)
        c.setFont("CJK", 8)
        c.drawString(self.left, 24, "工程证据已核验  |  真实诗歌效果尚未评估")
        c.drawRightString(self.width - self.right, 24, f"{self.page:02d}")

    def ensure(self, h: float):
        if self.y - h < self.bottom:
            raise RuntimeError(f"Page {self.page} overflow by {self.bottom-(self.y-h):.1f} pt")

    def p(self, text: str, style: str = "body", after: float = 9):
        paragraph = Paragraph(text, self.styles[style])
        _, h = paragraph.wrap(self.content, 1000)
        self.ensure(h + after)
        paragraph.drawOn(self.c, self.left, self.y - h)
        self.y -= h + after

    def h(self, text: str):
        self.p(text, "head", after=6)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float] | None = None):
        if widths is None:
            widths = [self.content / len(headers)] * len(headers)
        data = [[Paragraph(escape(str(v)), self.styles["tablehead"]) for v in headers]]
        data.extend([[Paragraph(escape(str(v)), self.styles["table"]) for v in row] for row in rows])
        table = Table(data, colWidths=widths, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, LINE),
        ]))
        _, h = table.wrap(self.content, 1000)
        self.ensure(h + 14)
        table.drawOn(self.c, self.left, self.y - h)
        self.y -= h + 14

    def box(self, title: str, text: str, color=TEAL):
        p1 = Paragraph(title, self.styles["head"])
        p2 = Paragraph(text, self.styles["body"])
        _, h1 = p1.wrap(self.content - 30, 1000)
        _, h2 = p2.wrap(self.content - 30, 1000)
        h = h1 + h2 + 29
        self.ensure(h + 13)
        self.c.setFillColor(PALE)
        self.c.roundRect(self.left, self.y - h, self.content, h, 5, fill=1, stroke=0)
        self.c.setFillColor(color)
        self.c.rect(self.left, self.y - h, 3, h, fill=1, stroke=0)
        p1.drawOn(self.c, self.left + 15, self.y - 12 - h1)
        p2.drawOn(self.c, self.left + 15, self.y - 18 - h1 - h2)
        self.y -= h + 13

    def matrix(self):
        h = 233
        self.ensure(h)
        c, top = self.c, self.y - 9
        cell = 23
        x = self.left + 25
        local = {10, 11, 15, 16}
        rhyme = {9, 19, 29, 39}
        for row in range(8):
            c.setFillColor(MUTED)
            c.setFont("CJK", 8)
            c.drawString(self.left, top - row * cell - 15, str(row + 1))
            for col in range(5):
                index = row * 5 + col
                fill = colors.HexColor("#D9EEF0") if index in local else colors.HexColor("#F8E8CC") if index in rhyme else PALE
                c.setFillColor(fill)
                c.setStrokeColor(colors.white)
                c.rect(x + col * cell, top - (row + 1) * cell, cell - 2, cell - 2, fill=1, stroke=0)
                c.setFillColor(INK)
                c.setFont("CJK", 8)
                c.drawCentredString(x + col * cell + (cell - 2) / 2, top - row * cell - 15,
                                    "A/B" if index in local | rhyme else "A")
        right = self.left + 180
        content = [
            ("完整状态：8行 × 5列", "32个字位冻结，8个位置具有A/B候选。"),
            ("局部可编辑：4个字位", "集中在同一联，供局部评分模块读取。"),
            ("全局联动：4个韵脚", "四个位置必须同时属于同一合成类别。"),
            ("精确核验：256 → 32", "全部256种组合中，32种满足合成约束。"),
        ]
        yy = top - 3
        for title, text in content:
            para = Paragraph(f"<b>{title}</b><br/>{text}", self.styles["body"])
            _, ph = para.wrap(self.content - 180, 200)
            para.drawOn(c, right, yy - ph)
            yy -= ph + 10
        c.setFillColor(MUTED)
        c.setFont("CJK", 8)
        c.drawString(self.left, top - 8 * cell - 20, "图1  实际运行的抽象字位矩阵。A/B不代表真实汉字、韵部或诗歌质量标签。")
        self.y -= h

    def save(self):
        self.c.save()


def build(root: Path, output: Path, fonts: Path):
    tests = json.loads((root / "results/test_receipt.json").read_text())
    matrix = json.loads((root / "results/matrix/matrix_engineering_report.json").read_text())
    audit = json.loads((root / "results/source_audit.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    r = Report(output, fonts)

    r.page_start("项目总览", "诗矩验证", "矩阵组织算法 × 词卷积  /  设计协议与工程证据")
    r.p("<b>验证一个组织过程：</b>把字词放入声律与结构约束定义的位置，通过候选选择、分支探索与反馈形成表达。主项目是矩阵组织算法；词卷积是可独立替换的局部组合评分模块。")
    r.box("当前决策：进入数据与规格阶段", "工程原型已经运行。接下来先核验真实诗族、字音和候选来源，再开展学习与盲评。当前没有训练诗歌模型，也没有真实诗歌质量提升的证据。")
    r.table(["本轮证据", "实际结果", "解释范围"], [
        ["风险测试", f"{tests['tests_run']} / {tests['tests_run']} 通过", "机制与防污染检查"],
        ["矩阵运行", "8次运行；4次穷举核对通过", "预算、合法状态、重放"],
        ["数据审计", "1000首分片 + 候选韵表", "发现真实输入缺陷"],
        ["真实诗歌效果", "尚未训练 / 尚未盲评", "不能作质量结论"],
    ], [111, 179, r.content - 290])
    r.h("三个问题分别回答")
    r.p("<b>H-M：</b>改变矩阵状态的展开顺序，是否在相同预算内改善搜索？<br/><b>H-C：</b>加入词卷积，是否在同一候选池中选出更好的表达？<br/><b>H-I：</b>两者结合，是否有共同外部评价支持的额外收益？")
    r.p("交付包括完整协议、NumPy原型、风险测试、实际运行轨迹、冻结来源快照与可复算审计。文本CNN与卷积语言模型提供技术先例；本项目的具体声律组织假设仍需单独验证。[S1][S2]", "small")

    r.page_start("01 / 表示", "矩阵是状态，算法是更新规则", "先定义对象，避免把二维排版直接当成算法贡献。")
    r.table(["对象", "内容", "需要守住的边界"], [
        ["X：字位矩阵", "8×5 / 8×7；每格一个字", "词可以跨格，不能默认一词一格"],
        ["D：候选域", "每格或词跨度的有限候选", "来源先冻结，不读取隐藏答案"],
        ["R：读音记录", "声调、韵部、语境依据、出处", "平仄与韵部必须来自同一读音"],
        ["H：硬约束", "完整模板、韵脚位置、统一韵部", "未知或未消歧不能静默通过"],
        ["F / Q", "多层特征与候选内部评分", "内部高分不等于文学质量高"],
        ["B / E", "有限预算与事件轨迹", "每次尝试计账，结果可以回放"],
    ], [90, 166, r.content - 256])
    r.h("声律规格必须保留关系")
    r.p("律诗的句内平仄、联内相对、联间粘接以及韵脚关系，共同构成约束。允许变式时，应保存有依据的完整模板或关系，不能把不同模板逐格取并集后拼出新模板。[S3][S4]")
    r.p("实际代码检查调用者提供的显式规格；它不证明该规格本身具有历史权威。多音字需要语境消歧，传统韵部与普通话拼音也不能直接互换。")
    r.box("第一条解释限制", "矩阵表示成立，只说明我们能够组织和检查这些关系。它不能独自证明某种搜索规则更好，也不能证明诗人实际在脑中执行该算法。", AMBER)

    r.page_start("02 / 组织算法", "完整矩阵上的有限搜索", "本轮实现：保留备选分支的FIFO / 得分优先调度。")
    r.matrix()
    r.table(["M0 / FIFO", "M1 / Priority"], [
        ["按发现顺序展开备选状态", "优先展开内部评分较高的备选状态"],
        ["两者都有完整起点、相同邻域与备选分支", "两者都能从之前保留的状态继续探索"],
        ["相同单格与四韵脚联合动作", "不额外获得更多动作、上下文或候选"],
    ])
    r.p("四个合成韵脚若必须一致，单格改变会产生非法中间态。原型显式提供四位置联合动作，且两个调度器都能使用。否则会出现合法状态图不连通的问题。")
    r.box("必须区分：可达最优与全域最优", "frontier_exhausted仅表示起点所在的合法可达分量耗尽。全域oracle独立枚举所有组合；没有连通性证明时，二者不能画等号。", AMBER)

    r.page_start("03 / 可审计性", "每次尝试都有可检查的记录", "拒绝、重复和探索低分分支，也属于算法成本。")
    r.table(["步骤", "具体行为", "记录或门槛"], [
        ["1 初始化", "检查完整起点，评分并保存best", "非法起点报错；初始评分计费"],
        ["2 展开", "按M0/M1选择已知合法状态", "同一状态图、同一动作集合"],
        ["3 尝试", "枚举替换，先扣proposal", "重复与非法候选也扣预算"],
        ["4 检查 / 评分", "合法新状态评分并进入分支", "非有限评分拒绝；记录词典身份"],
        ["5 更新best", "只保存更好的合法结果", "低分探索不能覆盖已知最好结果"],
        ["6 停止", "任一预算耗尽或可达分支耗尽", "返还best、计数、路径和完整trace"],
    ], [76, 208, r.content - 284])
    r.h("两种预算同时保留")
    r.p("proposal统计所有候选尝试；score_calls统计初始及合法新候选的评分。缓存命中仍计proposal，回到旧分支不返还预算。另记录真实耗时，避免把相同调用数误当相同算力。")
    r.h("回放的实际含义")
    r.p("逐事件哈希链覆盖展开、拒绝、入队与终止。replay重新运行约束和评分，并比较完整轨迹。best_path只是最佳解的父指针来源，实际探索顺序以trace为准。")
    r.p("本轮没有实现冲突优先调度、原地状态栈回滚或崩溃恢复。trace不是签名，也不是完整权重来源证明；代码与环境另存运行回执。精确浮点回放已在本次同一环境中验证，跨平台尚未承诺。", "small")

    r.page_start("04 / 词卷积", "让局部组合影响候选评分", "先把算子定义清楚，再讨论它是否改善诗句。")
    r.p("目标表示是字位上的多通道特征：语义、词跨度、给定音韵条件、句内位置及联的角色。共享局部核对邻域进行组合，产生用于候选打分的特征。")
    r.box("算子定义", "h[i,j] = activation( Σ W[a,b] · F[i+a,j+b] + bias )<br/>采用神经网络通常称为卷积的局部互相关约定。核沿字位关系移动，特征维度作为通道处理。")
    r.table(["关系", "实现方向", "边界"], [
        ["句内组合", "逐句横向共享核", "不把两句首尾拼成局部邻域"],
        ["同联对应", "指定两句的对齐核", "不把任意相邻两行都当对仗联"],
        ["位置差异", "位置与角色条件", "韵脚、句首并非职责相同"],
        ["全篇约束", "显式韵部及关系检查", "感受野扩大不等于合法性保证"],
    ], [88, 173, r.content - 261])
    r.h("当前原型的准确状态")
    r.p("已实现宽度3的横向核、同联对齐核与读出；特征是抽象token identity、合成读音及位置。权重随机初始化，trained=False。尚无真实语义编码、词跨度学习和训练器。")
    r.p("Kim的文本CNN说明词向量上的局部共享特征可用于句子任务；门控卷积语言模型说明卷积也可参与语言建模。这些文献并未验证本项目的声律矩阵或创作质量。[S1][S2]", "small")

    r.page_start("05 / H-M", "单独验证矩阵组织是否有效", "固定评分器和状态空间，只改变分支调度。")
    r.h("正式机制集设计")
    r.p("在完整8×5或8×7状态中，开放2至4个字词跨度，每跨度3至5个候选，最多625种组合。开发60题、封存测试240题；题型包含可分离评分、局部关联、联合约束与局部最优陷阱。题集尚未执行。")
    r.table(["必须相同", "必须报告"], [
        ["完整起点、候选域、上下文", "合法返回率与覆盖范围"],
        ["动作集合、约束及缓存", "预算内最优差距与唯一合法状态数"],
        ["评分器、proposal与评分预算", "首次达到目标所需调用与真实耗时"],
        ["oracle只在评价端可见", "可达分量与全域oracle的差异"],
    ])
    r.p("共同proposal预算为8 / 16 / 32 / 64，主预算32；评分调用也设相同上限。归一化regret仅在同一题、同一评分器内计算。最优与最差相等的题单独报告。")
    r.box("拟定Go / Stop门槛", "Go：所有工程性质通过；主预算下平均归一化regret相对降低至少20%，配对区间支持改善。<br/>Stop或修订：优势依赖额外动作、不同起点、更多调用，或出现预算超支、非法best、无法重放。")
    r.p("20%是项目决策阈值，不是数学定律。若基线已经几乎零regret，应报告天花板。本阶段通过只能支持该调度规则在冻结机制任务上的作用，不能推出真实诗歌更好。", "small")

    r.page_start("06 / H-C", "固定候选，单独检验词卷积", "先隔离评分价值，避免同时改搜索器与候选池。")
    r.p("任务：给定相同上句、题意/允许上下文、格律及韵部，从16至64个相同完整下句候选中选出一个。候选生成器只读取训练词表和题面，不能读取隐藏原句，也不能保证把原答案插入开放候选池。")
    r.table(["对照", "能回答什么", "角色"], [
        ["独立字位评分", "局部上下文是否有帮助", "弱基线"],
        ["平滑字符n-gram", "局部统计能否解释收益", "必要参照"],
        ["同感受野局部注意力", "共享卷积相对强局部模型的价值", "主非卷积基线"],
        ["Locally-connected", "参数共享本身的作用", "单独消融"],
    ], [128, 242, r.content - 370])
    r.h("训练与公平性")
    r.p("正式模型优先从审核语料自监督学习遮蔽恢复，不把合成候选标成“坏诗”。隐藏目标的身份及由它推导的额外读音必须遮蔽。所有模型使用一致的完整候选评分接口，不能让一组看全句、另一组只看前缀。")
    r.p("拟定上限：每模型200万参数、2000次optimizer update、batch 32、累积步数1、5个种子；开发试跑后冻结。共同控制编码器、样本曝光与调参次数，另报实测时间。预算不是效果或耗时承诺。")
    r.box("一个容易忽略的伪对照", "共享窗口的一层MLP常与卷积数学等价，不能把它包装成独立的强非卷积模型。当前演示的随机pointwise也只用于接线检查。", AMBER)

    r.page_start("07 / H-I", "把两个模块放进2×2设计", "内部评分不同，交互必须由共同外部评价定义。")
    r.table(["组织方式", "强非卷积评分 C0", "卷积评分 C1"], [
        ["M0 / FIFO", "M0C0", "M0C1"],
        ["M1 / Priority", "M1C0", "M1C1"],
    ], [149, 175, r.content - 324])
    r.p("四组共享起点、候选域、硬约束、上下文、合法动作和预算；都可以保留备选分支。先分别完成H-M和H-C，再开展联合实验。")
    r.box("交互定义", "ΔI = (Q_M1C1 - Q_M0C1) - (Q_M1C0 - Q_M0C0)<br/>Q必须来自相同外部标准。不能直接相减conv_score与unary_score。")
    r.h("共同评价怎么来")
    r.p("机制任务可使用同一个对调度器隐藏的oracle。真实诗句采用打乱模型身份的统一量表：连贯性0至4分，依次锚定不可读、关系断裂、基本可解但松散、连贯、连贯且层次清楚；归一化后得到共同Q。另报配对偏好作为稳健性分析。")
    r.p("该量表是研究操作定义，并非文学价值的客观计量。按源诗聚类估计交互区间；四臂自动分析需要在正式运行前补齐并冻结。")
    r.h("三种结果都可接受")
    r.p("矩阵有效而卷积无增益：保留组织器、更换评分器。卷积有效而调度无增益：保留评分模块、采用简单组织。两者各自有效但交互不显著：分别保留，不声称协同收益。")

    r.page_start("08 / 数据审计", "真实输入已经暴露出问题", "只审计小规模冻结快照，尚未建立可用benchmark。")
    r.table(["检查对象", "本次实测", "影响"], [
        ["诗库分片", "1000条；全文唯一981条", "19组重复必须整族切分"],
        ["结构候选", "8×5：217；8×7：47", "形状不等于核验律诗"],
        ["候选韵表", "105部；缺上声三讲", "不能直接用作完整规格"],
        ["多归属字符", "8232字中1823字", "需保留不同声部/韵部集合"],
        ["原样查表", "7329 / 11312字位命中", "繁简兼容诊断，不能叫合律率"],
    ], [115, 182, r.content - 297])
    r.p("诗库冻结：<b>b8594f81a89752241442f2ce267d6f66f96704ee</b><br/>韵表冻结：<b>ff0e9c13fb037c43e0eaa5dc929c0fe4fa2ffb18</b>", "small")
    r.p("两个候选仓库均附MIT声明，但许可、文字版本可靠性和读音可靠性是不同问题。诗库README提示互联网采集与文本错误；paragraphs常包含一整联，必须再拆成矩阵行。[S5][S6]")
    r.p("传统平水韵为106部；冻结候选表的上声只有28部，缺三讲。原文件和逐文件SHA均保留，便于复核这个缺陷。[S4][S7]")
    r.box("进入真实实验前的条件", "原文不覆盖；繁简与异体只建辅助视图。原诗、异文和派生题按同族划分；多音必须有语境证据。当前快照状态明确为audit_only_not_benchmark。", AMBER)

    r.page_start("09 / 质量判定", "让质量结论有独立依据", "正式盲评之前冻结主指标、主基线、样本与停止规则。")
    r.h("主指标与统计单位")
    r.p("相同上句和候选池中，卷积输出与预先指定强基线输出匿名比较；胜=1，平=0.5，负=0。目标1000个独立诗族，每题3名评审。左右随机，不展示模型名或参考下句，记录是否认出原诗。数量是拟定预算，尚未有人工作评。")
    r.p("先按源诗等权汇总，再做聚类bootstrap。同诗多个候选、评审或训练种子不是独立样本。缺失不伪造为平局；空候选覆盖率与模型执行失败单列，不能静默删题。")
    r.table(["报告量", "拟定通过条件", "对应关系"], [
        ["偏好分 p", "p ≥ 0.55；95%区间下界 > 0.50", "胜1 / 平0.5 / 负0"],
        ["配对差 δ", "δ ≥ 0.10；区间下界 > 0", "δ = 2p - 1"],
        ["结构与连贯性", "没有预定义的实质退化", "不能用总分掩盖结构失效"],
    ], [90, 255, r.content - 345])
    r.h("本轮工具守住的边界")
    r.p("盲评v2把比较身份、实际显示文本及左右方向绑定，拒绝修改后的展示或误配旧key。空白评价表返回no_evaluation_data，不输出偏好率。")
    r.p("当前bootstrap未建模评审者交叉效应或训练种子不确定性，正式阶段需要敏感性分析。开发试评后、正式测试解封前冻结样本精度；区间跨0.50记作证据不足，不能一边看结果一边补样本直到显著。", "small")

    r.page_start("10 / 已执行证据", "当前证据支持工程通过", "这页只报告实际执行，计划与假设不计入成绩。")
    r.table(["检查", "结果", "来源"], [
        ["自动风险测试", f"{tests['tests_run']}通过，0失败，0错误", "test_receipt.json / tests.log"],
        ["完整矩阵演示", "256组合；32合法；8次运行", "matrix_engineering_report.json"],
        ["充分预算核对", "4次均覆盖32合法状态并达各自最优", "独立穷举oracle"],
        ["重执行回放", "8条轨迹逐事件一致", "每次运行的JSONL轨迹"],
        ["候选演示", "5候选：2合法、2非法、1待复核", "wordconv工程报告"],
        ["人工评价", "0条；不产生数值质量结论", "空白盲评表"],
    ], [113, 219, r.content - 332])
    r.h("修复后保留的回归风险")
    r.p("非整数/布尔预算拒绝；最后分支在预算中断时不误报穷尽；可达分量不冒充全域；同候选但不同读音表不能判作公平对照；盲评文本、比较身份和方向必须一致。")
    r.h("尚未完成")
    r.p("正式训练、强局部基线、词跨度语义、真实音韵消歧、近重复发现器、正式机制测试集、独立盲评及四臂交互分析，均为NOT_RUN或未实现。随机卷积在工程样例中分数较高不能支持更会写诗。")
    r.p(f"实际环境：Python {tests['python']} / NumPy {tests['numpy']}。代码SHA逐文件存于运行回执。源快照7项字节数与SHA核验通过；审计脚本输出与本报告数据一致。", "small")

    r.page_start("11 / 执行路线", "下一步先把数据门做实", "用阶段门控制结论，而不是用更大的模型掩盖输入问题。")
    r.table(["阶段", "交付", "允许推进的依据"], [
        ["G0 已完成", "原型、风险测试、轨迹、空盲评表", "工程管线可运行"],
        ["G1 下一项", "可溯源五律种子、完整韵表、语境读音", "真实输入具备资格"],
        ["G2", "冻结机制集与共同预算", "矩阵调度是否有效"],
        ["G3", "训练强基线与卷积，独立盲评", "卷积是否改善受限选词"],
        ["G4", "共同外评2×2，再扩展完整八行", "组合是否有额外作用"],
    ], [83, 237, r.content - 320])
    r.p("立即工作单元：先整理200首具有版本依据的五律，核对原文和读音；修订候选韵表缺部、重复和繁简映射。200首只用于开发与流程核验，不能兼任最终独立测试。")
    r.h("复现命令")
    commands = [
        "python scripts/run_tests.py",
        "PYTHONPATH=src python scripts/run_matrix_demo.py --out results/matrix",
        "PYTHONPATH=src python -m wordconv_lab demo --out results/wordconv",
        "python scripts/audit_sources.py",
    ]
    for cmd in commands:
        r.p(escape(cmd), "small", after=5)
    r.p("当前实验仅依赖Python与NumPy。正式训练的时间、内存与成本需要开发试跑实测；目前没有GPU训练耗时承诺。完整命令、模型接口和限制在项目README及docs/PROTOCOL.md。")
    r.box("本版延期", "完整开放创作、自动审美评分、动态核、图卷积、多体裁混合、外部大模型评委、产品界面与网站发布。它们不能替代H-M与H-C的独立检验。")

    r.page_start("12 / 来源与交付", "证据与项目文件", "研究来源支持背景与规格；实验规模和门槛属于本项目设计。")
    sources = [
        ("S1", "Yoon Kim. Convolutional Neural Networks for Sentence Classification. EMNLP 2014.", "https://aclanthology.org/D14-1181/", "ACL Anthology / D14-1181"),
        ("S2", "Yann N. Dauphin et al. Language Modeling with Gated Convolutional Networks. ICML 2017.", "https://proceedings.mlr.press/v70/dauphin17a.html", "PMLR / volume 70 / dauphin17a"),
        ("S3", "曾永义：近体诗之体制规律。台湾大学开放课程讲义。", "https://ocw.aca.ntu.edu.tw/uploads/course_item_file/file/684/104S110_CT02L01.pdf", "台大开放课程 / 第二讲"),
        ("S4", "王力：诗律。用于声律、诗体及诗韵规格的文本依据。", "https://www.aisixiang.com/data/98039.html", "诗律 / 正文转载"),
        ("S5", "chinese-poetry/chinese-poetry。固定commit及许可证随审计快照保留。", "https://github.com/chinese-poetry/chinese-poetry", "GitHub / chinese-poetry"),
        ("S6", "charlesix59/chinese_word_rhyme。候选韵表，存在已核实的数据缺陷。", "https://github.com/charlesix59/chinese_word_rhyme", "GitHub / chinese_word_rhyme"),
        ("S7", "《平水韵》文本对照：维基文库固定修订oldid=8264804。", "https://zh.wikisource.org/w/index.php?oldid=8264804&title=%E5%B9%B3%E6%B0%B4%E9%9F%BB", "维基文库 / 固定修订"),
    ]
    for key, label, url, display in sources:
        r.p(f"<b>[{key}]</b> {escape(label)}<br/><link href='{escape(url, quote=True)}' color='#087F8C'>{escape(display)}</link>", "small", after=12)
    r.h("文件内的证据位置")
    r.p("docs/PROTOCOL.md保存完整定义、实验协议和Go/Stop条件；docs/STATUS.md区分已做与待做。results保存实际运行报告与事件轨迹；data/audit_sources保存原始快照、SHA和第三方许可证。")
    r.p("本报告附带源项目。工程结果没有被汇总为真实诗歌效果，也没有将候选数据源的许可证当作文本版本或音韵准确性的保证。")
    r.save()
    return r.page


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    n = build(Path(__file__).resolve().parents[1], args.out, args.fonts)
    print(json.dumps({"pages": n, "path": str(args.out), "bytes": args.out.stat().st_size}, ensure_ascii=False))
