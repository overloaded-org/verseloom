"""Render the v0.3 source-restoration report from actual experiment receipts."""
import argparse
from html import escape
import json
from pathlib import Path

from build_report import Report,AMBER,TEAL


def pct(x):return f"{100*x:.2f}%"
def pp(x):return f"{100*x:+.2f}"


def build(root,output,fonts):
    def read(path):return json.loads((root/path).read_text())
    result=read('results/restoration_v03/report.json')
    corpus=read('data/corpus_v03/audit.json')
    readings=read('data/reading_candidates_v03/audit.json')
    tests=read('results/test_receipt.json')
    m=result['metrics']
    cnn=m['cnn_w3_seed_mean'];point=m['aligned_w1_seed_mean'];ngram=m['ngram']
    decision='通过原文复原先导门，仍待文学评价' if result['proxy_promotion_decision']=='PASS_PROXY_ONLY' else '当前卷积未通过先导晋级门'
    output.parent.mkdir(parents=True,exist_ok=True)
    r=Report(output,fonts,'v0.3')

    r.page_start('本轮结果','词卷积学习：原文复原验证','固定真实来源、训练模型、同池比较，以及训练后精确求解的核对')
    r.p('v0.2验证了基本声律文法和有限局部目标的精确组织。v0.3开始学习真实文本中的组合规律：遮蔽一联下句首二字，由训练语料提供候选，比较统计评分与可训练局部评分。')
    r.box('本轮决策：'+decision,'以下准确率表示与冻结来源原文一致。它不能替代搭配、对仗、篇章和意境的独立评价；同样合理的其他措辞也可能被原文复原指标计为错误。',TEAL if result['proxy_promotion_decision']=='PASS_PROXY_ONLY' else AMBER)
    r.table(['证据','实际结果','解释范围'],[
        ['真实文本',f"{corpus['unique_text_poems']:,}首 / {corpus['families']:,}族",'八句等长形状，未逐诗认证格律'],
        ['独立留出',f"{result['test_tasks']:,}题 / {result['test_families']}族",'先分族，再切分数据'],
        ['统计基线',pct(ngram['family_macro']),'全测试题、诗族等权top1'],
        ['宽3卷积',pct(cnn['family_macro']),'三个固定种子平均'],
        ['宽1消融',pct(point['family_macro']),'保留上下句对齐，去掉横向邻位'],
        ['训练后DP',f"{result['dp_audit_cases']}项全部通过",'保持原二字池的目标最优'],
        ['风险测试',f"{tests['tests_run']}项通过",'训练、隔离、候选耦合与工程边界'],
    ],[100,170,r.content-270])
    r.p('代码、数据、模型权重、实际曝光、选择记录和测试输出均在同一项目包中。新增实验采用运行前本地冻结，不是外部预注册。', 'small')

    r.page_start('01 / 来源与切分','先按诗族隔离，再学习组合','固定提交 b8594f81a89752241442f2ce267d6f66f96704ee 的前10个分片。[S1]')
    r.p(f"源数据实际{corpus['raw_records']:,}条，接受{corpus['accepted_source_records']:,}条纯CJK、八句5/7字记录；全文去重后{corpus['unique_text_poems']:,}首。原字形保留，不做隐式繁简或异体合并。")
    r.table(['集合','诗数','诗族','复原题数','目标在训练片段表'],[
        [split,str(corpus['split'][split]['poems']),str(corpus['split'][split]['families']),
         str(corpus['split'][split]['tasks']),str(corpus['split'][split]['target_in_train_candidate_vocabulary'])]
        for split in ('train','dev','test')
    ],[68,72,72,95,r.content-307])
    r.h('分族关系')
    r.p('全文相同、共享完整诗句，或行内二字组集合Jaccard≥0.8的诗连成同族，再按固定哈希阈值分配70/15/15。实际诗族3382个，最大4首；38个全文重复组与213对共享行关系均保留。')
    r.h('题面只给可见信息')
    r.p('每首四联分别出题。模型看到上句、下句去掉首二字后的后缀、句长和遮蔽位置。target、原始完整下句、作者、题名都不进入候选生成器或特征器。二字片段是固定跨度，不代表已经完成古汉语分词。')
    r.box('数据边界','八句等长只是一项形状条件。表面分族不能穷尽异文，同作者可跨集合；固定前10个分片也不是全唐诗随机样本。本轮结果不提供作者留出或全部律诗上的泛化保证。',AMBER)

    r.page_start('02 / 候选','先分清候选缺失与排序失误','测试原文不注入候选池，缺失目标的题仍进入总分。')
    r.p('字符词表只有训练文本出现至少2次的3067个字，另有固定零向量的未知符号。候选片段只来自训练下句首二字、频次至少2：五言1317个、七言97个。')
    r.table(['测试集口径','题目数','题目micro比例','诗族等权比例'],[
        ['原文在训练候选词表',str(result['test_vocab_covered_tasks']),pct(result['vocabulary_coverage']['task_micro']),pct(result['vocabulary_coverage']['family_macro'])],
        ['原文在统计提议top16',str(result['test_pool_covered_tasks']),pct(result['proposal_pool_recall']['task_micro']),pct(result['proposal_pool_recall']['family_macro'])],
        ['所有测试题',str(result['test_tasks']),'100%','100%'],
    ],[164,73,130,r.content-367])
    r.box('一个直接的上界','全题复原准确率 ≤ 每题候选池召回率 ≤ 训练候选词表覆盖率。<br/>同一种加权口径下，这一关系必然成立；排序器无法选中池中不存在的原文。')
    r.h('统计提议器')
    r.p('双向字符1/2/3-gram按0.1/0.3/0.6插值，Lidstone alpha=0.1；加入同列下字给定上字的平滑统计。所有统计只用train。每题取前16个，再按字典序存放，所有系统共享内容身份和同分规则。')
    r.h('比较结论的范围')
    r.p('统计基线同时负责提议，因此本轮研究的是该提议池中的重排。候选上限和提议偏向都保留在报告中，不能把同池结果推广为全词表上的架构优势。')
    r.p('训练允许使用真实训练正例，配3个困难负例及4个随机负例；dev/test不加入答案。训练池与评价池分布不同，这是本轮限制。', 'small')

    r.page_start('03 / 模型与训练','两种神经结构都实际训练','固定特征保留局部可加形式，使训练后的宽3模型继续支持精确DP。')
    runs=result['training_runs']
    r.table(['模型','局部结构','核心参数','可训练总参数'],[
        ['宽3卷积','3字横向窗口；12通道','744',str(next(x for x in runs if x['arm']=='cnn_w3')['trainable_parameters'])],
        ['宽1消融','单列；20通道；上下句仍对齐','760',str(next(x for x in runs if x['arm']=='aligned_w1')['trainable_parameters'])],
        ['统计基线','双向trigram与同列条件统计','训练计数','不作神经参数匹配'],
    ],[100,202,77,r.content-379])
    r.p('两种神经模型共有8维字符embedding，加行号、归一化列号、句长、mask标志4维固定特征。相同种子初始化相同embedding。横向/同列tanh后按位置平均，再线性读出；未知embedding及其优化状态始终为零。')
    r.h('更新数与曝光如实计账')
    r.p('每次运行400次参数更新，batch32，梯度累积1次：实际12800次任务曝光。Adam学习率0.003、解耦衰减0.0001、全局梯度裁剪5。三个种子的两种结构共享候选和抽样schedule。')
    r.table(['种子','每模型曝光','实际不同题','实际不同诗族','两结构共享'],[
        [str(seed),'12800',str(next(x for x in runs if x['seed']==seed)['unique_tasks_exposed']),
         str(next(x for x in runs if x['seed']==seed)['unique_families_exposed']),'是'] for seed in (0,1,2)
    ],[55,107,107,124,r.content-393])
    r.p('每100次更新只用全dev诗族等权top1选检查点，未覆盖题计0，同分留较早版本。六个选择及模型SHA冻结后才计算神经测试评分和正式准确率。此前统计提议器可处理test可见题面以生成固定池。', 'small')

    r.page_start('04 / 实测结果','用外部原文检查内部评分','主指标为全题诗族等权top1；三个种子先平均，不当成独立题目。')
    r.table(['评分器','诗族等权top1','题目micro top1','池覆盖题top1'],[
        ['统计基线',pct(ngram['family_macro']),pct(ngram['task_micro']),pct(ngram['covered_task_top1'])],
        ['宽3卷积：种子均值',pct(cnn['family_macro']),pct(cnn['task_micro']),pct(cnn['covered_task_top1'])],
        ['宽1消融：种子均值',pct(point['family_macro']),pct(point['task_micro']),pct(point['covered_task_top1'])],
    ],[177,112,112,r.content-401])
    r.table(['种子','宽3 top1','宽1 top1','宽3选中更新','宽1选中更新'],[
        [str(seed),pct(m[f'cnn_w3_seed{seed}']['family_macro']),pct(m[f'aligned_w1_seed{seed}']['family_macro']),
         str(next(x for x in runs if x['seed']==seed and x['arm']=='cnn_w3')['selected_update']),
         str(next(x for x in runs if x['seed']==seed and x['arm']=='aligned_w1')['selected_update'])]
        for seed in (0,1,2)
    ],[48,109,109,117,r.content-383])
    r.table(['配对比较：宽3减比较对象','平均差 / 百分点','95%区间 / 百分点'],[
        [label,pp(result['paired_comparisons'][key]['mean_difference']),
         '['+', '.join(pp(x) for x in result['paired_comparisons'][key]['interval'])+']']
        for key,label in [('aligned_w1_seed_mean','宽1消融'),('ngram','统计基线')]
    ],[211,136,r.content-347])
    r.box(decision,'预先设定的先导门要求：相对两个比较对象都提高至少2个百分点，且各自配对区间下界大于0。该判断只对应原文复原，正式文学偏好门仍为NOT_EVALUATED。',AMBER)
    r.p('bootstrap以诗族为单位，族内先平均，三种子先等权汇总，共5000次。区间以这三个固定种子和当前提议机制为条件；分句长结果、全部逐题预测与训练曲线见项目包。', 'small')

    r.page_start('05 / 组织算法','保留词片段，才能保持同一搜索问题','训练后的局部目标与有限二字词表已接入精确动态规划。')
    r.p('若候选是AB与CD，直接把两列分别设为{A,C}和{B,D}，会额外产生AD、CB。它们不在原候选池中；即使字数正确，也改变了被比较的问题。')
    r.box('把片段成员约束放进局部因子','首因子 f[0] 检查下句第0、1列拼出的片段是否属于冻结候选池。若不属于，分数为负无穷；其余因子保留训练后的卷积评分。<br/>该约束仍只涉及相邻两列，原两列状态DP继续成立。')
    r.h('实际核对')
    r.p(f"事先指定每种句长按task hash排序前24题，对三个选定宽3模型分别核对，共{result['dp_audit_cases']}项。全部结果仍在原池中，均达到对原16候选完整评分的最优；最大数值偏差{result['dp_max_abs_error']:.3e}，低于1e-10阈值。")
    r.table(['输入','机制','保证'],[
        ['固定上句、下句后缀','题面允许列表','没有隐藏目标或远处输出依赖'],
        ['训练后的局部权重','冻结快照+局部因子','学习后仍满足可加分解'],
        ['完整二字候选池','首因子成员约束','不生成池外交叉片段'],
        ['有限条件最优','原始批量评分器枚举核对','不等于文学质量最优'],
    ],[143,162,r.content-305])
    r.p('本轮的span_solver接口固定处理下句首二字。任意长词、跨联语义或全篇禁重字需要扩展约束状态；不能把这个桥接推广为已经支持任意词跨度的全篇生成器。')

    r.page_start('06 / 读音参考','真实片段接到音韵层，仍保留疑点','此关联审计没有改变本轮检索池、训练特征或测试标签。')
    counts=readings['counts_by_length']
    r.table(['片段类别','五言候选','七言候选'],[
        [label,str(counts['5'][key]),str(counts['7'][key])]
        for key,label in [('spans','全部训练片段'),('both_characters_found','两字都能查到'),
            ('both_single_membership','两字各仅一个字—部候选'),
            ('at_least_one_cross_tone_character','至少一字跨平仄'),
            ('at_least_one_unknown_character','至少一字未知'),
            ('at_least_one_editorial_issue','关联已有校勘议题')]
    ],[282,109,r.content-391])
    r.p('类别可以重叠。每字保留主参考表所有韵部成员、平仄、来源行号和问题标识；所有片段及字符都为requires_context_review=True。单一候选也不自动证明语境读音。')
    r.h('缺字与字形问题单列')
    r.p('例如“啼”未在主索引出现，但仍关联既有缺字议题；“先”的相关疑点也保留。对“爲”“遊”等未命中字形，不自动归并到其他字。unknown表示该录文未收，不表示历史语言中不存在。')
    r.box('格律认证仍需自己的证据','本轮有真实来源文本与原文目标，也有韵部候选关联；它们还不足以成为逐字消歧的格律金标。后续真实律诗求解必须先完成语境读音审核，不能靠目标模板选读音。',AMBER)
    r.p('候选来源、全部成员与独立重算计数见data/reading_candidates_v03/。完整106部参考及其未决校勘仍见data/reference_v02/。', 'small')

    diagnostic=read('results/dev_candidate_diagnostic_v03/report.json')
    r.page_start('07 / 开发集诊断','词片段的来源位置限制了覆盖','正式先导之后的探索性诊断；只用train构词表，只查dev目标。')
    labels={'even_prefix_min2':'下句首二字，频次≥2','even_prefix_min1':'下句首二字，频次≥1',
            'even_adjacent_min2':'下句所有相邻二字，频次≥2','all_adjacent_min2':'全部句内相邻二字，频次≥2'}
    r.table(['候选来源策略','候选总数','覆盖dev题','题目覆盖率'],[
        [labels[x['policy']['id']],str(x['candidate_count_total_length_tagged']),
         str(x['overall']['covered_dev_tasks']),pct(x['overall']['task_weighted_vocabulary_coverage'])]
        for x in diagnostic['results']
    ],[228,86,93,r.content-407])
    r.p('总数按五言/七言分别计数，跨句长同文片段可能重复。四种策略全部公开；分句长和诗族等权口径见诊断报告。没有打开测试任务、模型或预测文件，也没有回改本轮候选和结果。')
    r.box('下一轮候选来源：训练全句内相邻片段','把片段从“固定下句开头”扩展到训练文本所有句内位置，同时保留频次至少2，dev词表覆盖从19.47%升至41.00%。候选数约为原来的8.40倍，需要另查检索召回和预算。此数字是词表覆盖上限，不是top16召回、准确率或测试集改善。')
    r.h('还有两个边界')
    r.p('即使采用这一候选来源，七言dev词表覆盖仍只有19.84%。更宽词表也不保证现有统计提议器能把正确片段排进前16；下一轮应在dev确定检索方案，再用新独立留出确认。')
    r.h('完整性事件保留')
    r.p('首次评测的SHA校验发现一个已选检查点文件与冻结记录不符并中止。按原种子、候选和schedule重放，恢复文件与原预期SHA逐字节一致；六个选择不变后完成评测。正式训练2400次更新/76800次曝光之外，恢复额外200次更新/6400次曝光单列。异常原件保留，根因未确定。详见INTEGRITY_v0.3.md。', 'small')

    r.page_start('08 / 决策与复核','下一轮先解决数据覆盖和目标','本轮测试已暴露；后续改模型或候选规则须使用新的独立确认集。')
    r.table(['项目层次','本轮状态','后续验收'],[
        ['基本声律与有限DP','v0.2证明与核对保留','变格、读音、全篇依赖扩展'],
        ['真实文本评分学习','已完成6次训练及一次正式留出评价','按结果决定保留或调整评分'],
        ['候选供给','dev已比较4种训练来源','训练全句片段优先；另查检索预算'],
        ['二字片段约束','训练后144项DP核对','更长跨度须显式增加状态'],
        ['文学质量','未进行人工评价','独立盲评搭配、对仗、篇章和意境'],
    ],[112,210,r.content-322])
    r.h('复现入口')
    r.p('python scripts/run_tests.py<br/>python scripts/build_corpus_v03.py --check<br/>python scripts/audit_candidate_readings_v03.py --check<br/>PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/run_restoration_v03.py --out results/restoration_replay', 'small')
    r.h('已有研究与本项目位置')
    r.p('文本局部卷积已有先例，经典文本CNN使用max-over-time pooling。[S2] 中文诗生成研究也已学习字符组合与跨行上下文，并采用人工评价；内容规划另有专门模型。[S3][S4] 本项目保留可加评分以便精确求解，不作卷积或诗歌生成的首创声明。', 'small')
    r.p("[S1] <link href='https://github.com/chinese-poetry/chinese-poetry'>chinese-poetry/chinese-poetry</link>；固定提交与各源SHA见data/corpus_v03/source_manifest.json。<br/>[S2] Kim, 2014. <link href='https://aclanthology.org/D14-1181/'>Convolutional Neural Networks for Sentence Classification</link>.<br/>[S3] Zhang &amp; Lapata, 2014. <link href='https://aclanthology.org/D14-1074/'>Chinese Poetry Generation with Recurrent Neural Networks</link>.<br/>[S4] Wang et al., 2016. <link href='https://aclanthology.org/C16-1100/'>Chinese Poetry Generation with Planning based Neural Network</link>.", 'small')
    r.save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fonts',type=Path,required=True)
    args=p.parse_args()
    build(Path(__file__).resolve().parents[1],args.output,args.fonts)
