"""Build the v0.4 retrieval report exclusively from saved experiment evidence."""
import argparse
import json
from pathlib import Path

from build_report import Report, AMBER, TEAL


def pct(value):
    return f'{100*value:.2f}%'


def pp(value):
    return f'{100*value:+.2f}'


def build(root, output, fonts):
    def read(name):
        return json.loads((root/name).read_text())
    formal = read('results/retrieval_v04/report.json')
    dev = read('results/retrieval_v04/dev_report.json')
    corpus = read('data/confirmation_v04/audit.json')
    diagnostic = read('results/retrieval_v04_diagnostic/report.json')
    tests = read('results/test_receipt.json')
    review = read('results/retrieval_v04/independent_review.json')
    if review['status'] != 'PASS_FROZEN_RETRIEVAL_RESULTS':
        raise ValueError('independent result review did not pass')
    metrics = formal['metrics']
    comparison = formal['paired_comparisons']['selected_minus_narrow']
    selected = formal['selected_method']
    d = diagnostic['overall']
    methods = [('narrow_ngram', '原窄池统计'), ('wide_ngram', '宽池原统计'), ('selected', '开发选中关联方案')]
    output.parent.mkdir(parents=True, exist_ok=True)
    r = Report(output, fonts, 'v0.4')

    r.page_start('本轮结论', '覆盖扩大，有限候选召回仍未过门',
                 '一次冻结开发选择，一批新的诗族隔离确认文本，全部负结果保留')
    r.p('v0.3发现候选词表限制了复原上限。v0.4把候选扩大到训练文本全部句内相邻二字，并检验上下对应片段与局部四字搭配统计能否改善前16候选。')
    r.box('本轮决策：不晋级这版检索器',
          f"主指标从{pct(metrics['narrow_ngram']['overall']['recall_at_16']['family_macro'])}到{pct(metrics['selected']['overall']['recall_at_16']['family_macro'])}；差{pp(comparison['mean_difference'])}个百分点，95%区间[{pp(comparison['interval'][0])}, {pp(comparison['interval'][1])}]。未达到点估计+2个百分点且区间下界大于0的预定门。", AMBER)
    r.table(['检索方案', '诗族等权R@16', '题目micro R@16', '找回题数'], [
        [label, pct(metrics[key]['overall']['recall_at_16']['family_macro']),
         pct(metrics[key]['overall']['recall_at_16']['task_micro']), str(d[key]['hits_at_16'])]
        for key, label in methods], [159,126,126,r.content-411])
    r.p('R@16表示原文是否出现在16个输出候选中，不是top1正确率。所有2096题保留，词表外与池外目标均计0；主指标族内平均后对512个诗族等权。')
    r.table(['工程证据', '实际完成', '解释范围'], [
        ['新确认集', '524首 / 512族 / 2096题', '按指定表面关系隔离'],
        ['开发比较', '16个宽池方案 + 原窄池基线', '只用旧dev选择'],
        ['快速基线', '固定16开发题整池精确一致', '分数和完整排名均核对'],
        ['风险测试', f"{tests['tests_run']}项通过", '来源、隔离、数值与冻结守卫'],
        ['新神经训练', '0次更新', '不改v0.3卷积结论'],
    ], [113,211,r.content-324])
    r.p('原声律矩阵、精确局部求解与训练后144项DP核验保留；文学质量和逐字语境读音金标仍未建立。', 'small')

    r.page_start('01 / 新确认数据', '先排除旧诗族，再固定整族抽取',
                 '同一源提交的新10个分片；旧train、dev、test全部进入排除集')
    source = corpus['source_record_audit']
    r.table(['步骤', '记录 / 诗 / 族', '规则'], [
        ['固定来源', f"{source['raw_records']:,}条", '唐诗分片10000至19000，步长1000'],
        ['形态通过', f"{source['accepted_source_records']:,}条", '纯CJK，8×5或8×7'],
        ['全文去重', f"{corpus['new_unique_text_poems']:,}首", '原字形，不做隐式繁简合并'],
        ['旧诗族关联排除', f"{corpus['new_poems_excluded_by_prior_component']}首", '含38首旧精确文本，保留传递关系'],
        ['剩余可选', f"{corpus['eligible_poems']:,}首 / {corpus['eligible_families']:,}族", '分量中不得含任何旧诗'],
        ['固定哈希前512族', '524首 / 2096题', '整族保留，每首四题'],
    ], [118,150,r.content-268])
    r.h('分族与独立性的具体含义')
    r.p('旧3579首与新文本联合建图。共享完整诗句，或行内二字组集合Jaccard≥0.8即连边；整条传递分量中只要有旧诗，新诗即排除。新诗间的间接桥接也不能绕开排除。')
    r.table(['句长', '确认诗数', '确认题数', '确认诗族'], [
        [str(length), str(corpus['selected_poems_by_length'][str(length)]),
         str(corpus['selected_tasks_by_length'][str(length)]),
         str(metrics['selected'][str(length)]['families'])] for length in (5,7)
    ], [97,122,122,r.content-341])
    r.box('独立性有明确范围', '表面关系可能漏掉异文与异体字；24位选中作者也见于旧集合。固定连续分片不是全体诗歌的随机样本。八句形态不是严格律诗认证，原文一致不是文学质量金标。', AMBER)
    r.p('来源提交 b8594f81a89752241442f2ce267d6f66f96704ee；政策与代码在下载前本地冻结。每个源文件、选取记录、排除边和SHA均在data/confirmation_v04/。', 'small')

    r.page_start('02 / 检索机制', '把片段来源与局部关联分开检验',
                 '候选与全部统计只由旧train的2502首建立；题面接口不接受隐藏原文')
    r.table(['候选来源', '五言', '七言', '训练位置'], [
        ['原窄池N', '1,317', '97', '偶数句开头二字，频次≥2'],
        ['扩展宽池W', '10,375', '1,504', '全部句内相邻二字，频次≥2'],
    ], [97,88,88,r.content-273])
    r.p('宽池包含窄池。两者的词面和频次分别与v0.3原候选资源、独立来源诊断逐项相同。候选字符均在原训练字符表中；未知可见题面字使用显式UNK。二字片段不是经过语言学分词的词。')
    r.h('评分形式')
    r.box('S(w | query) + a·A(w | 上句首二字) + b·B(w | 后缀首二字)',
          'S保持原双向字符1/2/3-gram与上下同列字统计。A来自训练各联所有对齐相邻二字；B来自训练句内四字窗口的前后二字关系。两项均按句长独立计数，只统计宽池内的目标片段。')
    r.p('宽池先验 p(w)=(片段频次+0.1)/(宽池频次总和+0.1×词表大小)。每项关联加分为 log1p(共同出现计数/(5×p(w)))。同一查询下对所有候选相同的条件归一化项被省略；该加分不是校准的正确概率。')
    r.h('输入边界')
    r.p('查询仅含上句、下句可见后缀、句长和固定遮蔽位置。先计算全部候选分数，再用独立保存的target计名次。作者、题名、完整下句不入模；开发与确认目标不加入候选。')
    r.box('能力边界', 'A是上下对应片段的统计关联，不能直接称为已学会对仗；B是局部搭配关联，不能直接称为诗意。此实验使用局部统计，不改变原卷积学习试验的未晋级结论。', AMBER)

    r.page_start('03 / 开发选择', '公开全部网格，按预定同分规则选择',
                 '主选择指标为旧dev的诗族等权R@16；使用精确有理数比较')
    values = [0., .25, .5, 1.]
    table = []
    for i, a in enumerate(values):
        table.append([str(a)] + [pct(dev['metrics'][f'wide_a{i}_b{j}']['overall']['recall_at_16']['family_macro']) for j in range(4)])
    r.table(['A权重 / B权重', '0', '0.25', '0.5', '1'], table,
            [138,91,91,91,r.content-411])
    best = max(item['family_recall'] for item in dev['selection_table'])
    ties = sum(item['family_recall'] == best for item in dev['selection_table'])
    r.box(f"选中 A={selected['alignment_weight']}，B={selected['boundary_weight']}",
          f"开发最佳主指标有{ties}个网格点并列。按A外层、B内层从小到大的固定顺序，选择第一个。这个结果没有识别出唯一最优的数值权重。")
    r.table(['开发集对照', '诗族R@16', '题目R@16', '题目词表覆盖'], [
        [label, pct(dev['metrics'][key]['overall']['recall_at_16']['family_macro']),
         pct(dev['metrics'][key]['overall']['recall_at_16']['task_micro']),
         pct(dev['metrics'][key]['overall']['vocabulary_coverage']['task_micro'])]
        for key,label in [('narrow_ngram','原窄池'), ('wide_a0_b0','宽池原统计'), (selected['id'],'选中关联方案')]
    ], [145,114,114,r.content-373])
    r.h('选择完成后才做确认')
    r.p('保存代码、输入、候选与关联计数、全部开发逐题结果和选中方案的身份。确认前从开发结果重新计算选择，要求与选择文件、开发报告三者一致；任何冻结文件改变即中止。')
    r.p('旧test在检索运行器中于JSON解码前跳过。新确认集在开发阶段只校验字节身份，未解析或评分；正式确认评分在selection_frozen之后开始。', 'small')

    r.page_start('04 / 独立确认', '有限输出池中的增益仍然很小',
                 '唯一确认性比较是selected减原窄池的诗族等权R@16')
    r.table(['确认方案', 'R@8', 'R@16', 'R@32', 'R@64'], [
        [label] + [pct(metrics[key]['overall'][f'recall_at_{k}']['family_macro']) for k in (8,16,32,64)]
        for key,label in methods], [151,87,87,87,r.content-412])
    r.p('上表均为诗族等权；R@8/32/64是预先列出的描述性指标，不能替代R@16的主门槛。增大K也改变后续输出预算。', 'small')
    r.table(['配对比较', '均值差 / 百分点', '95%区间 / 百分点'], [
        [label, pp(formal['paired_comparisons'][key]['mean_difference']),
         '['+', '.join(pp(x) for x in formal['paired_comparisons'][key]['interval'])+']']
        for key,label in [('selected_minus_narrow','选中 - 原窄池：主比较'),
                          ('wide_minus_narrow','宽池 - 原窄池：次要'),
                          ('selected_minus_wide','选中 - 宽池：次要')]
    ], [225,126,r.content-351])
    r.p('512个诗族成对重采样5000次，固定种子和线性百分位区间。主区间包含0，且点估计小于+2个百分点；不声称已证明正增益或统计劣势。')
    r.table(['描述性检查', '原窄池', '宽池原统计', '选中关联方案'], [
        ['五言R@16']+[pct(metrics[key]['5']['recall_at_16']['family_macro']) for key,_ in methods],
        ['七言R@16']+[pct(metrics[key]['7']['recall_at_16']['family_macro']) for key,_ in methods],
        ['全题top1']+[pct(metrics[key]['overall']['top1']['family_macro']) for key,_ in methods],
        ['全词表MRR']+[f"{metrics[key]['overall']['mrr_full_lexicon']['family_macro']:.4f}" for key,_ in methods],
    ], [145,112,112,r.content-369])
    r.box('同一召回率不代表相同预测', '选中关联方案相对宽池原统计新增命中3题、损失3题，主指标恰好相同。top1、MRR和切片变化不能用来推翻预定主判断。', AMBER)

    r.page_start('05 / 失败分解', '新增覆盖被有限候选竞争抵消',
                 '以下是确认后的描述性逐题计数，不改动任何正式评分或门槛')
    r.table(['确认题目去向', '原窄池', '扩展宽池 / 选中'], [
        ['原文在词表', str(d['narrow_ngram']['vocabulary_covered']), str(d['selected']['vocabulary_covered'])],
        ['原文在前16', str(d['narrow_ngram']['hits_at_16']), str(d['selected']['hits_at_16'])],
        ['在词表但未进前16', str(d['narrow_ngram']['covered_but_below_top16']), str(d['selected']['covered_but_below_top16'])],
        ['原文在词表外', str(d['narrow_ngram']['vocabulary_absent']), str(d['selected']['vocabulary_absent'])],
    ], [252,126,r.content-378])
    r.p(f"题目层面的词表覆盖从{pct(d['narrow_ngram']['task_micro_vocabulary_coverage'])}到{pct(d['selected']['task_micro_vocabulary_coverage'])}。新增423题原文可供选择，但选中方案仍有658题属于‘词表内、前16外’，另有1369题词表外。")
    transitions = d['paired_task_transitions']
    r.table(['相对原窄池', '保留旧命中', '新增命中', '损失旧命中', '净增'], [
        [label, str(transitions[key]['retained_hit']), str(transitions[key]['gain']),
         str(transitions[key]['loss']), str(transitions[key]['gain']-transitions[key]['loss'])]
        for key,label in [('wide_ngram_minus_narrow_ngram','宽池原统计'),
                          ('selected_minus_narrow_ngram','选中关联方案')]
    ], [147,95,95,95,r.content-432])
    r.box('扩大候选域不会自动改善已有候选的名次',
          '当评分和同分规则相同时，W包含N，则旧候选w满足 rank_W(w)≥rank_N(w)。新竞争者可能把原先命中的片段挤出固定前16。此关系已在确认集中304个原词表内目标上核对。')
    r.p('宽池原统计新增的20个命中都来自原词表外，同时损失14个旧命中，净增6。选中关联方案相对原窄池增加23、损失17，也净增6。这里是题目计数，不能直接当作诗族等权百分点。')
    r.h('下一机制问题')
    r.p('需要检验如何分配原片段与新增片段的有限候选位置，同时提高词表内排序能力。扩大词表、局部统计、精确求解分别负责不同环节，任何一项都不能替代语义和文学评价。')

    r.page_start('06 / 工程与成本', '给定可加目标的精确求解继续保留',
                 '新实验增加检索证据；没有把新的关联分数冒充已验证的整诗生成目标')
    r.table(['层次', '当前证据', '尚未证明'], [
        ['声律矩阵', 'v0.2基本五/七言模板与有限可行域', '语境读音与所有变格认证'],
        ['局部组织', '既有可加因子动态规划；训练后144项核验', '全篇语义或文学最优'],
        ['二字片段检索', '本轮新确认完成；未通过晋级门', '稳定超过原窄池'],
        ['词卷积学习', '保留v0.3六次训练与未晋级判断', '文学质量优势'],
    ], [105,240,r.content-345])
    r.h('数值兼容性和独立复核')
    r.p('快版保持原标量概率、math.log与当前运行时求和语义，不重定义同分。固定16道开发题全窄池分数与完整排名精确一致。独立审查再用原标量评分和独立Counter，对16道确认题、95032个宽候选、48套排名核对，全部一致。')
    r.h('固定K不是固定计算预算')
    account = formal['compute_accounting']
    r.p(f"确认阶段共享计算{account['shared_component_query_evaluations']:,}个查询、{account['ngram_candidate_items_in_shared_wide_evaluation']:,}个宽池候选分数，再做三套排序。原窄池逻辑排序量{account['ranked_items_by_method']['narrow_ngram']:,}项，每个宽池方案{account['ranked_items_by_method']['wide_ngram']:,}项。")
    r.p(f"开发17方法共享组件阶段约{dev['evaluation_seconds_all_17_methods_shared_components']:.2f}秒；确认3方法约{formal['evaluation_seconds_three_methods_shared_components']:.2f}秒。这不是各方法独立线上延迟。另有参考实现审计和数据构建工作；新神经训练为0次更新。", 'small')
    r.box('冻结记录与历史保留', '源文件、训练资源、完整关联计数、选择记录和逐题输出都可追溯。v0.3检查点异常、原件及精确恢复说明完整保留；本轮没有更改其结果。', TEAL)

    r.page_start('07 / 后续与复现', '固定配额也未超过已有开发方案',
                 '保留开发负结果，后续转向固定候选集中的上下文匹配学习')
    r.p('事后开发诊断固定比较10种配额方案：旧片段位置q取0、4、8、12、16；旧片段仍按原统计排序，新片段分别按原统计或本轮已选关联分数排序。两来源互斥，合计仍为16，不注入原文。')
    quota_path = root/'results/dev_quota_v04/report.json'
    if not quota_path.exists():
        raise ValueError('dev quota diagnostic must be complete before final report')
    quota = read('results/dev_quota_v04/report.json')
    best_quota = quota['descriptive_dev_best']
    quota_rows = []
    for old_quota in (0,4,8,12,16):
        cells = [str(old_quota)+' / '+str(16-old_quota)]
        for ranker in ('ngram','a025_b025'):
            entry = next(x for x in quota['results'] if x['method']['old_quota']==old_quota and x['method']['new_ranker']['id']==ranker)
            cells.append(pct(entry['overall']['family_macro_recall_at_16']))
        quota_rows.append(cells)
    r.table(['旧 / 新位置数','新增片段：原统计','新增片段：关联评分'], quota_rows,
            [145,179,r.content-324])
    r.box('配额结果只属于开发探索', '该诊断使用已多次分析的旧dev，在正式确认之后提出。全部10种方案及逐题池单独保存；q=16须逐题复现原窄池命中。即使开发指标提高，也不能报告为新的确认改善。', AMBER)
    r.h('本轮决策')
    r.p(f"配额网格内最佳为旧12/新4，dev诗族R@16={pct(best_quota['overall']['family_macro_recall_at_16'])}，仍低于正式开发选中宽池的{pct(dev['metrics'][selected['id']]['overall']['recall_at_16']['family_macro'])}。当前不优先推进它的独立确认；保留原基线与全部实验结果。")
    r.p('下一轮问题改为：固定宽词表与K=16，学习可见上下文与候选二字片段的匹配矩阵。先在训练/开发诗族上检查排序信号，再冻结新的独立确认；尚未声称这一方法有效。', 'small')
    r.h('复现入口')
    r.p('python scripts/run_tests.py<br/>python scripts/build_confirmation_v04.py --check<br/>PYTHONPATH=src python scripts/run_retrieval_v04.py dev --out results/retrieval_v04_replay<br/>PYTHONPATH=src python scripts/run_retrieval_v04.py confirm --out results/retrieval_v04_replay<br/>python scripts/analyze_retrieval_v04.py --check', 'small')
    r.h('证据索引')
    r.p('正式结果：results/retrieval_v04/；独立审查：independent_review.json；失败分解：results/retrieval_v04_diagnostic/；配额诊断：results/dev_quota_v04/；协议：docs/RETRIEVAL_v0.4_PROTOCOL.md。历史v0.1至v0.3报告与代码证据保留在项目包中。', 'small')
    r.p("来源：<link href='https://github.com/chinese-poetry/chinese-poetry'>chinese-poetry/chinese-poetry</link>；固定提交、SHA及许可随数据保留。原文复原、检索召回、格律合法与文学优劣是不同评价对象。", 'small')
    r.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fonts', type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.output, args.fonts)
