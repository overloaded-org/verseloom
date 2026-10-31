#!/usr/bin/env python3
"""Advisory projective chart for five/seven-character lines; pure stdlib. No labels in inference."""
import argparse, json, math, re
from pathlib import Path
from functools import lru_cache
ROOT=Path(__file__).resolve().parent
LABEL={'R':'可解','U':'存疑','B':'疑斷裂'}
# tag: word inventory (disambiguation is deliberately conservative); no line-specific rules.
GROUPS={
'N':'人 君 客 子 我 蘭 蕙 草 木 城 國 家 鄉 江 山 海 道 路 口 衣 身 心 意 情 恩 恨 信 書 雁 鴈 馬 酒 花 樹 門 庭 鐘 鼓 樂 杯 錢 兵 事 時 世 年 秋 春 夏 冬 手 頭 眼 命 名 夢 淚 君子 故人 家人 少年 青衫 人間',
'P':'我 吾 予 余 汝 爾 君 誰 人 其 彼 此 之 何人 何處 誰人',
'T':'今 昔 昨 旦 暮 曉 夜 晚 晨 夕 朝 秋 春 今日 今夜 明日 日夕 昨日 昔日 當年 去年 往日 年年 夜夜',
'L':'上 下 中 間 內 外 邊 前 後 東 西 南 北 裏 裡 畔 側 下方',
'A':'好 美 寒 冷 清 空 白 青 赤 黑 紅 黃 玄 遠 近 長 短 高 低 老 少 大 小 深 淺 窮 盡 殘 多 稀 久 遲 閑 閒 靜 碧 蒼 孤 獨 苦 悲 無窮',
'I':'來 去 歸 飛 笑 啼 哭 鳴 起 坐 立 行 走 生 死 老 落 盡 流 動 睡 眠 醒 醉 休 留 至 返 還 遊 聊 照 清 住 立 回 歇 謝 合 開 散 白 忘 開口',
'V':'見 看 望 問 訪 尋 覓 迎 送 別 待 逢 遇 憐 憶 念 思 懷 知 聞 聽 說 道 題 書 寄 作 取 持 把 執 引 視 觀 脫 著 飲 酌 乘 登 入 出 過 歸 殺 酬 買 賣 得 望 掃 採 拾 換 夾 抱 照 滿 含 辭 對 掩 傳 贈 開 留 戀 借 有 無 是 如 似 成 為 爲 愛 喜 煩 憂 驚 愁 悵 送別 懷念',
'C':'於 于 從 自 向 往 對 就 與 以 為 爲 隨 緣 因 朝 由 依 臨 當 到 至',
'M':'能 可 欲 肯 須 應 敢 願 將 當 莫 休 得 宜 合 擬 堪 能夠',
'D':'不 無 未 莫 勿 非 尚 且 方 正 初 纔 又 亦 更 皆 總 獨 空 徒 枉 遙 長 常 曾 猶 已 今 暫 復 永 相 自 共 同 卻 盡 誠 始 早 頻 只 還 但 宜 難 易 忽 唯 惟 好 可惜 可憐 如何 何曾 安得 不復',
'J':'而 若 雖 與 或 既 乃 則 及 如 若使 縱使',
'K':'一 二 三 四 五 六 七 八 九 十 百 千 萬 半 兩 數 多少 幾 千萬',
'F':'之 者 所 也 矣 焉 乎 哉 耳',
'G':'片 盞 杯 匹 枝 葉 寸 尺 重 道 行 畝 家 層 聲 點 里 歲 年 人 袂 字'
}
# Headed, local constituent patterns (not a gold dependency tree or POS inventory).
# lhs,left,right,head-side,dependent relation,penalty
RULES=[('NP','A','NP',1,'修飾',.13),('NP','NP','NP',1,'名詞修飾或並置',.65),('NP','NP','L',0,'方位',.12),('QP','K','G',1,'數量',.03),('NP','QP','NP',1,'數量修飾',.08),('NP','K','NP',1,'數量修飾',.12),('PP','C','NP',0,'介引對象',.03),('VP','V','NP',0,'賓語或所及對象',.03),('VP','V','S',0,'從句補語',.48),('VP','D','VP',1,'狀語',.07),('VP','M','VP',1,'情態',.07),('VP','PP','VP',1,'狀語',.15),('VP','VP','PP',0,'介詞補語',.20),('S','NP','VP',1,'主題或主語',.12),('S','T','S',1,'時間',.10),('S','T','VP',1,'時間',.10),('VP','VP','VP',1,'連動候選',.85),('S','S','S',1,'分句並置',1.1),('S','S','F',0,'語氣',.20),('VP','J','S',1,'連接',.45)]
LIFT={'N':['NP'],'P':['NP'],'T':['NP'],'I':['VP','S'],'A':['VP','S'],'VP':['S'],'L':['NP'],'G':['NP']}
DIRECTION=set('於于向往從由與對以隨緣依'); MODAL=set('須應欲宜擬肯敢能堪將'); CONNECT=set('而及或雖若'); GENERIC=object()

def han(c):return '\u3400'<=c<='\u9fff' or '\U00020000'<=c<='\U000323af' or '\uf900'<=c<='\ufaff'
def validate(line):
    if not isinstance(line,str) or len(line) not in (5,7) or not all(map(han,line)):raise ValueError('只接受恰五／七個漢字的正文，不轉換字形。')
@lru_cache(maxsize=1)
def model():
    j=json.loads((ROOT/'data'/'lexicon.json').read_text());d={w:set(v) for w,v in j['typed'].items()}
    for t,group in GROUPS.items():
        for w in group.split():d.setdefault(w,set()).add(t)
    return d,j['attested']

def parse(line):
    d,att=model();n=len(line); chart={}
    def add(cell,tag,cost,head,tok,edges,guesses):
        old=cell.get(tag);v=(cost,head,tok,edges,guesses)
        if old is None or cost<old[0]:cell[tag]=v
    for width in range(1,n+1):
        for i in range(n-width+1):
            j=i+width;w=line[i:j];cell=chart[i,j]={}
            tags=d.get(w,set()); inferred=not tags and width<=3 and att.get(w,0)>=3
            if tags or inferred:
                for tag in sorted(tags or {'N'}):
                    cost=(.32 if not inferred else 1.9)+(.08 if len(tags)>3 else 0)
                    add(cell,tag,cost,i,[(i,j,w,tag)],[],int(inferred))
            for k in range(i+1,j):
                for lhs,l,r,h,dep,penalty in RULES:
                    a=chart[i,k].get(l);b=chart[k,j].get(r)
                    if a is None or b is None:continue
                    head=b[1] if h else a[1];child=a[1] if h else b[1]
                    add(cell,lhs,a[0]+b[0]+penalty,head,a[2]+b[2],a[3]+b[3]+[(head,child,dep)],a[4]+b[4])
            for cat in list(cell):
                a=cell[cat]
                for dst in LIFT.get(cat,[]):add(cell,dst,a[0]+.02,a[1],a[2],a[3],a[4])
    best=chart[0,n].get('S'); nominal=chart[0,n].get('NP')
    if not best and nominal:best=nominal
    return best,bool(chart[0,n].get('S')),chart

def diagnose(line,mate=None,side=None,segments=None):
    validate(line)
    if mate is not None:
        validate(mate)
        if len(mate)!=len(line) or side not in ('upper','lower'):raise ValueError('對句長度、位置無效')
    if segments is not None and (not isinstance(segments,(list,tuple)) or not all(isinstance(x,str) and x for x in segments) or ''.join(segments)!=line):raise ValueError('分段須逐字拼回原句')
    best,sent,chart=parse(line);n=len(line);d,att=model();tail=line[-1];issues=[]
    def note(kind,i,j,explanation,weight):issues.append(dict(type=kind,span=[i,j],text=line[i:j],explanation=explanation,severity=weight))
    # Multiword readings stop a generic tail warning. Require a positive HEAD grouping, not merely collocation.
    lexical_end=any(line[-k:] in d and d[line[-k:]] & {'N','I','A','P','T'} for k in (2,3))
    if not lexical_end:
        if tail in DIRECTION|MODAL|CONNECT or tail in {'所'}:
            ty='指向對象缺失' if tail in DIRECTION else '謂語待補' if tail in MODAL else '連接成分缺失'
            # A directional ending can be lexical or elliptical. Suppress if the mate begins with an overt deictic/location phrase.
            supply=mate is not None and side=='upper' and tail in DIRECTION and any(mate.startswith(w) for w in ('此地','此間','彼處','其間','其上','其下'))
            if supply:note('省略可補',n-1,n,'下句開頭有指處成分，記為待確認的跨句補足，不直接報錯。','possible')
            elif tail in {'往','從','對','隨','依'} and chart[0,n].get('S') and best[4]==0:note(ty,n-1,n,'另有完整詞類讀法；尾字疑點不足以斷錯。','possible')
            else:note(ty,n-1,n,'單字介引／情態／連接讀法缺右側成分；其他古義仍待人工核對。','alarm')
        elif tail in set('贈授賜賦使令命遣持執把') and not lexical_end:
            # A noun on the other line alone cannot prove a dependency or human intent.
            if mate and any(mate.startswith(w) for w in ('此物','此書','此詩','其人','此身')):
                note('省略可補',n-1,n,'對句有顯式回指，可提出省賓解，不判斷裂。','possible')
            else:note('動詞缺賓',n-1,n,'按常見及物／使令讀法，目標或受事未落實；不排除名詞讀法或跨句省略。','possible')
    # Internal valency diagnostics: only raise a local alarm if the obligatory marker
    # faces another explicit operator and neither a known word nor a nominal can absorb it.
    for i in range(1,n-1):
        a,b=line[i:i+2]; w=a+b
        if a in set('於于把將以向') and b in set('未不莫勿毋且') and w not in d:
            if a not in set('將以') and not (chart[i+1,n].get('NP') or chart[0,n].get('S')):
                note('指向對象缺失',i,i+2,'介引成分後緊接否定／連接詞，未找到可信的名詞補足；標記歧義以供複核。','alarm')
        if a in set('盞匹畝尺艘') and b in set('與而及或') and not chart[0,n].get('S') and not w in d:
            note('斷詞可疑',i,i+2,'量名／連接詞交界，詞組無可用整句分析；需排除省略。','possible')
    if best and segments is not None:
        cuts=lambda seq:set(__import__('itertools').accumulate(map(len,seq)))
        ps=[t[2] for t in best[2]]; delta=cuts(segments)^cuts(ps)
        if delta:
            k=min(delta);note('斷詞可疑',max(0,k-1),min(n,k+1),'依存候選與給定詞段邊界不同；兩解並存，不憑分歧判錯。','info')
    if not best:note('無法解析',0,n,'有限詞典或組合規則無全句解。','abstain')
    elif not sent or best[4]:note('分析未定',0,n,'僅有名詞性分析或至少一個詞類取自無標註搭配；需覆讀。','possible')
    alarm=any(x['severity']=='alarm' for x in issues)
    uncertain=not sent or (best and best[4]>0) or any(x['severity']=='possible' for x in issues)
    verdict='B' if alarm else 'U' if uncertain else 'R'
    return dict(text=line,form=n,verdict=verdict,label=LABEL[verdict],unresolved=not best or (not alarm and verdict=='U'),source_segments=segments,
        tokens=[] if not best else [dict(span=[a,b],text=w,tag=t) for a,b,w,t in best[2]],
        arcs=[] if not best else [dict(head=h,dependent=v,relation=t) for h,v,t in best[3]],
        root=None if not best else best[1],issues=issues,action='retain')

def couplet(a,b,sa=None,sb=None):
    rows=[diagnose(a,b,'upper',sa),diagnose(b,a,'lower',sb)];v=min((r['verdict'] for r in rows),key='BUR'.index)
    return dict(form=len(a),verdict=v,label=LABEL[v],unresolved=any(r['unresolved'] for r in rows),lines=rows,action='retain')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('lines',nargs='+');x=p.parse_args().lines
    if not 1<=len(x)<=2:p.error('請輸入單句或一聯兩句。')
    try:print(json.dumps(diagnose(x[0]) if len(x)==1 else couplet(*x),ensure_ascii=False,indent=2))
    except ValueError as e:p.error(str(e))
