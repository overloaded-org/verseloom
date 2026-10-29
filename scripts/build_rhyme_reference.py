#!/usr/bin/env python3
"""Rebuild the frozen Pingshui *candidate* reference without network access.

This is a source-transcription parser, not a contextual pronunciation annotator.
The original Unicode code points, explicit source defects, and supplemental
lists remain distinguishable. Run with --check to verify the checked-in outputs.
Only Python's standard library is required (Python 3.9+).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "data" / "reference_v02"
SOURCE_REVISION = 8264804
SOURCE_URL = "https://zh.wikisource.org/w/index.php?title=平水韻&oldid=8264804"
SOURCE_API_URL = (
    "https://zh.wikisource.org/w/api.php?action=parse&oldid=8264804"
    "&prop=wikitext%7Crevid%7Cdisplaytitle&format=json&formatversion=2"
)
STATUS = "source_candidate_needs_context"
SECTIONS = [
    ("上平聲", "upper_level", "P", "東冬江支微魚虞齊佳灰真文元寒刪"),
    ("下平聲", "lower_level", "P", "先蕭肴豪歌麻陽庚青蒸尤侵覃鹽咸"),
    ("上聲", "rising", "Z", "董腫講紙尾語麌薺蟹賄軫吻阮旱潸銑篠巧皓哿馬養梗迥有寢感琰豏"),
    ("去聲", "departing", "Z", "送宋絳寘未御遇霽泰卦隊震問願翰諫霰嘯效號箇禡漾敬徑宥沁勘豔陷"),
    ("入聲", "entering", "Z", "屋沃覺質物月曷黠屑藥陌錫職緝合葉洽"),
]
SECTION_INFO = {row[0]: row for row in SECTIONS}
HEADING_RE = re.compile(
    r"^(上平聲|下平聲|上聲|去聲|入聲)([一二三四五六七八九十]+)"
    r"\{\{\+\+\|([^{}|]+)\}\}$"
)
SUPPLEMENT_RE = re.compile(r"^【(詞|辭)】(.*)$")

# Hashes are of the original downloaded API responses, not reserialized JSON.
# A changed revision/content must be reviewed, not accepted by silently updating
# a checksum or applying character conversion.
FROZEN_INPUTS = {
    "wikisource_revision_8264804_api.json": {
        "sha256": "b6216e4dbc5cbd82894e3c958b688072ab0ff81eac677b720807ceb43a6cdb2c",
        "url": SOURCE_API_URL,
        "purpose": "primary Pingshui transcription, frozen revision",
    },
    "wikisource_revision_8264804_metadata.json": {
        "sha256": "422f2192208002f2197f28868cac70e3c3043a9171a9b12c0940463b693e3a80",
        "url": "https://zh.wikisource.org/w/api.php?action=query&prop=revisions&revids=8264804&rvprop=ids%7Ctimestamp%7Csha1&format=json&formatversion=2",
        "purpose": "revision timestamp and MediaWiki content SHA-1",
    },
    "wikisource_rightsinfo.json": {
        "sha256": "e113e25ced1ad609affcd45e846e0fe125dc28f2c5c5592d54b9bfb5a1e97ca0",
        "url": "https://zh.wikisource.org/w/api.php?action=query&meta=siteinfo&siprop=rightsinfo&format=json&formatversion=2",
        "purpose": "site copyright/licensing notice at retrieval",
    },
    "guangyun_revision_770095_api.json": {
        "sha256": "22cb4bb05e00d543bdb53538c40c46f1c9940e7f8af428de779ae7082a3b4431",
        "url": "https://zh.wikisource.org/w/api.php?action=parse&oldid=770095&prop=wikitext%7Crevid%7Cdisplaytitle&format=json&formatversion=2",
        "purpose": "primary Guangyun transcription used only as an issue witness",
    },
    "guangyun_revision_770099_api.json": {
        "sha256": "271bde13eb004e61f27d96f0b074ff0093699916a04a35af809fa8b0d627d952",
        "url": "https://zh.wikisource.org/w/api.php?action=parse&oldid=770099&prop=wikitext%7Crevid%7Cdisplaytitle&format=json&formatversion=2",
        "purpose": "primary Guangyun transcription used only as an issue witness",
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def numeral(value: int) -> str:
    digits = "零一二三四五六七八九"
    if 0 < value < 10:
        return digits[value]
    if 10 <= value < 20:
        return "十" + (digits[value % 10] if value % 10 else "")
    if 20 <= value <= 30:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    raise ValueError(f"Out-of-scope group number: {value}")


def is_han(char: str) -> bool:
    cp = ord(char)
    return any(
        start <= cp <= end
        for start, end in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x323AF),
        )
    )


def ensure_han(text: str, line: int) -> None:
    bad = [(char, f"U+{ord(char):04X}") for char in text if not is_han(char)]
    if not text or bad:
        raise ValueError(f"Unexpected main/supplement content at source line {line}: {bad!r}")


def parse_source(wikitext: str) -> tuple[list[dict], list[dict]]:
    """Parse all 106 groups, rejecting unrecognized nonempty content.

    Whitespace around a structural line is ignored. No whitespace, normalization,
    conversion, or editorial insertion is applied inside the character strings.
    """
    groups: list[dict] = []
    supplements: list[dict] = []
    section = None
    current = None
    seen_sections = []
    for line_number, raw_line in enumerate(wikitext.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        section_match = re.fullmatch(r"==(上平聲|下平聲|上聲|去聲|入聲)部==", line)
        if section_match:
            section = section_match[1]
            seen_sections.append(section)
            current = None
            continue
        match = HEADING_RE.fullmatch(line)
        if match:
            if match[1] != section:
                raise ValueError(f"Section-heading mismatch at line {line_number}")
            _, slug, tone, names = SECTION_INFO[section]
            ordinal = 1 + sum(g["section"] == section for g in groups)
            if ordinal > len(names) or match[2] != numeral(ordinal):
                raise ValueError(f"Missing/repeated/out-of-order group at line {line_number}")
            # These expected names are source-revision assertions, not a claim
            # that other historic group-name conventions are wrong.
            if match[3] != names[ordinal - 1]:
                raise ValueError(f"Unexpected source group name at line {line_number}")
            current = {
                "group_id": f"pingshui:{slug}:{ordinal:02d}",
                "section": section,
                "ordinal": ordinal,
                "ordinal_text": match[2],
                "name": match[3],
                "label": "".join(match.groups()),
                "tone": tone,
                "source_heading_line": line_number,
                "source_heading_raw": raw_line,
                "main_lines": [],
            }
            groups.append(current)
            continue
        supplement = SUPPLEMENT_RE.fullmatch(line)
        if supplement:
            if current is None:
                raise ValueError(f"Supplement outside a group at line {line_number}")
            ensure_han(supplement[2], line_number)
            supplements.append(
                {
                    "group_id": current["group_id"],
                    "label": current["label"],
                    "marker": supplement[1],
                    "source_line": line_number,
                    "source_raw": raw_line,
                    "text": supplement[2],
                    "characters": list(dict.fromkeys(supplement[2])),
                    "default_enabled": False,
                    "status": "supplement_origin_unidentified",
                }
            )
            continue
        if line_number == 1 and line.startswith("{{Header|") and line.endswith("}}"):
            continue
        if line in {"{{PD-old}}", "{{wikipedia}}", "[[Category:韻書]]"}:
            continue
        if current is None:
            raise ValueError(f"Unrecognized content at source line {line_number}: {line!r}")
        ensure_han(line, line_number)
        current["main_lines"].append({"source_line": line_number, "text": line})

    if seen_sections != [row[0] for row in SECTIONS]:
        raise ValueError(f"Section sequence is incomplete or duplicated: {seen_sections!r}")
    expected_counts = {row[0]: len(row[3]) for row in SECTIONS}
    actual_counts = dict(Counter(g["section"] for g in groups))
    if actual_counts != expected_counts or len(groups) != 106:
        raise ValueError(f"106-group completeness check failed: {actual_counts!r}")
    for group in groups:
        main_text = "".join(line["text"] for line in group["main_lines"])
        if not main_text:
            raise ValueError(f"Group without main text: {group['label']}")
        group["main_text"] = main_text
        group["characters"] = list(dict.fromkeys(main_text))
        group["within_group_duplicates"] = [
            {"character": c, "occurrences": n}
            for c, n in Counter(main_text).items()
            if n > 1
        ]
        group["status"] = STATUS
    return groups, supplements


def witness(filename: str, data: dict, exact_quote: str, context: str) -> dict:
    parsed = data["parse"]
    content = parsed["wikitext"]
    pos = content.find(exact_quote)
    if pos < 0 or content.find(exact_quote, pos + 1) >= 0:
        raise ValueError(f"Witness quote missing or non-unique in {filename}")
    return {
        "source_file": filename,
        "title": parsed["title"],
        "revision_id": parsed["revid"],
        "url": f"https://zh.wikisource.org/w/index.php?oldid={parsed['revid']}",
        "wikitext_codepoint_offset": pos,
        "exact_wikitext_excerpt": exact_quote,
        "context": context,
    }


def build(directory: Path) -> tuple[dict[str, bytes], dict]:
    inputs = {}
    manifest_sources = []
    for name, expected in FROZEN_INPUTS.items():
        raw = (directory / name).read_bytes()
        digest = sha256(raw)
        if digest != expected["sha256"]:
            raise ValueError(f"Frozen input SHA-256 mismatch: {name}")
        inputs[name] = json.loads(raw)
        manifest_sources.append(
            {
                "filename": name,
                "url": expected["url"],
                "purpose": expected["purpose"],
                "bytes": len(raw),
                "sha256": digest,
                "retrieval_date_utc": "2026-10-29",
            }
        )
    parsed = inputs["wikisource_revision_8264804_api.json"]["parse"]
    metadata = inputs["wikisource_revision_8264804_metadata.json"]["query"]["pages"][0]
    revision = metadata["revisions"][0]
    if parsed["title"] != "平水韻" or parsed["revid"] != SOURCE_REVISION:
        raise ValueError("Wrong primary text or revision")
    if revision["revid"] != SOURCE_REVISION:
        raise ValueError("Metadata is for a different revision")
    wikitext = parsed["wikitext"]
    source_bytes = wikitext.encode("utf-8")
    if hashlib.sha1(source_bytes).hexdigest() != revision["sha1"]:
        raise ValueError("Raw wikitext does not match MediaWiki revision content SHA-1")
    groups, supplements = parse_source(wikitext)
    by_id = {group["group_id"]: group for group in groups}
    memberships = defaultdict(list)
    for group in groups:
        for char in group["characters"]:
            memberships[char].append(
                {
                    "group_id": group["group_id"],
                    "label": group["label"],
                    "tone": group["tone"],
                    "source_lines": [
                        item["source_line"] for item in group["main_lines"] if char in item["text"]
                    ],
                    "status": STATUS,
                }
            )
    memberships = dict(sorted(memberships.items()))
    source = {
        "title": "平水韻",
        "source_url": SOURCE_URL,
        "revision_id": SOURCE_REVISION,
        "revision_timestamp_utc": revision["timestamp"],
        "mediawiki_content_sha1": revision["sha1"],
        "wikitext_sha256": sha256(source_bytes),
        "wikitext_file": "pingshui_8264804.wikitext",
        "printed_edition": None,
        "printed_edition_note": "The source Header.from field is empty; no print exemplar is identified.",
    }
    policy = {
        "membership_status": STATUS,
        "contextual_gold": False,
        "unicode_policy": "exact source code points; no normalization, traditional/simplified conversion, or alias union",
        "main_index": "main character lines only; same-group repeats deduplicated in the index",
        "supplement_policy": "【詞】 and 【辭】 stored separately and disabled by default; origins unidentified",
        "editorial_policy": "source defects retained and flagged; no correction overlay applied",
        "lookup_absence": "unknown in this reference, not proof that the character or reading is impossible",
        "poem_annotation": "select contextual readings separately and preserve unresolved alternatives",
    }
    gy1 = inputs["guangyun_revision_770095_api.json"]
    gy2 = inputs["guangyun_revision_770099_api.json"]
    # Direct excerpts are deliberately short. Guangyun is an independent
    # historical witness, not a replacement Pingshui assignment dictionary.
    issues = [
        {
            "issue_id": "PS8264804-001",
            "character": "先",
            "kind": "source_head_character_omitted_from_main_list",
            "group_id": "pingshui:lower_level:01",
            "source_state": "The heading is 一先; 先 is absent from its main and supplement lists. The main index has 先 in 去聲十七霰 only.",
            "evidence": witness(
                "guangyun_revision_770099_api.json", gy2,
                "先{{SK notes|先後也又姓左傳晉有先軫蘇前切又蘇薦切四}}",
                "卷二下平聲一先; the entry explicitly records both 蘇前切 and another reading 蘇薦切.",
            ),
            "candidate_editorial_action": "Add a level-tone 一先 candidate only after approving an explicit editorial overlay; retain the existing departing-tone candidate.",
            "applied": False,
        },
        {
            "issue_id": "PS8264804-002",
            "character": "啼",
            "kind": "common_historical_character_absent_from_reference",
            "group_id": "pingshui:upper_level:08",
            "source_state": "啼 is absent from all main and supplemental lists.",
            "evidence": witness(
                "guangyun_revision_770095_api.json", gy1,
                "嗁{{SK notes|泣也説文曰號也杜奚切六十}}　啼㖒{{SK notes|竝上同}}",
                "卷一上平聲十二齊; 啼 is listed with 嗁, 杜奚切. Assignment to Pingshui 八齊 requires an explicit editorial mapping.",
            ),
            "candidate_editorial_action": "Review the absent 八齊 candidate against a second Pingshui edition before adding it to an operational overlay.",
            "applied": False,
        },
        {
            "issue_id": "PS8264804-003",
            "character": "頹",
            "kind": "suspicious_assignment_with_variant_sensitive_witness",
            "group_id": "pingshui:entering:01",
            "source_state": "頹 occurs in 入聲一屋 main text and 上平聲十灰 【詞】 supplement. The main index retains only the former, with this warning.",
            "evidence": witness(
                "guangyun_revision_770095_api.json", gy1,
                "穨{{SK notes|暴風也杜囘切十三}}頽{{SK notes|上同}}",
                "卷一上平聲十五灰 records 頽, not the same Unicode character as 頹. No alias conversion is applied here.",
            ),
            "candidate_editorial_action": "Review 頹/頽 variant evidence and an independent Pingshui edition; do not silently move or merge either glyph.",
            "applied": False,
        },
        {
            "issue_id": "PS8264804-004",
            "character": "龍",
            "kind": "suspicious_cross_tone_assignment_unresolved",
            "group_id": "pingshui:entering:14",
            "source_state": "The same glyph 龍 occurs in 上平聲二冬 and 入聲十四緝 main text.",
            "evidence": None,
            "candidate_editorial_action": "Retain both source candidates and require edition/reading review before using the entering-tone candidate.",
            "applied": False,
        },
        {
            "issue_id": "PS8264804-005",
            "character": None,
            "kind": "group_label_convention_requires_explicit_alias",
            "group_id": "pingshui:rising:28",
            "source_state": "The frozen source labels this group 二十八琰; other Pingshui lists commonly use 二十八儉. This build preserves 琰 and uses section+ordinal for identity.",
            "evidence": None,
            "candidate_editorial_action": "Keep any cross-edition label aliases in a separately sourced table; do not rename this source label.",
            "applied": False,
        },
    ]
    issue_ids_by_character = defaultdict(list)
    for issue in issues:
        if issue["character"] is not None:
            issue_ids_by_character[issue["character"]].append(issue["issue_id"])
    for char, records in memberships.items():
        if char in issue_ids_by_character:
            for record in records:
                record["source_issue_ids"] = issue_ids_by_character[char]

    # Assertions capture substantive failure modes of the previous 105-group
    # simplified table: a missing group, collapsed variants, lost ambiguity, and
    # claiming source candidates as verified contextual readings.
    assert by_id["pingshui:rising:03"]["main_text"] == "講港棒蚌項缿玤傋耩"
    assert memberships["講"][0]["group_id"] == "pingshui:rising:03"
    assert "講" in memberships and "讲" not in memberships
    assert "髮" in memberships and "發" in memberships
    assert all(record["status"] == STATUS for records in memberships.values() for record in records)
    assert len(memberships["先"]) == 1 and memberships["先"][0]["tone"] == "Z"
    assert "啼" not in memberships and "頽" not in memberships
    assert {r["tone"] for r in memberships["龍"]} == {"P", "Z"}

    duplicates = [
        {"group_id": g["group_id"], "label": g["label"], "duplicates": g["within_group_duplicates"]}
        for g in groups if g["within_group_duplicates"]
    ]
    ambiguous = {
        char: [r["group_id"] for r in records]
        for char, records in memberships.items() if len(records) > 1
    }
    cross_tone = {
        char: [r["group_id"] for r in records]
        for char, records in memberships.items() if len({r["tone"] for r in records}) > 1
    }
    supplemental_pairs = {
        (char, item["group_id"])
        for item in supplements for char in item["characters"]
    }
    main_pairs = {(char, r["group_id"]) for char, records in memberships.items() for r in records}
    counts = {
        "groups": len(groups),
        "groups_by_section": dict(Counter(g["section"] for g in groups)),
        "main_character_occurrences": sum(len(g["main_text"]) for g in groups),
        "main_unique_character_group_pairs": len(main_pairs),
        "main_unique_characters": len(memberships),
        "within_group_duplicate_excess_occurrences": sum(len(g["main_text"]) - len(g["characters"]) for g in groups),
        "groups_with_within_group_duplicates": len(duplicates),
        "characters_in_multiple_groups": len(ambiguous),
        "characters_in_both_level_and_oblique_groups": len(cross_tone),
        "supplemental_lines": len(supplements),
        "supplemental_lines_by_marker": dict(Counter(s["marker"] for s in supplements)),
        "supplemental_unique_character_group_pairs": len(supplemental_pairs),
        "supplemental_pairs_not_in_main": len(supplemental_pairs - main_pairs),
        "supplemental_characters_not_in_main": len({c for c, _ in supplemental_pairs} - set(memberships)),
        "main_plus_supplement_unique_characters_diagnostic_only": len(set(memberships) | {c for c, _ in supplemental_pairs}),
        "editorial_issues_flagged": len(issues),
        "editorial_corrections_applied": 0,
    }
    audit = {
        "schema_version": "0.2",
        "source": source,
        "policy": policy,
        "counts": counts,
        "group_validation": {
            "expected_total": 106,
            "expected_section_counts": {row[0]: len(row[3]) for row in SECTIONS},
            "actual_section_counts": counts["groups_by_section"],
            "missing_groups": [],
            "duplicate_group_ids": [],
            "out_of_order_group_ids": [],
            "head_character_absent_from_own_main": [g["group_id"] for g in groups if g["name"] not in g["characters"]],
            "interpretation": "All 106 structural groups and frozen-source names parse correctly. This does not establish completeness or accuracy of their contents.",
        },
        "within_group_duplicates": duplicates,
        "cross_group_memberships": ambiguous,
        "cross_tone_memberships": cross_tone,
        "open_issues": issues,
        "known_limitations": [
            "No identifiable printed exemplar or page-image collation in this Pingshui transcription.",
            "Source membership is neither a context-sensitive pronunciation nor a gold tone label.",
            "106 complete headings do not imply a complete character inventory.",
            "Unidentified supplemental-list provenance prevents default inclusion.",
            "Original and variant glyphs remain separate; unknown coverage is possible for the traditional corpus.",
            "Flagged defects are not exhaustive; unflagged entries have not all been individually verified.",
            "Guangyun witness excerpts support issue review, not automatic Pingshui reassignment or variant equivalence.",
        ],
    }
    outputs = {
        "pingshui_8264804.wikitext": source_bytes,
        "pingshui_106_source.json": encode_json({
            "schema_version": "0.2", "source": source, "policy": policy, "groups": groups,
        }),
        "pingshui_106_memberships.json": encode_json({
            "schema_version": "0.2", "source": source, "policy": policy, "characters": memberships,
        }),
        "supplemental_lists.json": encode_json({
            "schema_version": "0.2", "source": source, "policy": policy, "lists": supplements,
        }),
        "editorial_issues.json": encode_json({
            "schema_version": "0.2", "corrections_applied": 0, "issues": issues,
        }),
        "source_audit.json": encode_json(audit),
    }
    rights = inputs["wikisource_rightsinfo.json"]["query"]["rightsinfo"]
    outputs["LICENSE_NOTICE.md"] = license_notice(rights).encode("utf-8")
    outputs["README.md"] = readme(counts).encode("utf-8")
    outputs["source_manifest.json"] = encode_json({
        "schema_version": "0.2",
        "primary_source": source,
        "source_snapshots": manifest_sources,
        "derived_files": [
            {"filename": name, "bytes": len(raw), "sha256": sha256(raw)}
            for name, raw in sorted(outputs.items())
        ],
        "build_script": {
            "path_relative_to_project": "scripts/build_rhyme_reference.py",
            "sha256": sha256(Path(__file__).read_bytes()),
            "runtime": "Python 3.9+ standard library",
            "network_required_to_rebuild": False,
        },
        "licensing": {
            "source_work_public_domain_mark": "{{PD-old}} in the frozen Pingshui wikitext",
            "site_rightsinfo": rights,
            "notice": "LICENSE_NOTICE.md",
            "scope": "source transcription and derivative reference data only; not a license declaration for unrelated project files",
        },
        "normalization_or_character_conversion": None,
        "editorial_corrections_applied": [],
    })
    return outputs, counts


def license_notice(rights: dict) -> str:
    return f"""# 来源与许可说明

## 原作与录文应区分

本目录保存《平水韻》与《重修廣韻（四庫全書本）》的维基文库录文快照，
不是本项目新创作的历史字书。《平水韻》固定修订在源码中标注 `{{{{PD-old}}}}`。
原作的公有领域地位不等于可忽略现代录文、编排或校订的署名要求。

维基文库 `siteinfo/rightsinfo` 的实际响应为 **{rights['text']}**，许可链接：
<{rights['url']}>。原响应保存在 `wikisource_rightsinfo.json`。

本目录对录文及其派生参考数据保留维基文库作者/贡献者署名、固定版本链接与
CC BY-SA 4.0 通知；如现代贡献适用版权，复用或修改这些内容时应遵循该许可。
本通知不将整个项目的代码或其他独立文件重新授权为此许可。

## 署名与固定版本

- 《平水韻》：来源页署名劉淵、王文鬱；电子录文及现代编排贡献者为中文维基文库贡献者。
  固定版本：<{SOURCE_URL}>。
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
"""


def readme(counts: dict) -> str:
    return f"""# G1 韵表参考 v0.2：106 部原字形候选集合

这份数据解决旧表缺少“上声三讲”和简繁表面不一致的问题：冻结维基文库
《平水韻》修订 **8264804**，按其原始 Unicode 字形解析全部 106 部。
它是可追溯的字—韵部候选集合，**不是诗句语境读音金标**。
“106 部齐全”只证明结构完整，不证明所收字、各字归属或录文没有错误。

## 可复算结果

| 项目 | 实际值 |
|---|---:|
| 部数 | {counts['groups']} |
| 上平／下平／上／去／入 | 15／15／29／30／17 |
| 主表字符出现次数 | {counts['main_character_occurrences']:,} |
| 主表唯一字—部关系 | {counts['main_unique_character_group_pairs']:,} |
| 主表不同原字形 | {counts['main_unique_characters']:,} |
| 同部重复的多余出现次数 | {counts['within_group_duplicate_excess_occurrences']} |
| 跨部多归属字 | {counts['characters_in_multiple_groups']:,} |
| 同时有平、仄候选的字 | {counts['characters_in_both_level_and_oblique_groups']:,} |
| 独立保留的补充行 | {counts['supplemental_lines']} |
| 已应用的正文勘改 | 0 |

上聲三講原文为：**講港棒蚌項缿玤傋耩**。部号、部名、顺序、106 个
唯一组 ID 均经过构建时验证。`pingshui:rising:03` 等 ID 用声部和序号标识，
原文的“二十八琰”照录，避免不同韵目名惯例造成身份混淆。

## 文件

- `wikisource_revision_8264804_api.json`：下载的原始 API 字节快照。
- `pingshui_8264804.wikitext`：从 API 字段逐字节抽取的 UTF-8 原文；
  SHA-1 与 MediaWiki 该修订元数据一致。
- `pingshui_106_source.json`：106 部、原文字串、来源行号、去重字集及同部重复。
- `pingshui_106_memberships.json`：按字查询的完整主表候选索引。
  每项 `status=source_candidate_needs_context`，不输出 `verified`。
- `supplemental_lists.json`：原文的 `【詞】`、`【辭】` 行；来历未明，默认禁用。
- `source_audit.json`：完整统计、所有同部重复、所有跨部与跨平仄归属、问题清单。
- `editorial_issues.json`：独立勘误议题及短段原文证据，`applied=false`。
- `guangyun_revision_770095_api.json`、`guangyun_revision_770099_api.json`：
  问题复核所用《廣韻》固定录文，保留全文原始 API 响应。
- `source_manifest.json`：来源 URL、版本、时间、原始及派生文件 SHA-256。
- `LICENSE_NOTICE.md`：历史原作与现代录文的许可、署名和处理说明。

## 离线重建与验证

在项目根目录运行：

```bash
python scripts/build_rhyme_reference.py
python scripts/build_rhyme_reference.py --check
```

只使用 Python 3.9+ 标准库。构建先检查五份冻结输入的 SHA-256，并检查
抽取原文的 SHA-1 与修订元数据一致。遇到缺部、重复部号、次序变化、
未识别内容、非汉字正文或修订不符会失败，不会静默删掉异常内容。
`--check` 重算所有派生文件并逐字节核对，适合在来源升级后发现差异。

## 使用约束

字形按源码原样保留，以繁体为主；`發`、`髮`、`頹`、`頽` 是不同键。
没有运行简繁转换、NFC/NFKC 规范化或一对一字形映射。一个字的多个
候选部全部保留，同部重复只在索引中去重，原文字串仍保留重复。

`tone=P/Z` 是候选韵部的平仄分类，不是该字在某一诗句中的确定读音。
即使字只有一条候选，也不能自动标为语境 `verified`；表中缺字应记为
`unknown`，不能直接判诗不合律。对语境金标，应另建带诗句、义项、
读音依据、审核人及未决候选的记录。

## 已发现、尚未自动修正的问题

1. 下平一先部标题有“先”，本部名单却漏收“先”；主表只把它列在去声十七霰。
   《廣韻》卷二一先条记录苏前切及另读苏荐切，可作为显式补录的历史证据。
2. “啼”在全部主表及补充表中缺失。《廣韻》卷一齐韵记录杜奚切及啼的同文，
   支持启动对平水八齐候选的复核；这里没有自动补入。
3. “頹”在入声一屋主表与上平十灰补充行同时出现。《廣韻》灰韵见的是
   **頽** 与穨、杜囘切；因字形不同，这份证据不自动证明两键可合并。
4. “龍”同时列在二冬与入声十四缉，待第二版本或读音材料复核。
5. 该版用“上聲二十八琰”；常见的“二十八儉”应作为跨版本别名另行溯源。

维基文库该页 `Header.from` 为空，没有注明底本。本版保留的 `【詞】`、
`【辭】` 补充行也未说明来源。这些问题使它适合作为**透明、可审计的候选层**，
不能代替经过底本校勘的权威韵书或人工语境标注。问题清单并非穷尽性勘误。

来源与许可证见 `LICENSE_NOTICE.md`。原始快照固定于 2026-10-29 获取，
《平水韻》修订时间为 2026-08-12T08:23:22Z。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY,
                        help="Directory containing frozen source snapshots and derived outputs")
    parser.add_argument("--check", action="store_true", help="Verify derived files without writing them")
    args = parser.parse_args()
    try:
        outputs, counts = build(args.directory)
        if args.check:
            differences = [
                name for name, raw in outputs.items()
                if not (args.directory / name).exists() or (args.directory / name).read_bytes() != raw
            ]
            if differences:
                raise ValueError("Derived output differs or is absent: " + ", ".join(differences))
        else:
            for name, raw in outputs.items():
                (args.directory / name).write_bytes(raw)
        print(json.dumps({"mode": "check" if args.check else "build", "ok": True, "outputs": len(outputs), "counts": counts}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        print(f"Reference build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
