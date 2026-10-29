"""Source-grounded finite grammar for BASIC five/seven-character regulated verse.

This implements a deliberately incomplete metrical profile, not a classifier of
all historically legitimate poetry. P/Z labels must already have been resolved
from appropriate historical readings; modern Mandarin pinyin is not substituted.
Source: Zeng Yongyi, NTU OCW lecture 2, physical PDF pages 3-5.
"""
from dataclasses import asdict, dataclass
import itertools

SOURCE = "https://ocw.aca.ntu.edu.tw/uploads/course_item_file/file/684/104S110_CT02L01.pdf"
FIVE_LINE = {("P", "P"): "PPZZP", ("P", "Z"): "PPPZZ",
             ("Z", "P"): "ZZZPP", ("Z", "Z"): "ZZPPZ"}


def opposite(tone: str) -> str:
    if tone not in ("P", "Z"):
        raise ValueError("tone must be P or Z")
    return "Z" if tone == "P" else "P"


@dataclass(frozen=True)
class Template:
    template_id: str
    length: int
    start_second_tone: str
    first_rhymed: bool
    core: tuple[str, ...]
    rhyme_rows: tuple[int, ...]
    source: str = SOURCE

    def to_dict(self):
        return asdict(self)


def generate_template(length: int, start_second_tone: str, first_rhymed: bool) -> Template:
    if type(length) is not int or length not in (5, 7):
        raise ValueError("only 5/7-character regulated verse is supported")
    if type(first_rhymed) is not bool:
        raise ValueError("first_rhymed must be boolean")
    inverse = opposite(start_second_tone)
    seconds = (start_second_tone, inverse, inverse, start_second_tone,
               start_second_tone, inverse, inverse, start_second_tone)
    core = []
    rhymes = (0, 1, 3, 5, 7) if first_rhymed else (1, 3, 5, 7)
    for row, second in enumerate(seconds):
        ending = "P" if row in rhymes else "Z"
        five_second = second if length == 5 else opposite(second)
        line = FIVE_LINE[(five_second, ending)]
        core.append(line if length == 5 else second * 2 + line)
    return Template(f"L{length}-{start_second_tone}-R{int(first_rhymed)}", length,
                    start_second_tone, first_rhymed, tuple(core), rhymes)


def all_templates(length: int) -> tuple[Template, ...]:
    return tuple(generate_template(length, tone, rhyme) for tone in ("P", "Z") for rhyme in (False, True))


def variants(core: str, profile: str = "common_prefix") -> tuple[str, ...]:
    if profile not in ("strict_core", "common_prefix"):
        raise ValueError("unsupported profile")
    if len(core) not in (5, 7) or any(x not in "PZ" for x in core):
        raise ValueError("invalid core line")
    five = core[-5:]
    if five not in FIVE_LINE.values():
        raise ValueError("line is not a supported basic form")
    if len(core) == 7 and core[:2] != opposite(five[1]) * 2:
        raise ValueError("seven-character prefix disagrees with five-character core")
    if profile == "strict_core":
        return (core,)
    # Safe, sourced narrow relaxation only. No blanket 'odd positions are free'.
    free = ([0] if len(core) == 7 else [])
    if five != "PPZZP":  # This first position otherwise creates an isolated ping.
        free.append(len(core) - 5)
    found = set()
    for bits in itertools.product((False, True), repeat=len(free)):
        line = list(core)
        for p, flip in zip(free, bits):
            if flip:
                line[p] = opposite(line[p])
        found.add("".join(line))
    return tuple(sorted(found))


def audit_tones(lines: list[str] | tuple[str, ...], *, profile: str = "common_prefix") -> dict:
    if len(lines) != 8 or len(lines[0]) not in (5, 7) or any(len(x) != len(lines[0]) for x in lines):
        raise ValueError("expected eight equal lines of five or seven tones")
    if any(t not in "PZ?" for line in lines for t in line):
        raise ValueError("tone labels must be P/Z or ? for unresolved")
    if any("?" in line for line in lines):
        return {"status": "needs_reading_review", "profile": profile, "matches": [],
                "literary_quality": "not_evaluated"}
    evaluations = []
    for template in all_templates(len(lines[0])):
        closest = []
        changed = []
        for row, (line, base) in enumerate(zip(lines, template.core)):
            options = variants(base, profile)
            best = min(options, key=lambda candidate: (sum(a != b for a, b in zip(line, candidate)), candidate))
            closest.append(best)
            changed.extend([row, col] for col, (a, b) in enumerate(zip(line, best)) if a != b)
        evaluations.append({"template_id": template.template_id, "tone_edit_distance": len(changed),
                            "closest_supported_lines": closest, "differing_positions": changed})
    matches = [x["template_id"] for x in evaluations if x["tone_edit_distance"] == 0]
    return {"status": "matches_supported_profile" if matches else "outside_supported_profile",
            "profile": profile, "matches": matches,
            "nearest": sorted(evaluations, key=lambda x: (x["tone_edit_distance"], x["template_id"]))[0],
            "warning": "Outside this incomplete profile does NOT mean the poem is invalid; rhyme/meaning/antithesis are not certified.",
            "literary_quality": "not_evaluated", "source": SOURCE}
