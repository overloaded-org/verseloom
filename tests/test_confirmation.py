from copy import deepcopy
import unittest

from wordconv_lab.confirmation import select_confirmation


def poem(number=0, length=5, old=False):
    p = {"poem_id": f"p{number}", "lines": ["".join(chr(0x4e00 + number*100 + i*length + j)
         for j in range(length)) for i in range(8)], "length": length,
         "author": "合成作者", "title": "工程测试", "source_refs": []}
    if old:
        p.update(family_id=f"old{number}", split="train")
    return p


class ConfirmationTests(unittest.TestCase):
    def test_exact_prior_text_is_excluded_from_any_old_split(self):
        old = [poem(i, old=True) for i in range(3)]
        for p, split in zip(old, ["train", "dev", "test"]): p["split"] = split
        new = [deepcopy(p) for p in old] + [poem(3)]
        result = select_confirmation(old, new)
        self.assertEqual([p["poem_id"] for p in result["poems"]], ["p3"])
        self.assertEqual(result["audit"]["new_exact_text_already_in_prior"], 3)
        self.assertEqual(result["audit"]["prior_exclusion_by_split"], {"train": 1, "dev": 1, "test": 1})

    def test_new_bridge_to_old_excludes_indirectly_connected_poem(self):
        old = poem(0, old=True)
        bridge, indirect, independent = poem(1), poem(2), poem(3)
        bridge["lines"][0] = old["lines"][0]
        indirect["lines"][3] = bridge["lines"][4]
        result = select_confirmation([old], [bridge, indirect, independent])
        self.assertEqual([p["poem_id"] for p in result["poems"]], ["p3"])
        decision = {r["poem_id"]: r for r in result["selection_records"]}
        self.assertEqual(decision["p2"]["prior_component_poem_ids"], ["p0"])
        self.assertEqual(decision["p2"]["decision"], "exclude_prior_connected_component")

    def test_jaccard_relation_without_shared_line_is_excluded(self):
        old, new = poem(0, 7, old=True), poem(1, 7)
        alphabet = "".join(chr(0x4f00+i) for i in range(16))*2
        old["lines"] = [alphabet[i:i+7] for i in range(0, 16, 2)]
        new["lines"] = [alphabet[i:i+7] for i in range(1, 16, 2)]
        result = select_confirmation([old], [new])
        self.assertEqual(result["poems"], [])
        edge = result["joint_family_edges"][0]
        self.assertEqual(edge["shared_lines"], [])
        self.assertEqual(edge["bigram_jaccard"]["value"], 1.0)

    def test_selection_cap_applies_to_families_not_poems(self):
        old = poem(0, old=True)
        a, b = poem(1), poem(2)
        b["lines"][1] = a["lines"][2]
        result = select_confirmation([old], [a, b], max_families=1)
        self.assertEqual(len(result["poems"]), 2)
        self.assertEqual(len(result["tasks"]), 8)
        self.assertEqual(result["audit"]["selected_families"], 1)
        self.assertTrue(all(t["split"] == "confirm" for t in result["tasks"]))

    def test_hash_selection_is_order_independent_and_does_not_mutate_inputs(self):
        old = [poem(0, old=True)]
        new = [poem(i) for i in range(1, 6)]
        snapshots = deepcopy((old, new))
        a = select_confirmation(old, new, max_families=2)
        b = select_confirmation(list(reversed(old)), list(reversed(new)), max_families=2)
        self.assertEqual(a, b)
        self.assertEqual((old, new), snapshots)
        ranks = [c for c in a["components"] if c["eligible"]]
        expected = sorted(ranks, key=lambda c: (c["selection_hash"], c["family_id"]))[:2]
        self.assertEqual({c["family_id"] for c in expected}, {p["family_id"] for p in a["poems"]})

    def test_identity_collision_and_inconsistent_exact_identity_are_rejected(self):
        old = poem(0, old=True)
        wrong_id = deepcopy(old)
        wrong_id["poem_id"] = "different"
        with self.assertRaisesRegex(ValueError, "inconsistent stable identity"):
            select_confirmation([old], [wrong_id])
        wrong_text = poem(1)
        wrong_text["poem_id"] = old["poem_id"]
        with self.assertRaisesRegex(ValueError, "different source text"):
            select_confirmation([old], [wrong_text])

    def test_duplicate_input_and_empty_old_set_are_rejected(self):
        old = poem(0, old=True)
        with self.assertRaisesRegex(ValueError, "duplicate poem identity"):
            select_confirmation([old], [poem(1), poem(1)])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            select_confirmation([], [poem(1)])

    def test_invalid_family_budget_and_old_split_are_rejected(self):
        old = poem(0, old=True)
        for cap in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError): select_confirmation([old], [poem(1)], max_families=cap)
        old["split"] = "confirm"
        with self.assertRaisesRegex(ValueError, "existing train/dev/test"):
            select_confirmation([old], [poem(1)])


if __name__ == "__main__":
    unittest.main()
