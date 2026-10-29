"""Boundary checks for the user-facing completion entry point."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from wordconv_lab.completion_v05 import build_query, load_training_poems, selected_checkpoint


class CompletionBoundaryTests(unittest.TestCase):
    def test_query_shape_and_exact_orthography(self):
        query = build_query(" 空山新雨後 ", " 晚來秋 ")
        self.assertEqual(query.upper_line, "空山新雨後")
        self.assertEqual(query.lower_suffix, "晚來秋")
        self.assertEqual(query.length, 5)
        for upper, suffix in [("空山新雨後，", "晚來秋"), ("空山新雨後", "來秋"),
                              ("空山A雨後", "晚來秋")]:
            with self.assertRaises(ValueError):
                build_query(upper, suffix)

    def test_held_out_rows_never_decoded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poems.jsonl"
            path.write_bytes(b'{"split":"test",BROKEN}\n{"split":"dev",BROKEN}\n'
                             b'{"split":"train","poem_id":"example"}\n')
            self.assertEqual(load_training_poems(path), [{"split": "train", "poem_id": "example"}])

    def test_selection_identity_tamper_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / "update_0400.npz"
            checkpoint.write_bytes(b"fixed-checkpoint-bytes")
            record = {"seed": 0, "path": checkpoint.name,
                      "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            selection = {"primary_arm": "full_w3", "selection_scope": "dev_only_not_independent_confirmation",
                         "arms": {"full_w3": {"step": 400, "models": [record]}}}
            path = run / "selection_frozen.json"
            path.write_text(json.dumps(selection))
            self.assertEqual(selected_checkpoint(run, 0)[0], checkpoint)
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                selected_checkpoint(run, 0)
            record["path"] = "../outside.npz"
            path.write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                selected_checkpoint(run, 0)
            with self.assertRaises(ValueError):
                selected_checkpoint(run, True)


if __name__ == "__main__":
    unittest.main()
