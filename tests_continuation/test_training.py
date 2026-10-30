import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ranktrain',ROOT/'scripts/train_continuation_ranker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RankerTests(unittest.TestCase):
    def test_uniform_ties_do_not_privilege_gold_position(self):
        r,mrr=m.rank_scores(np.zeros((3,16)));np.testing.assert_equal(r,np.full(3,1/16))
    def test_rank_order_and_partial_ties(self):
        r,_=m.rank_scores(np.array([[3,2,1],[3,3,1],[1,3,2]]));np.testing.assert_equal(r,[1,.5,0])
    def test_disallowed_partitions_not_decoded(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x';p.write_bytes(b'{"split":"test_reserve", BROKEN JSON}\n')
            rows,skipped=m.read_partition(p,{'train_candidate'});self.assertEqual(rows,[]);self.assertEqual(skipped,1)
    def test_conflicting_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x';p.write_text('{"split":"train_candidate","split":"test_reserve"}\n')
            with self.assertRaises(ValueError):m.read_partition(p,{'train_candidate'})
    def test_pair_objective_learns_difference_without_content(self):
        x=np.zeros((20,2,len(m.NAMES)));x[:,0,0]=1
        a,_,_=m.fit(x);b,_,_=m.fit(x)
        self.assertGreater(a[0],0);np.testing.assert_array_equal(a,b)
    def test_model_has_no_authored_content_or_stored_sentences(self):
        p=ROOT/'output/continuation-v1/ranker.json'
        if not p.exists():self.skipTest('Integration model checked by Node separately')
        d=json.loads(p.read_bytes());self.assertEqual(len(d['coefficients']),8);self.assertEqual(d['authoredContentLexemes'],0)
        self.assertNotIn('lines',d);self.assertNotIn('lexicon',d);self.assertFalse(d['literaryQualityCertified'])
    def test_output_refuses_overwrite_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):m.run(Path('missing'),Path('missing'),Path(t))
if __name__=='__main__':unittest.main()
