import math
import unittest

import numpy as np

from wordconv_lab.restoration import BidirNGram,VisibleQuery
from wordconv_lab.retrieval import PhraseRetriever,_builtin_float_sum


def fixture_poems():
    return [
        {"poem_id":"five_a","split":"train","length":5,
         "lines":["甲乙丙丁戊","己庚辛壬癸"]*4},
        {"poem_id":"five_b","split":"train","length":5,
         "lines":["甲乙辛壬癸","丙丁甲乙戊"]*4},
        {"poem_id":"seven","split":"train","length":7,
         "lines":["甲乙丙丁戊甲乙","己庚辛壬癸己庚"]*4},
    ]


def vocabulary_for(poems):
    return {c:i for i,c in enumerate(["<UNK>"]+sorted({c for p in poems for line in p["lines"] for c in line}))}


class PhraseRetrievalTests(unittest.TestCase):
    def test_vectorized_sum_matches_python_runtime_builtin(self):
        rng=np.random.default_rng(721)
        for terms in (np.array([[1e16,1.,-1e16],[1.,1e-16,1e-16]]),
                      rng.normal(size=(500,7))*rng.lognormal(0,10,size=(500,7))):
            expected=np.asarray([sum(row.tolist()) for row in terms])
            np.testing.assert_array_equal(_builtin_float_sum(tuple(terms.T)),expected)

    def test_fast_scores_and_full_orders_equal_original_for_five_seven_and_unknown(self):
        poems=fixture_poems()
        vocabulary=vocabulary_for(poems)
        model=PhraseRetriever(poems,vocabulary)
        reference=BidirNGram(vocabulary).fit(poems)
        for query in [VisibleQuery(5,"甲乙丙丁戊","辛壬癸"),
                      VisibleQuery(5,"未知辛壬癸","天壬地"),
                      VisibleQuery(7,"甲乙丙丁戊甲乙","辛壬癸己庚"),
                      VisibleQuery(7,"未知甲乙丙丁戊","天地玄黃宇")]:
            with self.subTest(query=query):
                components=model.score_components(query)
                expected=np.asarray([reference.score(query,span) for span in components["candidates"]])
                np.testing.assert_array_equal(components["ngram_scores"],expected)
                for pool in ("wide","narrow"):
                    allowed=[i for i in range(len(expected)) if pool=="wide" or components["narrow_mask"][i]]
                    order=sorted(allowed,key=lambda i:(-expected[i],components["candidates"][i]))
                    result=model.rank_components(components,pool=pool)
                    np.testing.assert_array_equal(result["indices"],order)
                    self.assertEqual(result["candidates"],tuple(components["candidates"][i] for i in order))

    def test_exact_unknown_collisions_use_original_lexical_tie_rule(self):
        poems=fixture_poems()
        model=PhraseRetriever(poems,{"<UNK>":0})
        reference=BidirNGram({"<UNK>":0}).fit(poems)
        for query in [VisibleQuery(5,"天地玄黃宇","宙洪荒"),
                      VisibleQuery(7,"天地玄黃宇宙洪","日月盈昃辰")]:
            components=model.score_components(query)
            expected=[reference.score(query,span) for span in components["candidates"]]
            self.assertEqual(len(set(expected)),1)
            np.testing.assert_array_equal(components["ngram_scores"],expected)
            self.assertEqual(model.rank_components(components)["candidates"],tuple(sorted(components["candidates"])))
            self.assertTrue(components["unknown_visible_characters"])

    def test_hand_counted_priors_and_bonuses_include_all_aligned_and_boundary_positions(self):
        poems=fixture_poems()[:1]
        model=PhraseRetriever(poems,vocabulary_for(poems))
        components=model.score_components(VisibleQuery(5,"甲乙丙丁戊","辛壬癸"))
        spans=components["candidates"]
        i=spans.index("己庚")
        self.assertEqual(len(spans),8)
        self.assertEqual(components["span_prior"][i],1/8)
        self.assertEqual(components["alignment_counts"][i],4)
        self.assertEqual(components["boundary_counts"][i],4)
        expected=math.log1p(4/(5/8))
        self.assertAlmostEqual(components["alignment_bonus"][i],expected)
        self.assertAlmostEqual(components["boundary_bonus"][i],expected)
        self.assertEqual(components["alignment_bonus"][spans.index("甲乙")],0)
        interior=model.score_components(VisibleQuery(5,"乙丙甲乙戊","壬癸戊"))
        j=spans.index("庚辛")
        self.assertEqual(interior["alignment_counts"][j],4)
        self.assertEqual(interior["boundary_counts"][j],4)
        absent=model.score_components(VisibleQuery(5,"天地甲乙戊","玄黃戊"))
        np.testing.assert_array_equal(absent["alignment_bonus"],np.zeros(8))
        np.testing.assert_array_equal(absent["boundary_bonus"],np.zeros(8))

    def test_length_separated_counts_and_exact_training_provenance(self):
        poems=[fixture_poems()[0],fixture_poems()[2]]
        model=PhraseRetriever(poems,vocabulary_for(poems))
        resources=model.candidate_resources()
        five={row["text"]:row for row in resources["by_length"]["5"]}
        seven={row["text"]:row for row in resources["by_length"]["7"]}
        self.assertEqual(five["己庚"]["count"],4)
        self.assertEqual(seven["己庚"]["count"],8)
        self.assertEqual(five["己庚"]["source_poem_ids"],["five_a"])
        self.assertEqual(seven["己庚"]["source_poem_ids"],["seven"])
        self.assertEqual(five["己庚"]["narrow_count"],4)
        self.assertTrue(five["己庚"]["narrow_eligible"])
        self.assertFalse(five["庚辛"]["narrow_eligible"])
        components=model.score_components(VisibleQuery(7,"甲乙丙丁戊甲乙","辛壬癸己庚"))
        i=components["candidates"].index("己庚")
        self.assertEqual(components["alignment_counts"][i],8)
        self.assertEqual(components["boundary_counts"][i],4)
        audit=model.association_resources()
        self.assertIn({"context":"甲乙","span":"己庚","count":4},audit["aligned_span_pairs"]["5"])
        self.assertIn({"context":"甲乙","span":"己庚","count":8},audit["aligned_span_pairs"]["7"])

    def test_masked_target_or_metadata_never_enters_query_or_changes_rank(self):
        poems=fixture_poems()
        model=PhraseRetriever(poems,vocabulary_for(poems))
        task={"length":5,"upper_line":"甲乙丙丁戊","lower_suffix":"辛壬癸","start":0,"mask_length":2,
              "target":"己庚","author":"one","title":"one","poem_id":"one"}
        changed={**task,"target":"天地","author":"two","title":"two","poem_id":"two","lower_line":"天地辛壬癸"}
        one=model.rank(VisibleQuery.from_task(task),.5,1.0,limit=4)
        two=model.rank(VisibleQuery.from_task(changed),.5,1.0,limit=4)
        self.assertEqual(one["candidates"],two["candidates"])
        np.testing.assert_array_equal(one["scores"],two["scores"])
        self.assertEqual(one["query_hash"],two["query_hash"])
        self.assertEqual(one["candidate_set_hash"],two["candidate_set_hash"])
        with self.assertRaises(ValueError):
            model.score_components(task)
        class TargetQuery(VisibleQuery):
            target="己庚"
        with self.assertRaises(ValueError):
            model.score_components(TargetQuery(5,"甲乙丙丁戊","辛壬癸"))

    def test_nontraining_duplicate_or_malformed_fit_is_rejected(self):
        poems=fixture_poems()
        vocabulary=vocabulary_for(poems)
        for split in ("dev","test","confirm"):
            with self.subTest(split=split),self.assertRaises(ValueError):
                PhraseRetriever([{**poems[0],"split":split}],vocabulary)
        for bad in ([],[poems[0],poems[0]],[{**poems[0],"lines":["甲乙丙丁戊"]*7}],
                    [{**poems[0],"length":7}],[{**poems[0],"poem_id":None}]):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                PhraseRetriever(bad,vocabulary)
        with self.assertRaises(ValueError):
            PhraseRetriever(poems,{"甲":0})

    def test_constructor_snapshot_is_independent_of_caller_mutations_and_query_results_are_read_only(self):
        poems=fixture_poems()
        vocabulary=vocabulary_for(poems)
        model=PhraseRetriever(poems,vocabulary)
        query=VisibleQuery(5,"甲乙丙丁戊","辛壬癸")
        before=model.score_components(query)
        poems[0]["lines"][0]="天地玄黃宇"
        vocabulary["甲"]=0
        after=model.score_components(query)
        np.testing.assert_array_equal(before["ngram_scores"],after["ngram_scores"])
        self.assertEqual(before["candidates"],after["candidates"])
        for name in ("ngram_scores","alignment_bonus","boundary_bonus","narrow_mask","span_prior"):
            with self.assertRaises(ValueError):
                after[name][0]=0

    def test_rank_only_selects_source_spans_and_ties_ignore_input_order(self):
        components={"candidates":("己庚","甲乙","丙丁"),"ngram_scores":np.array([0.,0.,0.]),
                    "alignment_bonus":np.zeros(3),"boundary_bonus":np.zeros(3),
                    "narrow_mask":np.array([True,False,True]),"query_hash":"visible"}
        ranked=PhraseRetriever.rank_components(components,1,1)
        self.assertEqual(ranked["candidates"],tuple(sorted(components["candidates"])))
        narrow=PhraseRetriever.rank_components(components,pool="narrow")
        self.assertEqual(narrow["candidates"],tuple(sorted(("己庚","丙丁"))))
        self.assertNotIn("己乙",ranked["candidates"])
        limited=PhraseRetriever.rank_components(components,limit=1)
        self.assertEqual(limited["candidates"],(min(components["candidates"]),))

    def test_nonfinite_overflow_and_invalid_rank_arguments_are_rejected(self):
        components={"candidates":("甲乙","丙丁"),"ngram_scores":np.array([0.,1.]),
                    "alignment_bonus":np.array([1.,2.]),"boundary_bonus":np.zeros(2),
                    "narrow_mask":np.array([True,False])}
        for value in (float("nan"),float("inf"),-1,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                PhraseRetriever.rank_components(components,value,0)
        for kwargs in ({"limit":0},{"limit":True},{"pool":"test"}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                PhraseRetriever.rank_components(components,**kwargs)
        with self.assertRaises(ValueError):
            PhraseRetriever.rank_components({**components,"ngram_scores":np.array([0.,np.nan])})
        with self.assertRaises(ValueError):
            PhraseRetriever.rank_components({**components,"narrow_mask":np.array([1,0])})
        with np.errstate(over="ignore"),self.assertRaises(ValueError):
            PhraseRetriever.rank_components({**components,"alignment_bonus":np.array([1e308,1e308])},1e308,0)


if __name__=="__main__":
    unittest.main()
