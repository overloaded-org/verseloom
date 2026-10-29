import unittest
import numpy as np
from wordconv_lab.restoration import VisibleQuery,BidirNGram,encode_candidates,top_candidate,family_metric,paired_family_interval


class RestorationBoundaryTests(unittest.TestCase):
    def setup_model(self):
        chars=['<UNK>']+sorted(set('甲乙丙丁戊己庚辛壬癸'))
        vocab={c:i for i,c in enumerate(chars)}
        poems=[dict(poem_id='a',split='train',lines=['甲乙丙丁戊','己庚辛壬癸']*4)]
        return vocab,BidirNGram(vocab).fit(poems)

    def test_hidden_target_and_metadata_cannot_change_pool_or_features(self):
        vocab,model=self.setup_model()
        task=dict(length=5,upper_line='甲乙丙丁戊',lower_suffix='辛壬癸',start=0,mask_length=2,
                  target='己庚',title='secret',author='secret')
        changed=dict(task,target='甲乙',title='different',author='different',lower_line='甲乙辛壬癸')
        a,b=VisibleQuery.from_task(task),VisibleQuery.from_task(changed)
        pool_a=model.propose(a,['己庚','甲乙','丙丁'])
        pool_b=model.propose(b,['己庚','甲乙','丙丁'])
        self.assertEqual(pool_a,pool_b)
        np.testing.assert_array_equal(encode_candidates(a,pool_a['candidates'],vocab),
                                      encode_candidates(b,pool_b['candidates'],vocab))

    def test_nontraining_poems_cannot_fit_statistics(self):
        vocab,_=self.setup_model()
        with self.assertRaises(ValueError):
            BidirNGram(vocab).fit([dict(poem_id='test',split='test',lines=['甲乙丙丁戊']*8)])

    def test_same_visible_query_has_same_pool_and_no_target_injection(self):
        _,model=self.setup_model()
        query=VisibleQuery(5,'甲乙丙丁戊','辛壬癸')
        pool=model.propose(query,['甲乙','丙丁'],limit=16)
        self.assertNotIn('己庚',pool['candidates'])
        with self.assertRaises(ValueError):
            model.propose({'target':'己庚'},['甲乙'])

    def test_shared_lexical_tie_rule_ignores_retriever_order(self):
        self.assertEqual(top_candidate(['乙乙','甲甲'],[1,1]),1 if '甲甲'<'乙乙' else 0)
        choices=['己庚','甲乙','丙丁']
        winner=choices[top_candidate(choices,[0,0,0])]
        self.assertEqual(winner,min(choices))

    def test_metric_and_resampling_give_each_family_equal_weight(self):
        tasks=[{'family_id':'large'}]*100+[{'family_id':'small'}]
        score=[1.]*100+[0.]
        metric=family_metric(tasks,score)
        self.assertAlmostEqual(metric['family_macro'],.5)
        self.assertGreater(metric['task_micro'],.99)
        result=paired_family_interval(tasks,score,[0.]*101,repeats=100)
        self.assertAlmostEqual(result['mean_difference'],.5)

    def test_counter_model_predicts_an_observed_training_completion(self):
        _,model=self.setup_model()
        query=VisibleQuery(5,'甲乙丙丁戊','辛壬癸')
        self.assertGreater(model.score(query,'己庚'),model.score(query,'甲乙'))
