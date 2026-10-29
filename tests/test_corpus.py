from copy import deepcopy
import unittest

from wordconv_lab.corpus import (MASK, assert_split_integrity, assign_families_and_splits,
    collect_poems, family_edges, make_tasks, model_view, source_lines, train_resources,
    validate_tasks_and_resources)


def poem(number=0,length=5):
    return {"poem_id":f"p{number}", "lines":["".join(chr(0x4e00+number*100+i*length+j)
                for j in range(length)) for i in range(8)], "length":length,
            "author":"合成作者", "title":"工程测试", "source_refs":[],
            "family_id":f"f{number}", "split":"train"}


class CorpusIntegrityTests(unittest.TestCase):
    def test_source_shape_is_not_a_lushi_certificate_and_text_is_exact(self):
        p=poem()
        record={"paragraphs":[p['lines'][i]+'，'+p['lines'][i+1]+'。' for i in (0,2,4,6)],
                "author":"甲","title":"题","id":"id"}
        poems,duplicates,counts=collect_poems([('source-a',[record,deepcopy(record)])])
        self.assertEqual(len(poems),1)
        self.assertEqual(poems[0]['lines'],p['lines'])
        self.assertEqual(poems[0]['status'],'shape_only_not_verified_lushi')
        self.assertEqual(len(poems[0]['source_refs']),2)
        self.assertEqual(counts['exact_duplicate_excess_records'],1)
        self.assertEqual(len(duplicates),1)
        record['paragraphs'][0]='A'+record['paragraphs'][0][1:]
        self.assertEqual(source_lines(record)[1],'non_cjk_body')

    def test_shared_line_connects_families_before_split(self):
        a,b=poem(0),poem(1)
        b['lines'][3]=a['lines'][2]
        edges=family_edges([a,b])
        self.assertEqual(len(edges),1)
        self.assertEqual(edges[0]['shared_lines'],[a['lines'][2]])
        grouped=assign_families_and_splits([a,b],edges)
        self.assertEqual(grouped[0]['family_id'],grouped[1]['family_id'])
        self.assertEqual(grouped[0]['split'],grouped[1]['split'])
        assert_split_integrity(grouped)

    def test_shared_line_cannot_hide_behind_fake_different_family_ids(self):
        a,b=poem(0),poem(1)
        b['lines'][0]=a['lines'][0]
        b['split']='test'
        with self.assertRaisesRegex(ValueError,'crosses family/split'):
            assert_split_integrity([a,b])

    def test_near_duplicate_without_shared_line_cannot_cross_split(self):
        alphabet=''.join(chr(0x4f00+i) for i in range(16))*2
        a,b=poem(0,7),poem(1,7)
        a['lines']=[alphabet[i:i+7] for i in range(0,16,2)]
        b['lines']=[alphabet[i:i+7] for i in range(1,16,2)]
        b['split']='test'
        edges=family_edges([a,b])
        self.assertEqual(edges[0]['shared_lines'],[])
        self.assertEqual(edges[0]['bigram_jaccard']['value'],1.0)
        with self.assertRaisesRegex(ValueError,'crosses family/split'):
            assert_split_integrity([a,b])

    def test_transitive_family_and_hash_split_are_order_independent(self):
        a,b,c=poem(0),poem(1),poem(2)
        b['lines'][0]=a['lines'][0]
        c['lines'][3]=b['lines'][2]
        forward=assign_families_and_splits([a,b,c],family_edges([a,b,c]))
        reverse=assign_families_and_splits([c,b,a],family_edges([c,b,a]))
        self.assertEqual({x['poem_id']:(x['family_id'],x['split']) for x in forward},
                         {x['poem_id']:(x['family_id'],x['split']) for x in reverse})
        self.assertEqual(len({x['family_id'] for x in forward}),1)

    def test_masked_model_view_excludes_gold_and_identifiers(self):
        p=poem()
        tasks=make_tasks([p])
        self.assertEqual(len(tasks),4)
        self.assertEqual(tasks[0]['masked_lower_line'],MASK+p['lines'][1][2:])
        self.assertEqual(tasks[0]['target'],p['lines'][1][:2])
        view=model_view(tasks[0])
        self.assertEqual(set(view),{'length','upper_line','masked_lower_line','lower_suffix','start','mask_length'})
        self.assertNotIn('target',view)
        self.assertNotIn('title',view)
        self.assertNotIn('author',view)

    def test_train_only_vocab_has_explicit_unk_zero_and_keeps_oov_tasks(self):
        a,b=poem(0),poem(1)
        # Repeated training prefix becomes a two-occurrence candidate.
        a['lines'][3]=a['lines'][1][:2]+a['lines'][3][2:]
        b['split']='test'
        tasks=make_tasks([a,b])
        vocab,candidates=train_resources([a,b],tasks)
        self.assertEqual(vocab['unknown_id'],0)
        self.assertEqual(vocab['tokens'][0],'<UNK>')
        self.assertEqual(vocab['character_to_id'][vocab['characters'][0]],1)
        self.assertEqual(len(tasks),8)
        self.assertEqual([x['text'] for x in candidates['by_length']['5']],[a['lines'][1][:2]])
        self.assertFalse(any(t['target'] in {x['text'] for x in candidates['by_length']['5']}
                             for t in tasks if t['split']=='test'))
        self.assertTrue(all(x['source_poem_ids']==['p0'] for x in candidates['by_length']['5']))
        validate_tasks_and_resources([a,b],tasks,vocab,candidates)

    def test_injected_test_candidate_is_rejected(self):
        a,b=poem(0),poem(1)
        b['split']='test'
        tasks=make_tasks([a,b])
        vocab,candidates=train_resources([a,b],tasks)
        candidates['by_length']['5'].append({'text':tasks[-1]['target'],'count':2,'source_poem_ids':['p1']})
        with self.assertRaisesRegex(ValueError,'train-only'):
            validate_tasks_and_resources([a,b],tasks,vocab,candidates)

    def test_unmasked_lower_line_is_rejected(self):
        p=poem()
        tasks=make_tasks([p])
        vocab,candidates=train_resources([p],tasks)
        tasks[0]['masked_lower_line']=p['lines'][1]
        with self.assertRaisesRegex(ValueError,'mask/split'):
            validate_tasks_and_resources([p],tasks,vocab,candidates)


if __name__=='__main__':
    unittest.main()
