import itertools
import math
import unittest
import numpy as np
from wordconv_lab.factor_dp import convolution_factor, path_score, solve_local_chain
from wordconv_lab.convolution import LocalConvScorer


class ExactFactorDPTests(unittest.TestCase):
    def test_conv_factorization_equals_original_for_all_tiny_candidates(self):
        model=LocalConvScorer(4,seed=29)
        def feature(row,col,token):
            return np.array([float(token=="A"),float(token=="B"),row,col/2])
        factor=convolution_factor(model,feature,3)
        domain=tuple(itertools.product(("A","B"),repeat=2))
        for path in itertools.product(domain,repeat=3):
            x=np.array([[feature(row,col,path[col][row]) for col in range(3)] for row in (0,1)])
            self.assertAlmostEqual(path_score(path,factor),model.score(x),places=12)

    def test_dp_matches_independent_exhaustive_maximum(self):
        rng=np.random.default_rng(52)
        domain=(("A","A"),("A","B"),("B","A"))
        domains=(domain,)*5
        values={}
        for j in range(5):
            for a,b,c in itertools.product((None,) if j==0 else domain,domain,(None,) if j==4 else domain):
                values[(j,a,b,c)]=float(rng.normal())
        factor=lambda *k: values[k]
        exact=max((path_score(p,factor),p) for p in itertools.product(*domains))
        result=solve_local_chain(domains,factor)
        self.assertAlmostEqual(result.score,exact[0],places=12)
        self.assertEqual(result.path,exact[1])
        self.assertLess(result.factor_calls,243*5)

    def test_empty_domain_and_infeasible_factor_abstain(self):
        self.assertEqual(solve_local_chain(((),),lambda *x: 0).status,"infeasible")
        result=solve_local_chain(((("A","A"),),)*3,lambda *x: -math.inf)
        self.assertEqual(result.status,"infeasible")
        self.assertIsNone(result.path)

    def test_nan_or_positive_infinity_are_not_legitimate_scores(self):
        for x in (math.nan,math.inf):
            with self.assertRaises(ValueError):
                solve_local_chain(((("A","A"),),),lambda *args:x)

    def test_positive_accumulation_overflow_is_rejected(self):
        for length in (2,3):
            domains=((("A","A"),),)*length
            with self.assertRaisesRegex(ValueError,"overflowed"):
                solve_local_chain(domains,lambda *args:1e308)
            with self.assertRaisesRegex(ValueError,"overflowed"):
                path_score((("A","A"),)*length,lambda *args:1e308)

    def test_negative_accumulation_overflow_is_not_infeasible(self):
        for length in (2,3):
            with self.assertRaisesRegex(ValueError,"overflowed"):
                solve_local_chain(((("A","A"),),)*length,lambda *args:-1e308)

    def test_boundary_cases_and_deterministic_ties(self):
        domain=(("B","B"),("A","A"))
        for length in (1,2,3):
            result=solve_local_chain((domain,)*length,lambda *args:0)
            self.assertEqual(result.path,(("A","A"),)*length)

    def test_cannot_silently_factorize_other_widths(self):
        with self.assertRaises(ValueError):
            convolution_factor(LocalConvScorer(2,kernel_width=5),lambda *x:np.ones(2),5)
