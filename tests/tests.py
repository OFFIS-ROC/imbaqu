import unittest
import pandas as pd
from pandas.testing import assert_series_equal
import numpy as np
import imbaqu

class Test_discrete_series(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_discrete_series, self).__init__(*args, **kwargs)
        self.data = pd.Series([1,2,2,3,3,3,4,4,5,5, np.nan])

        self.lamb = imbaqu.probability_ratio(self.data, discrete= True, drop_inf_nan=True)
        self.true_lamb = pd.Series([0.5, 1.0, 1.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0])

        self.ir = imbaqu.imbalance_ratio(self.lamb)
        self.true_ir = pd.Series([2, 1.0, 1.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0])

        self.mir = imbaqu.mean_imbalance_ratio(self.data, discrete= True, drop_inf_nan= True)
        self.true_mir = 1.25
    
    def test_probability_ratio(self):
        try:
            assert_series_equal(self.lamb, self.true_lamb)
        except:
            self.fail("The probability ratio 'lamb' does not equal 'lamb_true'.")
        
    def test_imbalance_ratio(self):
        try:
            assert_series_equal(self.ir, self.true_ir)
        except:
            self.fail("The imbalance ratio 'ir' does not equal 'ir_true'.")

    def test_mean_imbalance_ratio(self):
        self.assertEqual(self.mir, self.true_mir, "The mean imbalance rate 'mir' is not correct.")
    
    def test_imbalanced_sample_percentage(self):
        with self.assertRaises(ValueError, msg= "ValueError should be raised as no isp bounds are given."):
            imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= True, drop_inf_nan= True)
        isp = imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= True, drop_inf_nan= True, ir_bound= 1)
        self.assertEqual(isp, 40.0, "The imbalanced sample percentage 'isp' using in 'ir_bound' is not correct.")
        isp = imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= True, drop_inf_nan= True, lower_bound= 1)
        self.assertEqual(isp, 10.0, "The imbalanced sample percentage 'isp' using in 'lower_bound' is not correct.")
        isp = imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= True, drop_inf_nan= True, upper_bound= 1)
        self.assertEqual(isp, 30.0, "The imbalanced sample percentage 'isp' using in 'upper_bound' is not correct.")


class Test_continuous_series(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_continuous_series, self).__init__(*args, **kwargs)
        rng = np.random.default_rng(seed=42)
        self.data = pd.Series(rng.normal(size=1000))

    def test_mean_imbalance_ratio(self):
        mir = imbaqu.mean_imbalance_ratio(self.data, relevance_pdf= None, discrete= False, drop_inf_nan= True)
        self.assertEqual(round(mir,3), 2.415, "The mean imbalance rate 'mir' is not correct.")

    def test_imbalanced_sample_percentage(self):
        with self.assertRaises(ValueError, msg= "ValueError should be raised as no isp bounds are given."):
            imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= False, drop_inf_nan= True)
        isp = imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= False, drop_inf_nan= True, ir_bound= 2)
        self.assertEqual(isp, 58.4, "The imbalanced sample percentage 'isp' using in 'ir_bound' is not correct.")
        isp = imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= False, drop_inf_nan= True, lower_bound= 2)
        self.assertEqual(isp, 47.0, "The imbalanced sample percentage 'isp' using in 'lower_bound' is not correct.")
        isp = imbaqu.imbalanced_sample_percentage(self.data, relevance_pdf= None, discrete= False, drop_inf_nan= True, upper_bound= 2)
        self.assertEqual(isp, 53.0, "The imbalanced sample percentage 'isp' using in 'upper_bound' is not correct.")



if __name__ == '__main__':
    unittest.main()