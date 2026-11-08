import unittest
from retrieval import tokenize, build_index, retrieve, metrics, evaluate

class RetrievalTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize('Hello, WORLD!'), ['hello', 'world'])
    def test_retrieve(self):
        idx = build_index(['red apples grow', 'blue oceans deep'])
        self.assertEqual(retrieve('blue ocean deep', idx, 1), [1])
    def test_hit_mrr(self):
        m = metrics([1, 2, None], 2)
        self.assertAlmostEqual(m['hit_at_k'], 2/3)
        self.assertAlmostEqual(m['mrr_at_k'], .5)
    def test_empty(self):
        self.assertEqual(metrics([], 5)['questions'], 0)
    def test_evaluate(self):
        r = evaluate(['cats purr softly', 'dogs bark loudly'], [('cats purr', 0), ('dogs bark', 1)], 1)
        self.assertEqual(r['hit_at_k'], 1.0)

if __name__ == '__main__':
    unittest.main()
