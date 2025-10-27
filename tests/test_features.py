import unittest
import pandas as pd

from src.features import build_lags


class TestFeatures(unittest.TestCase):
	def test_build_lags(self):
		df = pd.DataFrame({
			"country": ["A", "A", "A", "B", "B"],
			"sector": ["X", "X", "X", "Y", "Y"],
			"year": [2001, 2002, 2003, 2001, 2002],
			"P": [10, 20, 30, 5, 6],
			"Q": [1.0, 1.5, 2.0, 0.5, 0.6]
		})
		out = build_lags(df, vars=["P", "Q"], lags=2, group=["country", "sector"])
		row = out[(out["country"]=="A") & (out["sector"]=="X") & (out["year"]==2003)].iloc[0]
		self.assertEqual(row["P_lag1"], 20)
		self.assertEqual(row["P_lag2"], 10)
		self.assertAlmostEqual(row["Q_lag1"], 1.5)
		self.assertAlmostEqual(row["Q_lag2"], 1.0)


if __name__ == "__main__":
	unittest.main()
