import unittest

from src.schema import fuzzy_detect


class TestSchemaDetection(unittest.TestCase):
	def test_fuzzy_detect(self):
		headers = [
			"Publication_Number",
			"Applicant_Country",
			"Y02_Sector",
			"Pub_Year",
			"Quality_Index",
		]
		m = fuzzy_detect(headers)
		self.assertIn("country", m)
		self.assertIn("sector", m)
		self.assertIn("year", m)
		self.assertIn("patent_id", m)
		self.assertIn("quality", m)


if __name__ == "__main__":
	unittest.main()
