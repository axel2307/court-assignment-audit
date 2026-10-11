import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppFlowTests(unittest.TestCase):
    @staticmethod
    def open_app():
        app = Path(__file__).resolve().parents[1] / "app.py"
        return AppTest.from_file(str(app), default_timeout=30).run()

    def test_default_run_has_results_and_synthetic_context(self):
        app = self.open_app()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Court Assignment Audit")
        self.assertEqual(app.metric[0].value, "1,000")
        self.assertEqual(len(app.get("download_button")), 2)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertIn("invented cases", app.info[0].value)

    def test_scenario_switch_keeps_baseline_and_changes_observations(self):
        app = self.open_app()
        before = app.dataframe[0].value.copy()
        app.selectbox(key="scenario").select("Illustrative anomaly").run()
        self.assertEqual(len(app.exception), 0)
        after = app.dataframe[0].value
        self.assertEqual(before["Expected"].tolist(), after["Expected"].tolist())
        self.assertNotEqual(before["Observed"].tolist(), after["Observed"].tolist())
        self.assertGreater(after.iloc[0]["Observed"], before.iloc[0]["Observed"])

    def test_sparse_case_and_seed_round_trip(self):
        app = self.open_app()
        app.number_input(key="n_cases").set_value(1).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(sum(app.dataframe[0].value["Observed"]), 1)
        self.assertEqual(len(app.dataframe[0].value), 4)
        first = app.dataframe[0].value["Observed"].tolist()
        app.number_input(key="seed").set_value(7).run()
        app.number_input(key="seed").set_value(20261011).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(first, app.dataframe[0].value["Observed"].tolist())


if __name__ == "__main__":
    unittest.main()
