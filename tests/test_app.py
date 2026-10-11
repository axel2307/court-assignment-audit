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
        self.assertEqual(app.subheader[0].value, "Court A's count under the stated 40% rule")
        self.assertIn("Show the Court A calculation", [item.label for item in app.expander])
        self.assertIn(
            "does not show a clear departure", " ".join(item.value for item in app.markdown)
        )
        self.assertEqual(sum(app.dataframe[0].value["Observed"]), 1000)
        self.assertEqual(len(app.get("download_button")), 2)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertIn("100% SYNTHETIC DATA", " ".join(item.value for item in app.caption))
        self.assertEqual(
            [tab.label for tab in app.tabs], ["Analysis", "How to read this", "Data & run log"]
        )
        self.assertIn("Show the global test details", [item.label for item in app.expander])

    def test_scenario_switch_keeps_baseline_and_changes_observations(self):
        app = self.open_app()
        before = app.dataframe[0].value.copy()
        app.selectbox(key="scenario").select("Illustrative anomaly").run()
        self.assertEqual(len(app.exception), 0)
        after = app.dataframe[0].value
        self.assertEqual(before["Expected"].tolist(), after["Expected"].tolist())
        self.assertNotEqual(before["Observed"].tolist(), after["Observed"].tolist())
        self.assertGreater(after.iloc[0]["Observed"], before.iloc[0]["Observed"])
        text = " ".join(item.value for item in app.markdown)
        self.assertIn("is unusual under the specified random-assignment model", text)
        self.assertIn("Zero exceedances in a finite simulation does not mean", text)
        global_metric = next(
            item for item in app.metric if item.label == "Global Monte Carlo p-value"
        )
        self.assertGreater(float(global_metric.value), 0)

    def test_sparse_case_and_seed_round_trip(self):
        app = self.open_app()
        app.number_input(key="n_cases").set_value(1).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(sum(app.dataframe[0].value["Observed"]), 1)
        self.assertEqual(len(app.dataframe[0].value), 4)
        self.assertIn(
            "One assignment cannot establish a stable allocation pattern",
            " ".join(item.value for item in app.markdown),
        )
        first = app.dataframe[0].value["Observed"].tolist()
        app.number_input(key="seed").set_value(7).run()
        app.number_input(key="seed").set_value(20261011).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(first, app.dataframe[0].value["Observed"].tolist())

    def test_largest_supported_run_keeps_count_and_test_roles(self):
        app = self.open_app()
        app.number_input(key="n_cases").set_value(5000).run()
        app.selectbox(key="scenario").select("Illustrative anomaly").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(sum(app.dataframe[0].value["Observed"]), 5000)
        self.assertEqual(len(app.dataframe[1].value), 4)
        self.assertIn("Holm-adjusted p-value", app.dataframe[1].value.columns)
        self.assertEqual(len(app.get("download_button")), 2)


if __name__ == "__main__":
    unittest.main()
