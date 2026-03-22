import unittest

from aidetect.data import build_experiment_data
from aidetect.features import TfidfVectorizer, tokenize
from aidetect.paraphrase import paraphrase_text
from aidetect.pipeline import run_baseline_experiment, run_full_experiment


class PipelineTests(unittest.TestCase):
    def test_tokenize_lowercases_words(self) -> None:
        self.assertEqual(tokenize("AI-Written Text, indeed!"), ["ai", "written", "text", "indeed"])

    def test_experiment_split_has_examples(self) -> None:
        experiment = build_experiment_data()
        self.assertGreater(len(experiment.train_texts), 0)
        self.assertGreater(len(experiment.clean_test.texts), 0)
        self.assertIn("moderate", experiment.paraphrased_tests)
        self.assertGreater(len(experiment.paraphrased_tests["moderate"].texts), 0)

    def test_vectorizer_creates_dense_matrix(self) -> None:
        vectorizer = TfidfVectorizer.fit(["alpha beta", "beta gamma"])
        matrix = vectorizer.transform(["alpha gamma", "beta"])
        self.assertEqual(matrix.shape[0], 2)
        self.assertGreater(matrix.shape[1], 0)

    def test_paraphrase_levels_change_text(self) -> None:
        source = "This system provides effective support for students."
        self.assertNotEqual(source, paraphrase_text(source, "light"))
        self.assertNotEqual(source, paraphrase_text(source, "moderate"))
        self.assertNotEqual(source, paraphrase_text(source, "heavy"))

    def test_pipeline_returns_metrics_sections(self) -> None:
        metrics = run_baseline_experiment()
        self.assertIn("clean", metrics)
        self.assertIn("paraphrased", metrics)
        self.assertIn("degradation", metrics)

    def test_full_experiment_returns_model_comparison(self) -> None:
        metrics = run_full_experiment()
        self.assertIn("models", metrics)
        self.assertIn("tfidf_logreg", metrics["models"])
        self.assertIn("embedding_avg_nn", metrics["models"])


if __name__ == "__main__":
    unittest.main()
