import os
import unittest
from collections import Counter

import pandas as pd

from scripts.evaluate_diversity import alpha_ndcg, calculateNormalizedDiscountedKLDivergence
from scripts.evaluate_relevance import precision_at_k, ndcg
from scripts.evaluation import get_kl_divergence, evaluate_relevance


class TestAlphaNDCG(unittest.TestCase):
    def setUp(self):
        dir_test_data = "test_data"
        self.data_path = dir_test_data
        self.ground_truth_df = pd.read_json(os.path.join(self.data_path, "queries_dev.jsonl"), lines=True,
                                            orient="records")
        self.prediction_df = pd.read_json(os.path.join(self.data_path, "ground_truth_small_mutations.jsonl"),
                                          lines=True,
                                          orient="records")
        self.corpus = pd.read_json(os.path.join(self.data_path, "corpus.jsonl"), lines=True, orient="records")
        # sort corpus df by argument_id
        self.corpus = self.corpus.sort_values("argument_id")
        # set index to argument_id
        self.corpus = self.corpus.set_index("argument_id")


    def test_alpha_ndcg_example(self):
        """
        First, we calculate the relevance score product for each item (considering alpha and redundancy):
        item 1: 1 * 1 = 1 because it is relevant and there is no redundancy
        item 2: 0 because the relevance score is 0 so the product is 0
        item 3: 1 * (1-alpha) = 1 * 0.5 = 0.5 because it is relevant and there is redundancy.
        item 4: 1 * (1-alpha) = 1 * 0.5 = 0.5 because it is relevant and there is redundancy.
        item 5: 0 because the relevance score is 0 so the product is 0.
        Next we take the ranking into account, we want to weigh each score the lower the rank is so we divide
        each score by log2(rank + 1):
        1/(log2(1+1)) + 0/(log2(2+1)) + 0.5/(log2(3+1)) + 0.5/(log2(4+1)) + 0/(log2(5+1)) = 1.715
        For the ideal ranking we would just move the second item to the end. This would result in the following
        relevance score product:
        item 1: 1 * 1 = 1 because it is relevant and there is no redundancy
        item 2: 1*1 = 1 because it is relevant and there is no redundancy
        item 3: 1 * (1-alpha) = 1 * 0.5 = 0.5 because it is relevant and there is redundancy.
        item 4 and 5 are zero.
        the sum would then be:
        1/(log2(1+1)) + 1/(log2(2+1)) + 0.5/(log2(3+1)) + 0/(log2(4+1)) + 0/(log2(5+1)) = 1.881
        Finally we would divide the two scores and get 1.715/1.881 = ~0.911
        :return:
        """
        relevance_scores = [1, 0, 1, 1, 0]
        perspectives = ['A', 'A', 'B', 'A', 'B']
        ground_truth_relevance = [1, 1, 1, 0, 0]
        ground_truth_perspectives = ['A', 'A', 'B', 'B', 'A']
        alpha = 0.5
        alpha_ndcg_score = \
            alpha_ndcg(perspectives_global=ground_truth_perspectives, perspectives_predictions=perspectives,
                       relevance_scores_global=ground_truth_relevance, relevance_scores_predictions=relevance_scores,
                       alpha=alpha, k_range=[5])[5]
        expected_outcome = 0.911
        self.assertAlmostEqual(alpha_ndcg_score, expected_outcome, places=2)

    def test_normalized_kl_divergence(self):
        # Test data setup
        ranked_perspectives = ["A", "A", "B", "B", "C", "C"]
        gold_distribution = {"A": 0.33, "B": 0.33, "C": 0.33}
        cut_off_points = [2, 4, 6]
        k = 6

        # Expected value setup (this should be computed based on the expected behavior of the function)
        expected_rkl_value = 0.2424

        # Calculate normalized KL-divergence
        calculated_rkl_value = \
            calculateNormalizedDiscountedKLDivergence(ranked_perspectives=ranked_perspectives, gold_propotion=0.33,
                                                      protected_group="A", cut_off_points=cut_off_points, k=k)

        # Assert equality
        self.assertAlmostEqual(calculated_rkl_value, expected_rkl_value, places=2,
                               msg="The calculated rKL value does not match the expected value.")

    def test_perfect_kl_divergence(self):
        # create a ranked list that has a fair distribution of each group
        ranked_perspectives = ["A", "A", "B", "B", "C", "C"]
        gold_distribution = {"A": 0.33, "B": 0.33, "C": 0.33}
        cut_off_points = [6]
        k = 6
        for category, gold_proportion in gold_distribution.items():
            calculated_rkl_value = calculateNormalizedDiscountedKLDivergence(ranked_perspectives, gold_proportion,
                                                                             category,
                                                                             cut_off_points, k)
            self.assertAlmostEqual(calculated_rkl_value, 0.0, places=4,
                                   msg="The calculated rKL value does not match the expected value.")

    def test_example_scores(self):
        """
        Mini example to better understand the scores and how the ordering of the arguments matter (beyond pure relevance).
        Let's assume we have a query q = Are you in favor of the introduction of a tax on foods containing sugar (sugar tax)?,[age: < 36]",
        so we are interested in all matching arguments that are relevant to the question of the query and that were written by people
        below the age of 36.
        Let's assume we have the following sub-corpus c that matches the question of the query. c =
        [{id: 12, age: <36, political_spectrum: “Mitte und Liberal”},
        {id: 43, age: <36, political_spectrum: “Mitte und Liberal”},
        {id: 23, age: <36, political_spectrum: “Rechts und Liberal”},
        {id: 10, age: <36, political_spectrum: “Links und Konservativ”},
        {id: 17, age: <36, political_spectrum: “Rechts und Konservativ”},
        {id: 5, age: <36, political_spectrum: “Mitte und Liberal”}]
        {id: 25, age: 43, political_spectrum: “Rechts und Konservativ”},
        {id: 14, age: 65, political_spectrum: “Mitte und Liberal”}]
        We want to compare the scores for three different models. The predictions (top-7) of the models are the following:
        model a: [12, 25, 43, 23, 10, 17, 5]
        model b: [43, 10, 5, 17, 12, 23, 14]
        model c: [23, 12, 5, 25, 23, 17, 10]
        """
        # we estimate the relevance of each argument to the query, some arguments were relevant to the question but
        # the age of the author was not below 36, so we set the relevance score to 0.
        # model a has a non-relevant argument at rank 2.
        relevance_model_a = [1, 0, 1, 1, 1, 1, 1]
        # model b has correctly listed all relevant arguments at the top
        relevance_model_b = [1, 1, 1, 1, 1, 1, 0]
        # model c has a non-relevant argument at rank 4.
        relevance_model_c = [1, 1, 1, 0, 1, 1, 1, 1]
        # 8 arguments in total, 6 relevant, 2 non-relevant
        relevance_scores_global = [1, 1, 1, 1, 1, 1, 0, 0]
        # we first want to look at the scores for argument relevance.
        precision_at_5_model_a = precision_at_k(relevance_model_a, k=5)
        precision_at_5_model_b = precision_at_k(relevance_model_b, k=5)
        precision_at_5_model_c = precision_at_k(relevance_model_c, k=5)
        # precision at rank does not take the ordering into account, so model a and model b should have the same score,
        # model b should have the highest precision score
        self.assertEqual(precision_at_5_model_a, precision_at_5_model_c)
        self.assertGreater(precision_at_5_model_b, precision_at_5_model_c)
        self.assertGreater(precision_at_5_model_b, precision_at_5_model_a)
        # However, if we look at nDCG, we see that model a has a lower score than model c because the non-relevant
        # item is at rank 2 (higher ranked items are more important when computing this score).
        ndcg_model_a = ndcg(relevance_scores_predictions=relevance_model_a,
                            relevance_scores_global=relevance_scores_global, k_range=[5])[5]
        ndcg_model_b = ndcg(relevance_scores_predictions=relevance_model_b, relevance_scores_global=relevance_model_b,
                            k_range=[5])[5]
        ndcg_model_c = ndcg(relevance_scores_predictions=relevance_model_c, relevance_scores_global=relevance_model_c,
                            k_range=[5])[5]
        # so for this score the ordering of the arguments matters and model a should have a lower score than model c.
        self.assertGreater(ndcg_model_c, ndcg_model_a)
        self.assertGreater(ndcg_model_b, ndcg_model_c)
        self.assertGreater(ndcg_model_b, ndcg_model_a)
        # next we are interested in the score that take diversity into account. We have a global distribution of the
        # political spectrum of the arguments. We want to have a look at the score if we are interested in diversifying
        # the political spectrum of the authors of the arguments.
        # what are the political perspectives of the authors of all arguments?
        perspectives_global = ["Mitte und Liberal", "Mitte und Liberal", "Rechts und Liberal", "Links und Konservativ",
                               "Rechts und Konservativ", "Mitte und Liberal", "Rechts und Konservativ",
                               "Mitte und Liberal"]
        # what is the relative amount of each perspective in the global distribution?
        counter = Counter(perspectives_global)
        gold_distribution = {k: v / len(perspectives_global) for k, v in counter.items()}
        # what is the value for political spectrum for the predictions of model a?
        perspectives_model_a = ["Mitte und Liberal", "Rechts und Konservativ", "Mitte und Liberal",
                                "Rechts und Liberal",
                                "Links und Konservativ", "Rechts und Konservativ", "Mitte und Liberal"]
        # what is the value for political spectrum for the predictions of model b?
        perspectives_model_b = ["Mitte und Liberal", "Links und Konservativ", "Mitte und Liberal",
                                "Rechts und Konservativ",
                                "Mitte und Liberal", "Rechts und Liberal", "Mitte und Liberal"]
        # what is the value for political spectrum for the predictions of model c?
        perspectives_model_c = ["Rechts und Liberal", "Mitte und Liberal", "Mitte und Liberal",
                                "Rechts und Konservativ",
                                "Rechts und Liberal", "Rechts und Konservativ", "Links und Konservativ"]
        # we can now calculate the alpha-nDCG score for each model.
        alpha_ndcg_a = alpha_ndcg(relevance_scores_global=relevance_scores_global,
                                  perspectives_global=perspectives_global,
                                  relevance_scores_predictions=relevance_model_a,
                                  perspectives_predictions=perspectives_model_a,
                                  alpha=0.5, k_range=[5])[5]
        alpha_ndcg_b = alpha_ndcg(relevance_scores_global=relevance_scores_global,
                                  perspectives_global=perspectives_global,
                                  relevance_scores_predictions=relevance_model_b,
                                  perspectives_predictions=perspectives_model_b,
                                  alpha=0.5, k_range=[5])[5]
        alpha_ndcg_c = alpha_ndcg(relevance_scores_global=relevance_scores_global,
                                  perspectives_global=perspectives_global,
                                  relevance_scores_predictions=relevance_model_c,
                                  perspectives_predictions=perspectives_model_c,
                                  alpha=0.5, k_range=[5])[5]
        # the alpha-nDCG score should be highest for model b but it should be lower than the nDCG score for model b, because
        # it does not represent all different values for political attitude in the top-5.
        self.assertGreater(ndcg_model_b, alpha_ndcg_b)
        self.assertGreater(alpha_ndcg_b, alpha_ndcg_a)
        self.assertGreater(alpha_ndcg_b, alpha_ndcg_c)
        # the score for model c should be higher than the one for model a
        self.assertGreater(alpha_ndcg_c, alpha_ndcg_a)
        # but the distance between alpha_ndcg_c and alpha_ndcg_a should be smaller than the distance between
        # ndcg_model_c and ndcg_model_a
        self.assertGreater(ndcg_model_c - ndcg_model_a, alpha_ndcg_c - alpha_ndcg_a)
        # this is because model a has a more diverse set of political attitudes in the top-5 than model c.

        # finally we can compute the kullback-leibler divergence for the top-5 arguments of each model.
        rkl_a = get_kl_divergence(ranked_perspectives=perspectives_model_a, gold_distribution=gold_distribution,
                                  k=5, cutoff_points=[5])
        rkl_b = get_kl_divergence(ranked_perspectives=perspectives_model_b, gold_distribution=gold_distribution,
                                  k=5, cutoff_points=[5])
        rkl_c = get_kl_divergence(ranked_perspectives=perspectives_model_c, gold_distribution=gold_distribution,
                                  k=5, cutoff_points=[5])
        # since this does not take relevance into account the model wins that best represents the global distribution
        # for each perspective.
        # model a should have the best representation of each proportion (the lower the rkl the better)
        self.assertGreater(rkl_c, rkl_a)
        self.assertGreater(rkl_b, rkl_a)

    def test_duplicates_implicit_perspective_scenario(self):
        predictions_muted = self.prediction_df.sort_values("query_id")
        ground_truth = self.ground_truth_df.sort_values("query_id")
        # assert that the lists "predictions" and ground_truth are exactly the same (same values in the same order)
        assert list(predictions_muted["query_id"]) == list(ground_truth["query_id"])
        # assert the length are the same
        assert len(predictions_muted) == len(ground_truth)

        results_perspective = evaluate_relevance(predictions_df=predictions_muted,
                                                  ground_truth_df=ground_truth, output_dir=self.data_path,
                                                  corpus=self.corpus, implicit=True)
        all_ndcg_scores_perspective = results_perspective["ndcg@k"].values
        results_other = evaluate_relevance(predictions_df=predictions_muted, ground_truth_df=ground_truth,
                                           output_dir=self.data_path, corpus=self.corpus, implicit=False)
        all_ndcg_scores_other = results_other["ndcg@k"].values
        # all nDCG scores for the perspective scenario should be higher than the nDCG scores for the other scenario
        for i in range(len(all_ndcg_scores_perspective)):
            self.assertGreater(all_ndcg_scores_perspective[i], all_ndcg_scores_other[i])
        # all nDCG scores for the perspective scenario should be 1
        self.assertTrue(all([score == 1 for score in all_ndcg_scores_perspective]))


if __name__ == '__main__':
    unittest.main()
