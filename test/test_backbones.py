"""File with all the different tests we run."""
import logging
import os
import unittest
from pathlib import Path

from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from main import run

TEST_SEED = 473284789


class TestBackbones(unittest.TestCase):
    """The backbone testing case."""

    def setUp(self):
        """Set up some testing environment variables."""
        os.environ["FAST_DEV_RUN"] = "True"
        os.environ["WANDB_MODE"] = "disabled"
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        """Remove all files, created during testing."""
        saved_test_models = Path("/app/saved_models/").glob(f"*{TEST_SEED}*")
        for file in saved_test_models:
            file.unlink()

    def run_with_config(self, backbones, preprocessings, validations):
        """Run the pipeline with the specified backbones, preprocessings and validations.

        Args:
        ----
            backbones (list[str]): the backbones to test.
            preprocessings (list[str]): the preprocessings to test the backbones on.
            validations (list[str]): the validations to run for each backbone.
        """
        for backbone in backbones:
            for preprocessing in preprocessings:
                with self.subTest(f"{backbone}_{preprocessing}"), initialize(
                    "../config", version_base=None
                ):
                    cfg = compose(
                        "master.yaml",
                        overrides=[
                            f"backbone={backbone}",
                            f"preprocessing={preprocessing}",
                            f"validation={validations}",
                            f"seed={TEST_SEED}",
                        ],
                        return_hydra_config=True,
                    )
                    instance = HydraConfig.instance()
                    instance.set_config(cfg)
                    run(cfg)

    def test_run_config(self):
        """The general test case, to test all backbones on all datasets, with all validations enabled."""
        self.run_with_config(
            backbones=[
                "ae_nlp_pretrained",
                "ae_nlp_from_scratch",
                "ae_nlp_frozen",
                "coles_churn",
                "coles_emb_churn",
                "mlm",
                # "cotic", # doesn't run on a small GPU
                "ts2vec_churn",
                "coles_timecl_churn",
                "seq2vec_coles_emb_churn",
                "nhp",
                "attn_nhp",
            ],
            preprocessings=["default", "churn"],
            validations=["local_target", "event_time", "event_type", "global_target"],
        )


if __name__ == "__main__":
    unittest.main()
