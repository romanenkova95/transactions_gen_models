import os
import unittest

import logging
from main import run
from pathlib import Path
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig

TEST_SEED = 473284789

class TestBackbones(unittest.TestCase):
    def setUp(self):
        os.environ["FAST_DEV_RUN"] = "True"
        os.environ["WANDB_MODE"] = "disabled"
        logging.disable(logging.CRITICAL)
        
    def tearDown(self) -> None:
        saved_test_models = Path("/app/saved_models/").glob(f"*{TEST_SEED}*")
        for file in saved_test_models:
            file.unlink()

    def run_with_config(self, backbone, preprocessing, validations):
        with self.subTest("train"), initialize("../config", version_base=None):
            cfg = compose(
                "master.yaml",
                overrides=[
                    f"backbone={backbone}",
                    f"preprocessing={preprocessing}",
                    f"validation={validations}",
                    f"seed={TEST_SEED}"
                ],
                return_hydra_config=True,
            )
            instance = HydraConfig.instance()
            instance.set_config(cfg)
            run(cfg)

    def test_ae_nlp(self):
        self.run_with_config(
            backbone="ae_nlp",
            preprocessing="churn",
            validations=["local_target", "event_time", "event_type", "global_target"],
        )

    def test_coles_churn(self):
        self.run_with_config(
            backbone="coles_churn",
            preprocessing="churn",
            validations=["local_target", "event_time", "event_type", "global_target"],
        )

    def test_coles_emb_churn(self):
        self.run_with_config(
            backbone="coles_emb_churn",
            preprocessing="churn",
            validations=["local_target", "event_time", "event_type", "global_target"],
        )

    def test_transformer(self):
        self.run_with_config(
            backbone="transformer",
            preprocessing="churn",
            validations=["local_target", "event_time", "event_type", "global_target"],
        )


if __name__ == "__main__":
    unittest.main()
