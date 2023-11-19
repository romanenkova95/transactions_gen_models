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

    def run_with_config(self, backbones, preprocessings, validations):
        for backbone in backbones:
            for preprocessing in preprocessings:
                with self.subTest(f"{backbone}_{preprocessing}"), initialize("../config", version_base=None):
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

    def test_run_config(self):
        self.run_with_config(
            backbones=[
                "ae_nlp_pretrained", 
                "ae_nlp_from_scratch",
                "ae_nlp_frozen",
                "coles_churn", 
                "coles_emb_churn", 
                "mlm"
            ],
            preprocessings=["default", "churn"],
            validations=["local_target", "event_time", "event_type", "global_target"],
        )

if __name__ == "__main__":
    unittest.main()
