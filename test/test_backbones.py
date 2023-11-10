import os
import unittest
import warnings

import logging
from main import run
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig


class TestBackbones(unittest.TestCase):
    def setUp(self):
        os.environ["FAST_DEV_RUN"] = "True"
        os.environ["WANDB_MODE"] = "disabled"
        logging.disable(logging.CRITICAL)

    def run_with_config(self, backbone, preprocessing, validations):
        with self.subTest("train"), initialize("../config", version_base=None):
            cfg = compose(
                "master.yaml",
                overrides=[
                    f"backbone={backbone}",
                    f"preprocessing={preprocessing}",
                    f"validation={validations}",
                ],
                return_hydra_config=True,
            )
            instance = HydraConfig.instance()
            instance.set_config(cfg)
            run(cfg)

    def test_ae_nlp(self):
        self.run_with_config(
            backbone="ae_nlp",
            preprocessing="churn_100",
            validations=["local_target", "event_time", "event_type_100"],
        )

    def test_coles_churn(self):
        self.run_with_config(
            backbone="coles_churn",
            preprocessing="churn",
            validations=["local_target", "event_time", "event_type"],
        )

    def test_coles_emb_churn(self):
        self.run_with_config(
            backbone="coles_emb_churn",
            preprocessing="churn",
            validations=["local_target", "event_time", "event_type"],
        )

    def test_transformer(self):
        self.run_with_config(
            backbone="transformer",
            preprocessing="churn_100",
            validations=["local_target", "event_time", "event_type_100"],
        )


if __name__ == "__main__":
    unittest.main()
