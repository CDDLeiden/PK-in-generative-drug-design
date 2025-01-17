"""Scorers and DrugEx environment definitions for reinforcement learning"""

import logging
from typing import Callable

import numpy as np
from drugex.logs import logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoTanimotoDistance
from drugex.training.scorers.interfaces import Scorer
from drugex.training.scorers.modifiers import ClippedScore
from qsprpred.data.tables.mol import MoleculeTable
from qsprpred.data.tables.qspr import QSPRDataset
from qsprpred.models.scikit_learn import SklearnModel
from rdkit import Chem
from rdkit.Chem import Mol

# set up logging
logger = logging.getLogger(__name__)


class MyQSPRModel(SklearnModel):
    """Custom wrapper for SklearnModel to mark molecules with missing descriptor values as invalid"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def cast(cls, obj: SklearnModel) -> "MyQSPRModel":
        """Cast an SklearnModel object to this class."""
        assert isinstance(obj, SklearnModel)
        obj.__class__ = cls
        assert isinstance(obj, cls)
        return obj

    def createPredictionDatasetFromMols(
        self,
        mols: list[str | Mol],
        smiles_standardizer: str | Callable[[str], str] = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
    ) -> tuple[QSPRDataset, np.ndarray]:
        """Create a `QSPRDataset` instance from a list of SMILES strings.

        Args:
            mols (list[str | Mol]): list of SMILES strings
            smiles_standardizer (str, callable): smiles standardizer to use
            n_jobs (int): number of parallel jobs to use
            fill_value (float): value to fill for missing features

        Returns:
            tuple:
                a tuple containing the `QSPRDataset` instance and a boolean mask
                indicating which molecules failed to be processed
        """
        # make a molecule table first and add the target properties
        if isinstance(mols[0], Mol):
            mols = [Chem.MolToSmiles(mol) for mol in mols]
        dataset = MoleculeTable.fromSMILES(
            f"{self.__class__.__name__}_{hash(self)}",
            mols,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        for target_property in self.targetProperties:
            target_property.imputer = None
            dataset.addProperty(target_property.name, np.nan)
        # create the dataset and get failed molecules
        dataset = QSPRDataset.fromMolTable(
            dataset,
            self.targetProperties,
            drop_empty=False,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        dataset.standardizeSmiles(smiles_standardizer, drop_invalid=False)
        failed_mask = dataset.dropInvalids()  # returns pd series
        # prepare dataset and return it
        dataset.prepareDataset(
            smiles_standardizer=smiles_standardizer,
            feature_calculators=self.featureCalculators,
            feature_standardizer=self.featureStandardizer,
            feature_fill_value=fill_value,
            shuffle=False,
        )
        features = dataset.getFeatures(
            concat=True, ordered=True, refit_standardizer=False
        )
        # Mask out molecules with nan values in the features
        nan_rows = np.isnan(features).any(axis=1)
        if np.any(nan_rows):
            # print smiles of molecules with nan values
            for mol in dataset.df[dataset.smilesCol].values[nan_rows]:
                logger.warning(f"Nan values in features for: {mol}")
        nan_indices = dataset.df.index[nan_rows]
        dataset.df.drop(nan_indices, inplace=True)
        dataset.restoreTrainingData()

        # set indexes in the failed_mask to true where nan_rows is true
        failed_mask.loc[nan_indices] = True

        return dataset, failed_mask.values


# Define DrugEX QSPRpred scorer with applicability domain
class QSPRPredScorer(Scorer):
    """Custom QSPRpredscorer to enable applicability domain scoring"""

    def __init__(
        self, model, invalids_score=0.0, modifier=None, name="scorer", **kwargs
    ):
        self.score_modifier = modifier
        super(QSPRPredScorer, self).__init__(modifier)
        self.model = model
        self.invalidsScore = invalids_score
        self.name = name
        self.kwargs = kwargs
        # toggle to switch between scores and applicability
        self.toggle = False

    def calulateScores(self, mols, frag=None):
        valid_mols = [mol for mol in mols if mol is not None]

        if len(valid_mols) == 0:
            logger.warning(
                "No valid molecules to score. Return only invalids scores..."
            )
            self.modifier = self.score_modifier
            yield np.full(len(mols), self.invalidsScore, dtype=float)
            self.modifier = None
            yield np.full(len(mols), 0, dtype=float)
        else:
            scores, applicability = self.model.predictMols(
                valid_mols, use_applicability_domain=True, **self.kwargs
            )

            def replace_invalids(mols, scores_to_replace, invalids_score):
                scores_to_replace = np.array(scores_to_replace).flatten()

                # Replace all None values in scores_to_replace with invalids_score
                scores = np.where(
                    scores_to_replace == None, invalids_score, scores_to_replace
                )

                # Replace unparsable mols with invalids_score
                replaced_scores = np.full(len(mols), invalids_score, dtype=float)
                valid_mask = np.array([mol is not None for mol in mols])
                replaced_scores[valid_mask] = scores

                return replaced_scores

            scores = replace_invalids(mols, scores, self.invalidsScore)
            applicability = replace_invalids(mols, applicability, 0)

            self.modifier = self.score_modifier
            yield scores
            self.modifier = None
            yield applicability

    def setModifier(self, modifier):
        self.score_modifier = modifier
        return super().setModifier(modifier)

    def getScores(self, mols, frags=None):
        try:
            return next(self.calc)
        except (StopIteration, AttributeError):
            self.calc = self.calulateScores(mols, frags)
            return next(self.calc)

    def getKey(self):
        self.toggle = not self.toggle
        if self.toggle:
            return self.name
        else:
            return self.name.removesuffix("_scorer") + "_applicability_scorer"


environ_finetune_dict = {
    "A2AR": "A2AR",
    "FUmin": "FU",
    "FUmax": "FU",
    "VDSSmin": "VDSS",
    "VDSSmax": "VDSS",
    "CLmin": "CL",
    "CLmax": "CL",
    "A2AR_FUmin": "A2AR_FU",
    "A2AR_FUmax": "A2AR_FU",
    "A2AR_VDSSmin": "A2AR_VDSS",
    "A2AR_VDSSmax": "A2AR_VDSS",
    "A2AR_CLmin": "A2AR_CL",
    "A2AR_CLmax": "A2AR_CL",
}


def get_environ_dict(
    QSPR_PATH: str,
    A2AR_model: str = "SVR_A2AR_MorganFP_RDkit_B_80_HC_0.99",
    FU_model: str = "SVR_FU_MorganFP_RDkit_B_80_HC_0.99",
    VDSS_model: str = "SVR_VDSS_MorganFP_RDkit_B_80_HC_0.99",
    CL_model: str = "SVR_CL_MorganFP_RDkit_B_80_HC_0.95",
    N_CPU: int = 5,
    reward_scheme=ParetoTanimotoDistance,
    return_scorers: bool = False,
):
    logger.info(
        {
            "A2AR_model": A2AR_model,
            "CL_model": CL_model,
            "VDSS_model": VDSS_model,
            "FU_model": FU_model,
            "N_CPU": N_CPU,
            "reward_scheme": reward_scheme,
        }
    )

    # Define Adenosine receptor A2AR_scorer
    predictor_A2AR = MyQSPRModel.cast(
        SklearnModel.fromFile(f"{QSPR_PATH}/{A2AR_model}/{A2AR_model}_meta.json")
    )
    A2AR_scorer = QSPRPredScorer(
        predictor_A2AR, invalids_score=0.1, name="A2AR_scorer", n_jobs=N_CPU
    )
    A2AR_scorer.setModifier(ClippedScore(lower_x=5.5, upper_x=8.6))

    # Define FU Scorers
    predictor_fu = MyQSPRModel.cast(
        SklearnModel.fromFile(f"{QSPR_PATH}/{FU_model}/{FU_model}_meta.json")
    )
    FUmax_scorer = QSPRPredScorer(
        predictor_fu, invalids_score=0, name="FUmax_scorer", n_jobs=N_CPU
    )
    FUmax_scorer.setModifier(ClippedScore(lower_x=0.1, upper_x=0.95))
    FUmin_scorer = QSPRPredScorer(
        predictor_fu, invalids_score=1, name="FUmin_scorer", n_jobs=N_CPU
    )
    FUmin_scorer.setModifier(ClippedScore(lower_x=0.95, upper_x=0.1))

    # Define VDSS Scorers
    predictor_vdss = MyQSPRModel.cast(
        SklearnModel.fromFile(f"{QSPR_PATH}/{VDSS_model}/{VDSS_model}_meta.json")
    )
    VDSSmax_scorer = QSPRPredScorer(
        predictor_vdss, invalids_score=-1, name="VDSSmax_scorer", n_jobs=N_CPU
    )
    VDSSmax_scorer.setModifier(ClippedScore(lower_x=-0.74, upper_x=0.95))
    VDSSmin_scorer = QSPRPredScorer(
        predictor_vdss, invalids_score=4, name="VDSSmin_scorer", n_jobs=N_CPU
    )
    VDSSmin_scorer.setModifier(ClippedScore(lower_x=0.95, upper_x=-0.74))

    # Define CL Scorers
    predictor_cl = MyQSPRModel.cast(
        SklearnModel.fromFile(f"{QSPR_PATH}/{CL_model}/{CL_model}_meta.json")
    )
    CLmax_scorer = QSPRPredScorer(
        predictor_cl, invalids_score=-1, name="CLmax_scorer", n_jobs=N_CPU
    )
    CLmax_scorer.setModifier(ClippedScore(lower_x=-0.13, upper_x=1.34))
    CLmin_scorer = QSPRPredScorer(
        predictor_cl, invalids_score=4, name="CLmin_scorer", n_jobs=N_CPU
    )
    CLmin_scorer.setModifier(ClippedScore(lower_x=1.34, upper_x=-0.13))

    if return_scorers:
        return (
            A2AR_scorer,
            FUmax_scorer,
            FUmin_scorer,
            VDSSmax_scorer,
            VDSSmin_scorer,
            CLmax_scorer,
            CLmin_scorer,
        )

    # Make dict of environments to test
    environ_dict = {
        "A2AR": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "FUmin": DrugExEnvironment(
            [FUmin_scorer, FUmin_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "FUmax": DrugExEnvironment(
            [FUmax_scorer, FUmax_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "VDSSmin": DrugExEnvironment(
            [VDSSmin_scorer, VDSSmin_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "VDSSmax": DrugExEnvironment(
            [VDSSmax_scorer, VDSSmax_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "CLmin": DrugExEnvironment(
            [CLmin_scorer, CLmin_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "CLmax": DrugExEnvironment(
            [CLmax_scorer, CLmax_scorer], [0.5, 0.5], reward_scheme=reward_scheme()
        ),
        "A2AR_FUmin": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer, FUmin_scorer, FUmin_scorer],
            [0.5, 0.5, 0.5, 0.5],
            reward_scheme=reward_scheme(),
        ),
        "A2AR_FUmax": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer, FUmax_scorer, FUmax_scorer],
            [0.5, 0.5, 0.5, 0.5],
            reward_scheme=reward_scheme(),
        ),
        "A2AR_VDSSmin": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer, VDSSmin_scorer, VDSSmin_scorer],
            [0.5, 0.5, 0.5, 0.5],
            reward_scheme=reward_scheme(),
        ),
        "A2AR_VDSSmax": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer, VDSSmax_scorer, VDSSmax_scorer],
            [0.5, 0.5, 0.5, 0.5],
            reward_scheme=reward_scheme(),
        ),
        "A2AR_CLmin": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer, CLmin_scorer, CLmin_scorer],
            [0.5, 0.5, 0.5, 0.5],
            reward_scheme=reward_scheme(),
        ),
        "A2AR_CLmax": DrugExEnvironment(
            [A2AR_scorer, A2AR_scorer, CLmax_scorer, CLmax_scorer],
            [0.5, 0.5, 0.5, 0.5],
            reward_scheme=reward_scheme(),
        ),
    }

    all_scorers = [
        A2AR_scorer,
        A2AR_scorer,
        FUmax_scorer,
        FUmax_scorer,
        VDSSmax_scorer,
        VDSSmax_scorer,
        CLmax_scorer,
        CLmax_scorer,
    ]

    all_scorers_env = DrugExEnvironment(
        all_scorers, [0.5] * len(all_scorers), reward_scheme=reward_scheme()
    )
    return environ_dict, all_scorers_env
