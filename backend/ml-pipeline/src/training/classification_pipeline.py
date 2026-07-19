"""
===============================================================================
Classification Pipeline
===============================================================================

Executes the complete classification workflow for logistics action prediction.

Pipeline Stages
---------------
1. Data Preparation
2. Exploratory Data Analysis
3. Feature Engineering
4. Synthetic Data Generation
5. Feature Selection
6. Validation Feature Engineering
7. Validation Feature Selection
8. Test Feature Engineering
9. Test Feature Selection
10. Model Training
11. Hyperparameter Tuning
12. Model Evaluation
13. Model Comparison

===============================================================================
"""

import logging
import time

from src.preprocessing.classification_data_preparation import (
    ClassificationDataPreparation,
)

from src.analysis.classification_eda import (
    ClassificationEDA,
)

from src.feature_engineering.classification_feature_engineering import (
    ClassificationFeatureEngineering,
)

from src.feature_engineering.classification_feature_transformer import (
    ClassificationFeatureTransformer,
)

from src.feature_selection.classification_feature_selection import (
    ClassificationFeatureSelection,
)

from src.feature_selection.classification_feature_selector import (
    ClassificationFeatureSelector,
)

from src.balancing.classification_synthetic_data_generator import (
    ClassificationSyntheticDataGenerator,
)

from src.training.classification_model_trainer import (
    ClassificationModelTrainer,
)

from src.training.classification_hyperparameter_tuner import (
    ClassificationHyperparameterTuner,
)

from src.evaluation.classification_model_evaluator import (
    ClassificationModelEvaluator,
)

from src.evaluation.classification_model_comparator import (
    ClassificationModelComparator,
)


class ClassificationPipeline:

    def __init__(
            self,
            data_path: str,
            version: str = "v1",
            random_state: int = 42,
        ):

            self.data_path = data_path

            self.version = version

            self.random_state = random_state

            logging.basicConfig(

                level=logging.INFO,

                format="%(asctime)s | %(levelname)s | %(message)s"

            )

            self.logger = logging.getLogger(__name__)


    ####################################################################
    # Execute Stage
    ####################################################################

    def execute_stage(

        self,

        stage_name,

        stage_object

    ):

        self.logger.info(

            "=" * 70

        )

        self.logger.info(

            f"Starting : {stage_name}"

        )

        start = time.time()

        stage_object.run()

        end = time.time()

        self.logger.info(

            f"Completed : {stage_name}"

        )

        self.logger.info(

            f"Execution Time : {end-start:.2f} seconds"

        )

        self.logger.info(

            "=" * 70

        )

    ####################################################################
    # Run Pipeline
    ####################################################################

    def run(self):

        total_start = time.time()

        self.logger.info(

            "\n"

            "=====================================================\n"

            " COLD CHAIN CLASSIFICATION PIPELINE\n"

            "====================================================="

        )

        ################################################################
        # Stage 1 - Data Preparation
        ################################################################

        self.execute_stage(

            "Classification Data Preparation",

            ClassificationDataPreparation(

                data_path=self.data_path,

                version=self.version,

                random_state=self.random_state

            )
        )

        ################################################################
        # Stage 2 - Exploratory Data Analysis
        ################################################################

        self.execute_stage(

            "Classification EDA",

            ClassificationEDA(

                version=self.version

            )

        )

        ################################################################
        # Stage 3 - Feature Engineering
        ################################################################

        self.execute_stage(

            "Classification Feature Engineering",

            ClassificationFeatureEngineering(

                version=self.version

            )

        )

        ################################################################
        # Stage 4 - Synthetic Data Generation
        ################################################################

        self.execute_stage(

            "Classification Synthetic Data Generation",

            ClassificationSyntheticDataGenerator(

                version=self.version,

                random_state=self.random_state

            )

        )

        ################################################################
        # Stage 5 - Feature Selection
        ################################################################

        self.execute_stage(

            "Classification Feature Selection",

            ClassificationFeatureSelection(

                version=self.version

            )

        )

        ################################################################
        # Stage 6 - Validation Feature Engineering
        ################################################################

        self.execute_stage(

            "Validation Feature Engineering",

            ClassificationFeatureTransformer(

                input_file="data/processed/classification_validation.csv",

                output_file="data/engineered/classification_validation_engineered.csv",

                version=self.version

            )

        )

        ################################################################
        # Stage 7 - Validation Feature Selection
        ################################################################

        self.execute_stage(

            "Validation Feature Selection",

            ClassificationFeatureSelector(

                input_file="data/engineered/classification_validation_engineered.csv",

                output_file="data/final/classification_validation_selected.csv",

                version=self.version

            )

        )

        ################################################################
        # Stage 8 - Test Feature Engineering
        ################################################################

        self.execute_stage(

            "Test Feature Engineering",

            ClassificationFeatureTransformer(

                input_file="data/processed/classification_test.csv",

                output_file="data/engineered/classification_test_engineered.csv",

                version=self.version

            )

        )

        ################################################################
        # Stage 9 - Test Feature Selection
        ################################################################

        self.execute_stage(

            "Test Feature Selection",

            ClassificationFeatureSelector(

                input_file="data/engineered/classification_test_engineered.csv",

                output_file="data/final/classification_test_selected.csv",

                version=self.version

            )

        )

        ################################################################
        # Stage 10 - Model Training
        ################################################################

        self.execute_stage(

            "Classification Model Training",

            ClassificationModelTrainer(

                version=self.version,

                random_state=self.random_state

            )

        )

        ################################################################
        # Stage 11 - Hyperparameter Tuning
        ################################################################

        self.execute_stage(

            "Classification Hyperparameter Tuning",

            ClassificationHyperparameterTuner(

                version=self.version,

                random_state=self.random_state

            )

        )

        ################################################################
        # Stage 12 - Model Evaluation
        ################################################################

        self.execute_stage(

            "Classification Model Evaluation",

            ClassificationModelEvaluator(

                version=self.version

            )

        )

        ################################################################
        # Stage 13 - Model Comparison
        ################################################################

        self.execute_stage(

            "Classification Model Comparison",

            ClassificationModelComparator(

                version=self.version

            )

        )

        total_end = time.time()

        self.logger.info(

            "\n"

            "=====================================================\n"

            " CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY\n"

            "====================================================="

        )

        self.logger.info(

            f"Total Execution Time : "

            f"{total_end - total_start:.2f} seconds"

        )

        self.logger.info(

            "All pipeline stages executed successfully."

        )


###############################################################################
# Main
###############################################################################

def main():

    try:

        pipeline = ClassificationPipeline()

        pipeline.run()

    except KeyboardInterrupt:

        logging.warning(

            "Pipeline execution interrupted by user."

        )

    except Exception as exception:

        logging.exception(

            f"Pipeline execution failed: {exception}"

        )

        raise


if __name__ == "__main__":

    main()