import os
import sys
from scenario_writing.utilities.context_determination import ContextDeterminator
from scenario_writing.utilities.specification_reading import SpecificationReader
from scenario_writing.utilities.data_management import CSVFileManager
from scenario_writing.utilities.file_management import ModelArtefactManager
from scenario_writing.model_training.model_training import ModelTrainer

if __name__ == '__main__':

    # Declare file handles.
    specification_xml = \
        r'/workspaces/025_scenario_writing/train_model.xml'

    # Determine context.
    context_determinator = ContextDeterminator()
    context_determinator.determine_context()

    # Read specification.
    specification_reader = SpecificationReader(
        context=context_determinator.get_context(),
        specification_xml=specification_xml
    )
    specification_reader.read_specification()

    # Get Data.
    file_manager = CSVFileManager(
        context=context_determinator.get_context(),
        specification=specification_reader.specification
    )
    file_manager.read_data()

    # Data splitting.
    model_trainer = ModelTrainer(
        raw_data_set=file_manager.get_data()[0],
        specification=specification_reader.specification
    )
    model_trainer.create_pipeline()
    model_trainer.execute_pipeline()

    # Write model artefacts.
    model_artefact_manager = ModelArtefactManager(
        context=context_determinator.get_context(),
        specification=specification_reader.specification,
        trained_artefacts={
            'artefacts' : {
                'vectorizer': model_trainer.trained_vectorizer,
                'lsi': model_trainer.trained_lsi
            }
        }
    )
    model_artefact_manager.write_model_artefacts()
    sys.exit()
