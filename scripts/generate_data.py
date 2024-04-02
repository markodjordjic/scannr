from scenario_writing.data_preparation.data_generation import DataGenerationManager
from scenario_writing.utilities.specification_reading import SpecificationReader
from scenario_writing.utilities.context_determination import ContextDeterminator
from scenario_writing.utilities.data_management import CSVFileManager

if __name__ == '__main__':

    # Declare file handles.
    specification_xml = r'D:\Projects\025_scenario_writing\data_generation.xml'

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

    # Data generation.
    data_generation_manager = DataGenerationManager(
        raw_data=file_manager.individual_data_sets,
        specification=specification_reader.specification
    )
    data_generation_manager.generate_data()
    file_manager.individual_data_sets = data_generation_manager.training_data
    file_manager.write_data()

