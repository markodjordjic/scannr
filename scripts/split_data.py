from scenario_writing.utilities.context_determination import ContextDeterminator
from scenario_writing.utilities.specification_reading import SpecificationReader
from scenario_writing.utilities.data_management import CSVFileManager
from scenario_writing.data_preparation.data_splitting import DataSplittingManager


if __name__ == '__main__':

    # Declare file handles.
    specification_xml = r'D:\Projects\025_scenario_writing\data_splitting.xml'

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

    # Determine the base size.
    case_count = min([
        len(file_manager.individual_data_sets[data_set_key])
        for data_set_key in file_manager.individual_data_sets.keys()
    ])
    # Data splitting.
    data_generation_manager = DataSplittingManager(
        raw_data=file_manager.individual_data_sets,
        specification=specification_reader.specification
    )
    data_generation_manager.generate_data(base_size=case_count)
    file_manager.individual_data_sets = data_generation_manager.output_data
    file_manager.write_data()

