from scenario_writing.utilities.context_determination import ContextDeterminator
from scenario_writing.utilities.specification_reading import SpecificationReader


if __name__ == '__main__':

    # This is just a test.

    # Declare file handles.
    specification_xml = r'D:\Projects\025_scenario_writing' \
                        r'\cluster_analysis.xml'

    # Determine context.
    context = ContextDeterminator()
    context.determine_context()

    # Get specification.
    specification = SpecificationReader(
        context=context.get_context(),
        specification_xml=specification_xml
    )
    specification.read_specification()


#TODO: In reference with above, introduction of FileManager class.
#TODO: Method to return specification.
