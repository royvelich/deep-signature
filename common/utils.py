# python peripherals
import os


# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(input_list, chunks_count):
    return [input_list[i:i + chunks_count] for i in range(0, len(input_list), chunks_count)]


def list_subdirectories(base_dir='.'):
    result = []
    for current_sub_dir in os.listdir(base_dir):
        full_sub_dir_path = os.path.join(base_dir, current_sub_dir)
        if os.path.isdir(full_sub_dir_path):
            result.append(full_sub_dir_path)

    return result


def get_latest_subdirectory(base_dir='.'):
    subdirectories = list_subdirectories(base_dir)
    return os.path.normpath(max(subdirectories, key=os.path.getmtime))
