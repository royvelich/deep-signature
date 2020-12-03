# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(input_list, chunks_count):
    return [input_list[i:i + chunks_count] for i in range(0, len(input_list), chunks_count)]