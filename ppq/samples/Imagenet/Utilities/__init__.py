from typing import List


def shape_to_str(shape: List[int]) -> str:
    if len(shape) == 1:
        return str(shape[0])
    string_builder = str(shape[0])
    for s in shape[1: ]:
        string_builder += '_' + str(s)
    return string_builder