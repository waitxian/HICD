"""Utils for interacting with huggingface tokenizers."""
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence, Tuple

from pastalib.utils.typing import StrSequence, Tokenizer, TokenizerOffsetMapping
import ipdb

def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[Any] = None,
    offset_mapping: Optional[Any] = None,
    **kwargs: Any,
) -> list[Tuple[int, int]]:
    """Find index ranges of tokenized string for each substring separated by space.

    Args:
        string: The original string.
        substring: Substrings separated by space, e.g., 'batman night'.
        tokenizer: Tokenizer object. If not set, offset_mapping must be.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If any substring is not found in the string.

    Returns:
        List[Tuple[int, int]]: A list of token ranges (start, end) for each substring.
    """
    ipdb.set_trace()
    if tokenizer is None and offset_mapping is None:
        raise ValueError("Must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("Cannot set return_offsets_mapping in kwargs")

    # Split the input substring into separate words
    substrings = substring.split()

    # Prepare to generate offset mapping if not provided
    if offset_mapping is None:
        assert tokenizer is not None
        tokens = tokenizer(string, return_offsets_mapping=True, **kwargs)
        offset_mapping = tokens['offset_mapping']

    token_ranges = []
    for sub in substrings:
        if sub not in string:
            raise ValueError(f'"{sub}" not found in "{string}"')
        
        # Find character start and end positions for each substring
        char_start = string.index(sub)
        char_end = char_start + len(sub)

        # Find token start and end positions
        token_start, token_end = None, None
        for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
            if token_start is None and token_char_start <= char_start < token_char_end:
                token_start = index
            if token_end is None and token_char_start < char_end <= token_char_end:
                token_end = index
                break

        # Ensure valid range was found
        assert token_start is not None
        assert token_end is not None
        assert token_start <= token_end

        # Append the token range to the list
        token_ranges.append((token_start, token_end + 1))
    ipdb.set_trace()

    return token_ranges


if __name__ == "__main__":
    string = 'The batman is the night.'
    substring = 'batman night'
    # 假设分词器的 token 列表为 ['the', 'bat', '##man', 'is', 'the', 'night']
    # 其中 'batman' 对应的 token 为索引 1 到 3，'night' 对应的 token 为索引 5

    #assert find_token_ranges(string, substring, tokenizer) == [(1, 3), (5, 6)]
    print(find_token_ranges(string, substring, tokenizer) )

