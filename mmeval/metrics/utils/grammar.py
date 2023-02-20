# Copyright (c) OpenMMLab. All rights reserved.
from collections import Counter
from typing import Sequence


def get_n_gram(token: Sequence[str], n_gram: int) -> Counter:
    """A function get n-gram of sentences.

    Args:
        token (Sequence[str]): A series of tokens about sentences.
        n_gram (int): The maximum number of words contained in a phrase
            when calculating word fragments.

    Returns:
        Counter: The n_gram contained in sentences with Counter format.
    """
    counter: Counter = Counter()
    for i in range(1, n_gram + 1):
        for j in range(len(token) - i + 1):
            key = tuple(token[j:(i + j)])
            counter[key] += 1

    return counter


def infer_language(text: str) -> str:
    """Determine the type of language.

    Args:
        text (str): Input for language judgment.

    Returns:
        str: The type of language.
    """
    language = 'en'
    for _char in text:
        if '\u4e00' <= _char <= '\u9fa5':
            language = 'cn'
            break
    return language


def get_tokenizer(language: str):
    """A function to choose tokenizer."""
    if language == 'en':
        return str.split
    elif language in ('cn', 'zh'):
        return list
    else:
        return None
