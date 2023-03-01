# Copyright (c) OpenMMLab. All rights reserved.
# This class is modified from `torchmetrics
# <https://github.com/Lightning-AI/metrics/blob/master/src/torchmetrics/text/sacre_bleu.py>`_.
import re
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

from mmeval.metrics import BLEU
from mmeval.utils import try_import

if TYPE_CHECKING:
    import regex
else:
    regex = try_import('regex')

CHINESE_UCODE_RANGES = (
    ('\u3400', '\u4db5'),
    ('\u4e00', '\u9fa5'),
    ('\u9fa6', '\u9fbb'),
    ('\uf900', '\ufa2d'),
    ('\ufa30', '\ufa6a'),
    ('\ufa70', '\ufad9'),
    ('\u20000', '\u2a6d6'),
    ('\u2f800', '\u2fa1d'),
    ('\uff00', '\uffef'),
    ('\u2e80', '\u2eff'),
    ('\u3000', '\u303f'),
    ('\u31c0', '\u31ef'),
    ('\u2f00', '\u2fdf'),
    ('\u2ff0', '\u2fff'),
    ('\u3100', '\u312f'),
    ('\u31a0', '\u31bf'),
    ('\ufe10', '\ufe1f'),
    ('\ufe30', '\ufe4f'),
    ('\u2600', '\u26ff'),
    ('\u2700', '\u27bf'),
    ('\u3200', '\u32ff'),
    ('\u3300', '\u33ff'),
)

_REGEX = (
    # tokenize period and comma unless preceded by a digit
    (re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'), r' \1 '),
    # tokenize period and comma unless followed by a digit
    (re.compile(r'([^0-9])([\.,])'), r'\1 \2 '),
    # tokenize dash when preceded by a digit
    (re.compile(r'([\.,])([^0-9])'), r' \1 \2'),
    # one space only between words
    (re.compile(r'([0-9])(-)'), r'\1 \2 '),
)

if regex is not None:
    _INT_REGEX = (
        # p{S} : Match all special symbols, including symbols, operators, punctuation marks, etc.  # noqa: E501
        # p{P} : Match all punctuation characters, including symbols, punctuation marks, etc.  # noqa: E501
        # P{N} : Match all non-numeric characters
        # Separate out punctuations preceded by a non-digit
        (regex.compile(r'(\P{N})(\p{P})'), r'\1 \2 '),
        # Separate out punctuations followed by a non-digit
        (regex.compile(r'(\p{P})(\P{N})'), r' \1 \2'),
        # Separate out symbols
        (regex.compile(r'(\p{S})'), r' \1 '),
    )


class _BaseTokenizer:
    """Tokenizer used for SacreBLEU calculation.。

    Args:
         lowercase (bool): Whether to lower case chars when setting tokenizer.
             Defaults to False.
    """

    def __init__(self, lowercase: bool = False) -> None:
        self.lowercase = lowercase

    @classmethod
    def _tokenize_regex(cls, line: str) -> str:
        """The general post-processing tokenizer of ``13a`` and ``zh``.

        Args:
            line (str): a segment to tokenize

        Return:
            str: the tokenized line
        """
        for (_re, repl) in _REGEX:
            line = _re.sub(repl, line)
        return ' '.join(line.split())

    @classmethod
    def _tokenize_base(cls, line: str) -> str:
        """Do not process the input, and return directly.

        Args:
            line (str): The input string to tokenize.

        Returns:
            str: The tokenized string.
        """
        return line

    @staticmethod
    def _lower(line: str, lowercase: bool) -> str:
        """To lower case chars when setting tokenizer.

        Args:
            line (str): The input need to tokenize.
            lowercase (bool): Whether to minimize chars.

        Returns:
            str: The input after minimization.
        """
        if lowercase:
            return line.lower()
        return line

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_base(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _13aTokenizer(_BaseTokenizer):
    """Tokenizer used for SacreBLEU calculation of ``13a``.。

    Args:
         lowercase (bool): Whether to lower case chars when setting tokenizer.
             Defaults to False.
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @classmethod
    def _tokenize_13a(cls, line: str) -> str:
        """Tokenizes an input line using a relatively minimal tokenization that
        is however equivalent to mteval-v13a, used by WMT.

        Args:
            line (str): The input string to tokenize.

        Returns:
            str: The tokenized string.
        """
        line = line.replace('<skipped>', '')
        line = line.replace('-\n', '')
        line = line.replace('\n', ' ')

        if '&' in line:
            line = line.replace('&quot;', '"')
            line = line.replace('&amp;', '&')
            line = line.replace('&lt;', '<')
            line = line.replace('&gt;', '>')

        return cls._tokenize_regex(line)

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_13a(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _zhTokenizer(_BaseTokenizer):
    """Tokenizer used for SacreBLEU calculation of ``zh``.。

    Args:
         lowercase (bool): Whether to lower case chars when setting tokenizer.
             Defaults to False.
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @staticmethod
    def _is_chinese_char(uchar: str) -> bool:
        """separates out Chinese characters and tokenizes the non-Chinese parts
        using 13a tokenizer.

        Args:
            uchar (str): Input char in unicode

        Returns:
            str: Whether the input char is a Chinese character.
        """
        for start, end in CHINESE_UCODE_RANGES:
            if start <= uchar <= end:
                return True
        return False

    @classmethod
    def _tokenize_zh(cls, line: str) -> str:
        """The tokenization of Chinese text in this script contains two
        steps: separate each Chinese characters.

        Args:
            line (str): The input string to tokenize.

        Returns:
            str: The tokenized string.
        """
        line = line.strip()
        line_in_chars = ''

        for char in line:
            if cls._is_chinese_char(char):
                line_in_chars += ' '
                line_in_chars += char
                line_in_chars += ' '
            else:
                line_in_chars += char

        return cls._tokenize_regex(line_in_chars)

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_zh(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _intlTokenizer(_BaseTokenizer):
    """Tokenizer used for SacreBLEU calculation of ``international``.。

    Args:
         lowercase (bool): Whether to lower case chars when setting tokenizer.
             Defaults to False.
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @classmethod
    def _tokenize_international(cls, line: str) -> str:
        """Tokenizes a string following the official BLEU implementation.

        Args:
            line (str): The input string to tokenize.

        Returns:
            str: The tokenized string.
        """
        for (_re, repl) in _INT_REGEX:
            line = _re.sub(repl, line)

        return ' '.join(line.split())

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_international(line)
        return self._lower(tokenized_line, self.lowercase).split()


class _charTokenizer(_BaseTokenizer):
    """Tokenizer used for SacreBLEU calculation of ``char``.。

    Args:
         lowercase (bool): Whether to lower case chars when setting tokenizer.
             Defaults to False.
    """

    def __init__(self, lowercase: bool = False):
        super().__init__(lowercase)

    @classmethod
    def _tokenize_char(cls, line: str) -> str:
        """Tokenizes all the characters in the input line.

        Args:
            line (str): The input string to tokenize.

        Returns:
            str: The tokenized string.
        """
        return ' '.join(char for char in line)

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self._tokenize_char(line)
        return self._lower(tokenized_line, self.lowercase).split()


def _get_tokenizer(tokenizer: str, lowercase: bool):
    """Choose tokenizer of sacre_bleu。

    Args:
         tokenizer (str): Abbreviation of each tokenizer class.
         lowercase (bool): Whether to lower case chars when setting tokenizer.
             Defaults to False.
    """
    if tokenizer == 'none':
        tokenizer_fn = _BaseTokenizer(lowercase)
    elif tokenizer == '13a':
        tokenizer_fn = _13aTokenizer(lowercase)
    elif tokenizer == 'zh':
        tokenizer_fn = _zhTokenizer(lowercase)
    elif tokenizer == 'intl':
        tokenizer_fn = _intlTokenizer(lowercase)
    elif tokenizer == 'char':
        tokenizer_fn = _charTokenizer(lowercase)
    else:
        raise ValueError('Right now, `tokenizer_fn` only supports pre-defined '
                         "'none', '13a', 'intl', 'char', 'zh'.")
    return tokenizer_fn


class SacreBLEU(BLEU):
    """Calculate `BLEU score`_ of machine translated text with one or more
    references.

    This metric proposed in `A Call for Clarity in Reporting BLEU Scores
    <https://aclanthology.org/W18-6319.pdf>`_ is an improvement of BLEU calculation.

    Args:
        n_gram (int): The maximum number of words contained in a phrase
            when calculating word fragments. Defaults to 4.
        smooth (bool): Whether or not to apply to smooth. Defaults to False.
        ngram_weights (Sequence[float], optional): Weights used
            for unigrams, bigrams, etc. to calculate BLEU score.
            If not provided, uniform weights are used. Defaults to None.
        lowercase (bool): Whether to lower case chars when setting tokenizer.
            Defaults to False.
        tokenizer_fn (Callable or str, optional): A user's own tokenizer function.
            Defaults to None.
        **kwargs: Keyword parameters passed to :class:`BLEU`.

    Examples:
        >>> from mmeval import SacreBLEU
        >>> predictions = ['the cat is on the mat', 'There is a big tree near the park here']  # noqa: E501
        >>> references = [['a cat is on the mat'], ['A big tree is growing near the park here']]  # noqa: E501
        >>> bleu = SacreBLEU()
        >>> bleu_results = bleu(predictions, references)
        {'bleu': 0.5226045319355426}
    """

    def __init__(self,
                 n_gram: int = 4,
                 smooth: bool = False,
                 ngram_weights: Optional[Sequence[float]] = None,
                 lowercase: bool = False,
                 tokenizer_fn: Union[Callable, str] = None,
                 **kwargs) -> None:
        super().__init__(
            n_gram=n_gram,
            smooth=smooth,
            ngram_weights=ngram_weights,
            **kwargs)
        self.lowercase = lowercase
        if tokenizer_fn is None:
            tokenizer_fn = 'none'
        if tokenizer_fn == 'intl' and not try_import('regex'):
            raise ValueError('`intl` needs regex package, please make sure '
                             'you have already installed it`')
        if callable(tokenizer_fn):
            self.tokenizer_fn = tokenizer_fn
        elif isinstance(tokenizer_fn, str):
            self.tokenizer_fn = _get_tokenizer(tokenizer_fn, self.lowercase)
        else:
            raise ValueError(f'`tokenizer_fn` supports Callable, str or None, '
                             f'but not `{type(tokenizer_fn)}`')
