# Copyright (c) OpenMMLab. All rights reserved.
"""This module provides unified file I/O related functions, which support
operating I/O with different file backends based on the specified filepath or
backend_args.

MMEval currently supports five file backends:

- LocalBackend
- PetrelBackend
- HTTPBackend
- LmdbBackend
- MemcacheBackend

Note that this module provide a union of all of the above file backends so
NotImplementedError will be raised if the interface in the file backend is not
implemented.

There are two ways to call a method of a file backend:

- Initialize a file backend with ``get_file_backend`` and call its methods.
- Directory call unified I/O functions, which will call ``get_file_backend``
  first and then call the corresponding backend method.

Examples:
    >>> # Initialize a file backend and call its methods
    >>> import mmeval.fileio as fileio
    >>> backend = fileio.get_file_backend(backend_args={'backend': 'petrel'})
    >>> backend.get('s3://path/of/your/file')

    >>> # Directory call unified I/O functions
    >>> fileio.get('s3://path/of/your/file')
"""
import json
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union

from mmeval.utils import is_filepath
from .backends import backends, prefix_to_backends
from .handlers import file_handlers

backend_instances: dict = {}


def _parse_uri_prefix(uri: Union[str, Path]) -> str:
    """Parse the prefix of uri.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.

    Examples:
        >>> _parse_uri_prefix('/home/path/of/your/file')
        ''
        >>> _parse_uri_prefix('s3://path/of/your/file')
        's3'
        >>> _parse_uri_prefix('clusterName:s3://path/of/your/file')
        's3'

    Returns:
        str: Return the prefix of uri if the uri contains '://'. Otherwise,
        return ''.
    """
    assert is_filepath(uri)
    uri = str(uri)
    # if uri does not contains '://', the uri will be handled by
    # LocalBackend by default
    if '://' not in uri:
        return ''
    else:
        prefix, _ = uri.split('://')
        # In the case of PetrelBackend, the prefix may contain the cluster
        # name like clusterName:s3://path/of/your/file
        if ':' in prefix:
            _, prefix = prefix.split(':')
        return prefix


def _get_file_backend(prefix: str, backend_args: dict):
    """Return a file backend based on the prefix or backend_args.

    Args:
        prefix (str): Prefix of uri.
        backend_args (dict): Arguments to instantiate the corresponding
            backend.
    """
    # backend name has a higher priority
    if 'backend' in backend_args:
        backend_name = backend_args.pop('backend')
        backend = backends[backend_name](**backend_args)
    else:
        backend = prefix_to_backends[prefix](**backend_args)
    return backend


def get_file_backend(
    uri: Union[str, Path, None] = None,
    *,
    backend_args: Optional[dict] = None,
    enable_singleton: bool = False,
):
    """Return a file backend based on the prefix of uri or backend_args.

    Args:
        uri (str or Path): Uri to be parsed that contains the file prefix.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        enable_singleton (bool): Whether to enable the singleton pattern.
            If it is True, the backend created will be reused if the
            signature is same with the previous one. Defaults to False.

    Returns:
        BaseStorageBackend: Instantiated Backend object.

    Examples:
        >>> # get file backend based on the prefix of uri
        >>> uri = 's3://path/of/your/file'
        >>> backend = get_file_backend(uri)
        >>> # get file backend based on the backend_args
        >>> backend = get_file_backend(backend_args={'backend': 'petrel'})
        >>> # backend name has a higher priority if 'backend' in backend_args
        >>> backend = get_file_backend(uri, backend_args={'backend': 'petrel'})
    """
    global backend_instances

    if backend_args is None:
        backend_args = {}

    if uri is None and 'backend' not in backend_args:
        raise ValueError(
            'uri should not be None when "backend" does not exist in '
            'backend_args')

    if uri is not None:
        prefix = _parse_uri_prefix(uri)
    else:
        prefix = ''

    if enable_singleton:
        # TODO: whether to pass sort_key to json.dumps
        unique_key = f'{prefix}:{json.dumps(backend_args)}'
        if unique_key in backend_instances:
            return backend_instances[unique_key]

        backend = _get_file_backend(prefix, backend_args)
        backend_instances[unique_key] = backend
        return backend
    else:
        backend = _get_file_backend(prefix, backend_args)
        return backend


def get(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bytes:
    """Read bytes from a given ``filepath`` with 'rb' mode.

    Args:
        filepath (str or Path): Path to read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bytes: Expected bytes object.

    Examples:
        >>> filepath = '/path/of/file'
        >>> get(filepath)
        b'hello world'
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.get(filepath)


def get_text(
    filepath: Union[str, Path],
    encoding='utf-8',
    backend_args: Optional[dict] = None,
) -> str:
    """Read text from a given ``filepath`` with 'r' mode.

    Args:
        filepath (str or Path): Path to read data.
        encoding (str): The encoding format used to open the ``filepath``.
            Defaults to 'utf-8'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: Expected text reading from ``filepath``.

    Examples:
        >>> filepath = '/path/of/file'
        >>> get_text(filepath)
        'hello world'
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.get_text(filepath, encoding)


def exists(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Check whether a file path exists.

    Args:
        filepath (str or Path): Path to be checked whether exists.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

    Examples:
        >>> filepath = '/path/of/file'
        >>> exists(filepath)
        True
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.exists(filepath)


def isdir(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Check whether a file path is a directory.

    Args:
        filepath (str or Path): Path to be checked whether it is a
            directory.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return ``True`` if ``filepath`` points to a directory,
        ``False`` otherwise.

    Examples:
        >>> filepath = '/path/of/dir'
        >>> isdir(filepath)
        True
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.isdir(filepath)


def isfile(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bool:
    """Check whether a file path is a file.

    Args:
        filepath (str or Path): Path to be checked whether it is a file.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        bool: Return ``True`` if ``filepath`` points to a file, ``False``
        otherwise.

    Examples:
        >>> filepath = '/path/of/file'
        >>> isfile(filepath)
        True
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.isfile(filepath)


def join_path(
    filepath: Union[str, Path],
    *filepaths: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Union[str, Path]:
    """Concatenate all file paths.

    Join one or more filepath components intelligently. The return value
    is the concatenation of filepath and any members of *filepaths.

    Args:
        filepath (str or Path): Path to be concatenated.
        *filepaths (str or Path): Other paths to be concatenated.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Returns:
        str: The result of concatenation.

    Examples:
        >>> filepath1 = '/path/of/dir1'
        >>> filepath2 = 'dir2'
        >>> filepath3 = 'path/of/file'
        >>> join_path(filepath1, filepath2, filepath3)
        '/path/of/dir/dir2/path/of/file'
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    return backend.join_path(filepath, *filepaths)


@contextmanager
def get_local_path(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Generator[Union[str, Path], None, None]:
    """Download data from ``filepath`` and write the data to local path.

    ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
    can be called with ``with`` statement, and when exists from the
    ``with`` statement, the temporary path will be released.

    Note:
        If the ``filepath`` is a local path, just return itself and it will
        not be released (removed).

    Args:
        filepath (str or Path): Path to be read data.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: Only yield one path.

    Examples:
        >>> with get_local_path('s3://bucket/abc.jpg') as path:
        ...     # do something here
    """
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True)
    with backend.get_local_path(str(filepath)) as local_path:
        yield local_path


def list_dir_or_file(
    dir_path: Union[str, Path],
    list_dir: bool = True,
    list_file: bool = True,
    suffix: Optional[Union[str, Tuple[str]]] = None,
    recursive: bool = False,
    backend_args: Optional[dict] = None,
) -> Iterator[str]:
    """Scan a directory to find the interested directories or files in
    arbitrary order.

    Note:
        :meth:`list_dir_or_file` returns the path relative to ``dir_path``.

    Args:
        dir_path (str or Path): Path of the directory.
        list_dir (bool): List the directories. Defaults to True.
        list_file (bool): List the path of files. Defaults to True.
        suffix (str or tuple[str], optional): File suffix that we are
            interested in. Defaults to None.
        recursive (bool): If set to True, recursively scan the directory.
            Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Yields:
        Iterable[str]: A relative path to ``dir_path``.

    Examples:
        >>> dir_path = '/path/of/dir'
        >>> for file_path in list_dir_or_file(dir_path):
        ...     print(file_path)
        >>> # list those files and directories in current directory
        >>> for file_path in list_dir_or_file(dir_path):
        ...     print(file_path)
        >>> # only list files
        >>> for file_path in list_dir_or_file(dir_path, list_dir=False):
        ...     print(file_path)
        >>> # only list directories
        >>> for file_path in list_dir_or_file(dir_path, list_file=False):
        ...     print(file_path)
        >>> # only list files ending with specified suffixes
        >>> for file_path in list_dir_or_file(dir_path, suffix='.txt'):
        ...     print(file_path)
        >>> # list all files and directory recursively
        >>> for file_path in list_dir_or_file(dir_path, recursive=True):
        ...     print(file_path)
    """
    backend = get_file_backend(
        dir_path, backend_args=backend_args, enable_singleton=True)
    yield from backend.list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                        recursive)


def load(file, file_format=None, backend_args=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    ``load`` supports loading data from serialized files those can be storaged
    in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = file_handlers[file_format]
    if isinstance(file, str):
        file_backend = get_file_backend(file, backend_args=backend_args)
        if handler.str_like:
            with StringIO(file_backend.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(file_backend.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj
