# Copyright (c) OpenMMLab. All rights reserved.
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple, Union

from mmeval.utils import has_method
from .base import BaseStorageBackend


class PetrelBackend(BaseStorageBackend):
    """Petrel storage backend (for internal usage).

    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Defaults to None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Defaults to True.

    Examples:
        >>> backend = PetrelBackend()
        >>> filepath1 = 'petrel://path/of/file'
        >>> filepath2 = 'cluster-name:petrel://path/of/file'
        >>> backend.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = True):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def _replace_prefix(self, filepath: Union[str, Path]) -> str:
        filepath = str(filepath)
        return filepath.replace('petrel://', 's3://')

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        value = self._client.Get(filepath)
        return value

    def get_text(
        self,
        filepath: Union[str, Path],
        encoding: str = 'utf-8',
    ) -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        return str(self.get(filepath), encoding=encoding)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.exists(filepath)
            True
        """
        if not (has_method(self._client, 'contains')
                and has_method(self._client, 'isdir')):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` and `isdir` methods, please use a higher'
                'version or dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath) or self._client.isdir(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/dir'
            >>> backend.isdir(filepath)
            True
        """
        if not has_method(self._client, 'isdir'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `isdir` method, please use a higher version or dev'
                ' branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.isfile(filepath)
            True
        """
        if not has_method(self._client, 'contains'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `contains` method, please use a higher version or '
                'dev branch instead.')

        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        filepath = self._replace_prefix(filepath)
        return self._client.contains(filepath)

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.

        Examples:
            >>> backend = PetrelBackend()
            >>> filepath = 'petrel://path/of/file'
            >>> backend.join_path(filepath, 'another/path')
            'petrel://path/of/file/another/path'
            >>> backend.join_path(filepath, '/another/path')
            'petrel://path/of/file/another/path'
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
            formatted_paths.append(formatted_path.lstrip('/'))

        return '/'.join(formatted_paths)

    @contextmanager
    def get_local_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str or Path): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = PetrelBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> filepath = 'petrel://path/of/file'
            >>> with backend.get_local_path(filepath) as path:
            ...     # do something here
        """
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = PetrelBackend()
            >>> dir_path = 'petrel://path/of/dir'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path, list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path, list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path, suffix='.txt'):
            ...     print(file_path)
            >>> # list all files and directory recursively
            >>> for file_path in backend.list_dir_or_file(dir_path, recursive=True):
            ...     print(file_path)
        """  # noqa: E501
        if not has_method(self._client, 'list'):
            raise NotImplementedError(
                'Current version of Petrel Python SDK has not supported '
                'the `list` method, please use a higher version or dev'
                ' branch instead.')

        dir_path = self._map_path(dir_path)
        dir_path = self._format_path(dir_path)
        dir_path = self._replace_prefix(dir_path)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        if (suffix is not None) and not isinstance(suffix, (str, tuple)):
            raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not dir_path.endswith('/'):
            dir_path += '/'

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                              recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith('/'):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root):-1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(next_dir_path, list_dir,
                                                     list_file, suffix,
                                                     recursive)
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root):]
                    if (suffix is None
                            or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix,
                                 recursive)
