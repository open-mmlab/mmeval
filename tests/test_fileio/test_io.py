# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pytest
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import mmeval.fileio as fileio

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()

test_data_dir = Path(__file__).parent.parent / 'data'
text_path = test_data_dir / 'filelist.txt'
img_path = test_data_dir / 'color.jpg'
img_url = 'https://raw.githubusercontent.com/mmengine/tests/data/img.png'


@contextmanager
def build_temporary_directory():
    """Build a temporary directory containing many files to test
    ``FileClient.list_dir_or_file``.

    . \n
    | -- dir1 \n
    | -- | -- text3.txt \n
    | -- dir2 \n
    | -- | -- dir3 \n
    | -- | -- | -- text4.txt \n
    | -- | -- img.jpg \n
    | -- text1.txt \n
    | -- text2.txt \n
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        text1 = Path(tmp_dir) / 'text1.txt'
        text1.open('w').write('text1')
        text2 = Path(tmp_dir) / 'text2.txt'
        text2.open('w').write('text2')
        dir1 = Path(tmp_dir) / 'dir1'
        dir1.mkdir()
        text3 = dir1 / 'text3.txt'
        text3.open('w').write('text3')
        dir2 = Path(tmp_dir) / 'dir2'
        dir2.mkdir()
        jpg1 = dir2 / 'img.jpg'
        jpg1.open('wb').write(b'img')
        dir3 = dir2 / 'dir3'
        dir3.mkdir()
        text4 = dir3 / 'text4.txt'
        text4.open('w').write('text4')
        yield tmp_dir


def test_parse_uri_prefix():
    # input path is None
    with pytest.raises(AssertionError):
        fileio.io._parse_uri_prefix(None)

    # input path is list
    with pytest.raises(AssertionError):
        fileio.io._parse_uri_prefix([])

    # input path is Path object
    assert fileio.io._parse_uri_prefix(uri=text_path) == ''

    # input path starts with https
    assert fileio.io._parse_uri_prefix(uri=img_url) == 'https'

    # input path starts with s3
    uri = 's3://your_bucket/img.png'
    assert fileio.io._parse_uri_prefix(uri) == 's3'

    # input path starts with clusterName:s3
    uri = 'clusterName:s3://your_bucket/img.png'
    assert fileio.io._parse_uri_prefix(uri) == 's3'


def test_get_file_backend():
    # other unit tests may have added instances so clear them here.
    fileio.io.backend_instances = {}

    # uri should not be None when "backend" does not exist in backend_args
    with pytest.raises(ValueError, match='uri should not be None'):
        fileio.get_file_backend(None, backend_args=None)

    # uri is not None
    backend = fileio.get_file_backend(uri=text_path)
    assert isinstance(backend, fileio.backends.LocalBackend)

    uri = 'petrel://your_bucket/img.png'
    backend = fileio.get_file_backend(uri=uri)
    assert isinstance(backend, fileio.backends.PetrelBackend)

    backend = fileio.get_file_backend(uri=img_url)
    assert isinstance(backend, fileio.backends.HTTPBackend)
    uri = 'http://raw.githubusercontent.com/mmengine/tests/data/img.png'
    backend = fileio.get_file_backend(uri=uri)
    assert isinstance(backend, fileio.backends.HTTPBackend)

    # backend_args is not None and it contains a backend name
    backend_args = {'backend': 'local'}
    backend = fileio.get_file_backend(uri=None, backend_args=backend_args)
    assert isinstance(backend, fileio.backends.LocalBackend)

    backend_args = {'backend': 'petrel', 'enable_mc': True}
    backend = fileio.get_file_backend(uri=None, backend_args=backend_args)
    assert isinstance(backend, fileio.backends.PetrelBackend)

    # backend name has a higher priority
    backend_args = {'backend': 'http'}
    backend = fileio.get_file_backend(uri=text_path, backend_args=backend_args)
    assert isinstance(backend, fileio.backends.HTTPBackend)

    # test enable_singleton parameter
    assert len(fileio.io.backend_instances) == 0
    backend1 = fileio.get_file_backend(uri=text_path, enable_singleton=True)
    assert isinstance(backend1, fileio.backends.LocalBackend)
    assert len(fileio.io.backend_instances) == 1
    assert fileio.io.backend_instances[':{}'] is backend1

    backend2 = fileio.get_file_backend(uri=text_path, enable_singleton=True)
    assert isinstance(backend2, fileio.backends.LocalBackend)
    assert len(fileio.io.backend_instances) == 1
    assert backend2 is backend1

    backend3 = fileio.get_file_backend(uri=text_path, enable_singleton=False)
    assert isinstance(backend3, fileio.backends.LocalBackend)
    assert len(fileio.io.backend_instances) == 1
    assert backend3 is not backend2

    backend_args = {'path_mapping': {'src': 'dst'}, 'enable_mc': True}
    uri = 'petrel://your_bucket/img.png'
    backend4 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend4, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 2
    unique_key = 'petrel:{"path_mapping": {"src": "dst"}, "enable_mc": true}'
    assert fileio.io.backend_instances[unique_key] is backend4
    assert backend4 is not backend2

    uri = 'petrel://your_bucket/img1.png'
    backend5 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend5, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 2
    assert backend5 is backend4
    assert backend5 is not backend2

    backend_args = {'path_mapping': {'src1': 'dst1'}, 'enable_mc': True}
    backend6 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=True)
    assert isinstance(backend6, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 3
    unique_key = 'petrel:{"path_mapping": {"src1": "dst1"}, "enable_mc": true}'
    assert fileio.io.backend_instances[unique_key] is backend6
    assert backend6 is not backend4
    assert backend6 is not backend5

    backend7 = fileio.get_file_backend(
        uri=uri, backend_args=backend_args, enable_singleton=False)
    assert isinstance(backend7, fileio.backends.PetrelBackend)
    assert len(fileio.io.backend_instances) == 3
    assert backend7 is not backend6


def test_get():
    # test LocalBackend
    filepath = Path(img_path)
    img_bytes = fileio.get(filepath)
    assert filepath.open('rb').read() == img_bytes


def test_get_text():
    # test LocalBackend
    filepath = Path(text_path)
    text = fileio.get_text(filepath)
    assert filepath.open('r').read() == text


def test_exists():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert fileio.exists(tmp_dir)
        filepath = Path(tmp_dir) / 'test.txt'
        assert not fileio.exists(filepath)
        filepath.open('w').write('disk')
        assert fileio.exists(filepath)


def test_isdir():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert fileio.isdir(tmp_dir)
        filepath = Path(tmp_dir) / 'test.txt'
        filepath.open('w').write('disk')
        assert not fileio.isdir(filepath)


def test_isfile():
    # test LocalBackend
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert not fileio.isfile(tmp_dir)
        filepath = Path(tmp_dir) / 'test.txt'
        filepath.open('w').write('disk')
        assert fileio.isfile(filepath)


def test_join_path():
    # test LocalBackend
    filepath = fileio.join_path(test_data_dir, 'file')
    expected = osp.join(test_data_dir, 'file')
    assert filepath == expected

    filepath = fileio.join_path(test_data_dir, 'dir', 'file')
    expected = osp.join(test_data_dir, 'dir', 'file')
    assert filepath == expected


def test_get_local_path():
    # test LocalBackend
    with fileio.get_local_path(text_path) as filepath:
        assert str(text_path) == filepath


def test_list_dir_or_file():
    # test LocalBackend
    with build_temporary_directory() as tmp_dir:
        # list directories and files
        assert set(fileio.list_dir_or_file(tmp_dir)) == {
            'dir1', 'dir2', 'text1.txt', 'text2.txt'
        }

        # list directories and files recursively
        assert set(fileio.list_dir_or_file(tmp_dir, recursive=True)) == {
            'dir1',
            osp.join('dir1', 'text3.txt'), 'dir2',
            osp.join('dir2', 'dir3'),
            osp.join('dir2', 'dir3', 'text4.txt'),
            osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
        }

        # only list directories
        assert set(fileio.list_dir_or_file(
            tmp_dir, list_file=False)) == {'dir1', 'dir2'}

        with pytest.raises(
                TypeError,
                match='`suffix` should be None when `list_dir` is True'):
            list(
                fileio.list_dir_or_file(
                    tmp_dir, list_file=False, suffix='.txt'))

        # only list directories recursively
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_file=False,
                recursive=True)) == {'dir1', 'dir2',
                                     osp.join('dir2', 'dir3')}

        # only list files
        assert set(fileio.list_dir_or_file(
            tmp_dir, list_dir=False)) == {'text1.txt', 'text2.txt'}

        # only list files recursively
        assert set(
            fileio.list_dir_or_file(tmp_dir, list_dir=False,
                                    recursive=True)) == {
                                        osp.join('dir1', 'text3.txt'),
                                        osp.join('dir2', 'dir3', 'text4.txt'),
                                        osp.join('dir2', 'img.jpg'),
                                        'text1.txt', 'text2.txt'
                                    }

        # only list files ending with suffix
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_dir=False,
                suffix='.txt')) == {'text1.txt', 'text2.txt'}
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_dir=False,
                suffix=('.txt', '.jpg'))) == {'text1.txt', 'text2.txt'}

        with pytest.raises(
                TypeError,
                match='`suffix` must be a string or tuple of strings'):
            list(
                fileio.list_dir_or_file(
                    tmp_dir, list_dir=False, suffix=['.txt', '.jpg']))

        # only list files ending with suffix recursively
        assert set(
            fileio.list_dir_or_file(
                tmp_dir, list_dir=False, suffix='.txt', recursive=True)) == {
                    osp.join('dir1', 'text3.txt'),
                    osp.join('dir2', 'dir3', 'text4.txt'), 'text1.txt',
                    'text2.txt'
                }

        # only list files ending with suffix
        assert set(
            fileio.list_dir_or_file(
                tmp_dir,
                list_dir=False,
                suffix=('.txt', '.jpg'),
                recursive=True)) == {
                    osp.join('dir1', 'text3.txt'),
                    osp.join('dir2', 'dir3', 'text4.txt'),
                    osp.join('dir2', 'img.jpg'), 'text1.txt', 'text2.txt'
                }


def test_load():
    # invalid file format
    with pytest.raises(TypeError, match='Unsupported format'):
        fileio.load('filename.txt')

    # file is not a valid input
    with pytest.raises(
            TypeError, match='"file" must be a filepath str or a file-object'):
        fileio.load(123, file_format='json')

    test_data_dir = osp.dirname(osp.dirname(__file__))
    for filename in [
            'data/handler.json', 'data/handler.pkl', 'data/handler.yaml'
    ]:
        filepath = osp.join(test_data_dir, filename)
        content = fileio.load(filepath)
        assert content == {'cat': 1, 'dog': 2, 'panda': 3}
