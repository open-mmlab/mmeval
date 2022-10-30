# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mmeval.fileio.backends import PetrelBackend
from mmeval.utils import has_method


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


sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()


class MockPetrelClient:

    def __init__(self, enable_mc=True, enable_multi_cluster=False):
        self.enable_mc = enable_mc
        self.enable_multi_cluster = enable_multi_cluster

    def Get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content

    def contains(self):
        pass

    def isdir(self):
        pass

    def list(self, dir_path):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                yield entry.name
            elif osp.isdir(entry.path):
                yield entry.name + '/'


@contextmanager
def delete_and_reset_method(obj, method):
    method_obj = deepcopy(getattr(type(obj), method))
    try:
        delattr(type(obj), method)
        yield
    finally:
        setattr(type(obj), method, method_obj)


@patch('petrel_client.client.Client', MockPetrelClient)
class TestPetrelBackend(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.img_path = cls.test_data_dir / 'color.jpg'
        cls.img_shape = (300, 400, 3)
        cls.text_path = cls.test_data_dir / 'filelist.txt'
        cls.petrel_dir = 'petrel://user/data'
        cls.petrel_path = f'{cls.petrel_dir}/test.jpg'
        cls.expected_dir = 's3://user/data'
        cls.expected_path = f'{cls.expected_dir}/test.jpg'

    def test_name(self):
        backend = PetrelBackend()
        self.assertEqual(backend.name, 'PetrelBackend')

    def test_map_path(self):
        backend = PetrelBackend(path_mapping=None)
        self.assertEqual(backend._map_path(self.petrel_path), self.petrel_path)

        backend = PetrelBackend(path_mapping={'data/': 'petrel://user/data/'})
        self.assertEqual(backend._map_path('data/test.jpg'), self.petrel_path)

    def test_format_path(self):
        backend = PetrelBackend()
        formatted_filepath = backend._format_path(
            'petrel://user\\data\\test.jpg')
        self.assertEqual(formatted_filepath, self.petrel_path)

    def test_replace_prefix(self):
        backend = PetrelBackend()
        self.assertEqual(
            backend._replace_prefix(self.petrel_path), self.expected_path)

    def test_join_path(self):
        backend = PetrelBackend()
        self.assertEqual(
            backend.join_path(self.petrel_dir, 'file'),
            f'{self.petrel_dir}/file')
        self.assertEqual(
            backend.join_path(f'{self.petrel_dir}/', 'file'),
            f'{self.petrel_dir}/file')
        self.assertEqual(
            backend.join_path(f'{self.petrel_dir}/', '/file'),
            f'{self.petrel_dir}/file')
        self.assertEqual(
            backend.join_path(self.petrel_dir, 'dir', 'file'),
            f'{self.petrel_dir}/dir/file')

    def test_get(self):
        backend = PetrelBackend()
        with patch.object(
                backend._client, 'Get', return_value=b'petrel') as patched_get:
            self.assertEqual(backend.get(self.petrel_path), b'petrel')
            patched_get.assert_called_once_with(self.expected_path)

    def test_get_text(self):
        backend = PetrelBackend()
        with patch.object(
                backend._client, 'Get', return_value=b'petrel') as patched_get:
            self.assertEqual(backend.get_text(self.petrel_path), 'petrel')
            patched_get.assert_called_once_with(self.expected_path)

    def test_exists(self):
        backend = PetrelBackend()
        self.assertTrue(has_method(backend._client, 'contains'))
        self.assertTrue(has_method(backend._client, 'isdir'))
        # raise Exception if `_client.contains` and '_client.isdir' are not
        # implemented
        with delete_and_reset_method(backend._client, 'contains'), \
                delete_and_reset_method(backend._client, 'isdir'):
            self.assertFalse(has_method(backend._client, 'contains'))
            self.assertFalse(has_method(backend._client, 'isdir'))
            with self.assertRaises(NotImplementedError):
                backend.exists(self.petrel_path)

        with patch.object(
                backend._client, 'contains',
                return_value=True) as patched_contains:
            self.assertTrue(backend.exists(self.petrel_path))
            patched_contains.assert_called_once_with(self.expected_path)

    def test_isdir(self):
        backend = PetrelBackend()
        self.assertTrue(has_method(backend._client, 'isdir'))
        # raise Exception if `_client.isdir` is not implemented
        with delete_and_reset_method(backend._client, 'isdir'):
            self.assertFalse(has_method(backend._client, 'isdir'))
            with self.assertRaises(NotImplementedError):
                backend.isdir(self.petrel_path)

        with patch.object(
                backend._client, 'isdir',
                return_value=True) as patched_contains:
            self.assertTrue(backend.isdir(self.petrel_path))
            patched_contains.assert_called_once_with(self.expected_path)

    def test_isfile(self):
        backend = PetrelBackend()
        self.assertTrue(has_method(backend._client, 'contains'))
        # raise Exception if `_client.contains` is not implemented
        with delete_and_reset_method(backend._client, 'contains'):
            self.assertFalse(has_method(backend._client, 'contains'))
            with self.assertRaises(NotImplementedError):
                backend.isfile(self.petrel_path)

        with patch.object(
                backend._client, 'contains',
                return_value=True) as patched_contains:
            self.assertTrue(backend.isfile(self.petrel_path))
            patched_contains.assert_called_once_with(self.expected_path)

    def test_get_local_path(self):
        backend = PetrelBackend()
        with patch.object(backend._client, 'Get',
                          return_value=b'petrel') as patched_get, \
            patch.object(backend._client, 'contains',
                         return_value=True) as patch_contains:
            with backend.get_local_path(self.petrel_path) as path:
                self.assertTrue(osp.isfile(path))
                self.assertEqual(Path(path).open('rb').read(), b'petrel')
            # exist the with block and path will be released
            self.assertFalse(osp.isfile(path))
            patched_get.assert_called_once_with(self.expected_path)
            patch_contains.assert_called_once_with(self.expected_path)

    def test_list_dir_or_file(self):
        backend = PetrelBackend()

        # raise Exception if `_client.list` is not implemented
        self.assertTrue(has_method(backend._client, 'list'))
        with delete_and_reset_method(backend._client, 'list'):
            self.assertFalse(has_method(backend._client, 'list'))
            with self.assertRaises(NotImplementedError):
                list(backend.list_dir_or_file(self.petrel_dir))

        with build_temporary_directory() as tmp_dir:
            # list directories and files
            self.assertEqual(
                set(backend.list_dir_or_file(tmp_dir)),
                {'dir1', 'dir2', 'text1.txt', 'text2.txt'})

            # list directories and files recursively
            self.assertEqual(
                set(backend.list_dir_or_file(tmp_dir, recursive=True)), {
                    'dir1', '/'.join(('dir1', 'text3.txt')), 'dir2', '/'.join(
                        ('dir2', 'dir3')), '/'.join(
                            ('dir2', 'dir3', 'text4.txt')), '/'.join(
                                ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

            # only list directories
            self.assertEqual(
                set(backend.list_dir_or_file(tmp_dir, list_file=False)),
                {'dir1', 'dir2'})
            with self.assertRaisesRegex(
                    TypeError,
                    '`list_dir` should be False when `suffix` is not None'):
                backend.list_dir_or_file(
                    tmp_dir, list_file=False, suffix='.txt')

            # only list directories recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        tmp_dir, list_file=False, recursive=True)),
                {'dir1', 'dir2', '/'.join(('dir2', 'dir3'))})

            # only list files
            self.assertEqual(
                set(backend.list_dir_or_file(tmp_dir, list_dir=False)),
                {'text1.txt', 'text2.txt'})

            # only list files recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        tmp_dir, list_dir=False, recursive=True)),
                {
                    '/'.join(('dir1', 'text3.txt')), '/'.join(
                        ('dir2', 'dir3', 'text4.txt')), '/'.join(
                            ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        tmp_dir, list_dir=False, suffix='.txt')),
                {'text1.txt', 'text2.txt'})
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        tmp_dir, list_dir=False, suffix=('.txt', '.jpg'))),
                {'text1.txt', 'text2.txt'})
            with self.assertRaisesRegex(
                    TypeError,
                    '`suffix` must be a string or tuple of strings'):
                backend.list_dir_or_file(
                    tmp_dir, list_dir=False, suffix=['.txt', '.jpg'])

            # only list files ending with suffix recursively
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        tmp_dir, list_dir=False, suffix='.txt',
                        recursive=True)), {
                            '/'.join(('dir1', 'text3.txt')), '/'.join(
                                ('dir2', 'dir3', 'text4.txt')), 'text1.txt',
                            'text2.txt'
                        })

            # only list files ending with suffix
            self.assertEqual(
                set(
                    backend.list_dir_or_file(
                        tmp_dir,
                        list_dir=False,
                        suffix=('.txt', '.jpg'),
                        recursive=True)),
                {
                    '/'.join(('dir1', 'text3.txt')), '/'.join(
                        ('dir2', 'dir3', 'text4.txt')), '/'.join(
                            ('dir2', 'img.jpg')), 'text1.txt', 'text2.txt'
                })

    def test_generate_presigned_url(self):
        pass
