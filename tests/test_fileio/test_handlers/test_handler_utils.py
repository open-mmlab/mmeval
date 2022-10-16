import os.path as osp
import pytest

import mmeval


def test_register_handler():

    with pytest.raises(
            TypeError, match='handler must be a child of BaseFileHandler'):

        @mmeval.register_handler('txt')
        class TxtHandler1:

            def load_from_fileobj(self, file):
                return file.read()

    with pytest.raises(
            TypeError, match='file_formats must be a str or a list of str'):

        @mmeval.register_handler(123)
        class TxtHandler2(mmeval.BaseFileHandler):

            def load_from_fileobj(self, file):
                return file.read()

    @mmeval.register_handler('txt')
    class TxtHandler3(mmeval.BaseFileHandler):

        def load_from_fileobj(self, file):
            return file.read()

    test_data_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
    content = mmeval.load(osp.join(test_data_dir, 'data/filelist.txt'))
    assert content == '1.jpg\n2.jpg\n3.jpg\n4.jpg\n5.jpg'
