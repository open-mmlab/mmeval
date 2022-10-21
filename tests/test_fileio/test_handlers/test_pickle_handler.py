import os.path as osp

from mmeval import PickleHandler


def test_pickle_handler():
    test_data_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
    pickle_handler = PickleHandler()
    content = pickle_handler.load_from_path(
        osp.join(test_data_dir, 'data/handler.pkl'))
    assert content == {'cat': 1, 'dog': 2, 'panda': 3}
