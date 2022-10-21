import os.path as osp

from mmeval import JsonHandler


def test_json_handler():
    test_data_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
    json_handler = JsonHandler()
    content = json_handler.load_from_path(
        osp.join(test_data_dir, 'data/handler.json'))
    assert content == {'cat': 1, 'dog': 2, 'panda': 3}
