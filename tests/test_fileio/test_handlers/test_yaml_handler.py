import os.path as osp

from mmeval import YamlHandler


def test_yaml_handler():
    test_data_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))
    yaml_handler = YamlHandler()
    content = yaml_handler.load_from_path(
        osp.join(test_data_dir, 'data/handler.yaml'))
    assert content == {'cat': 1, 'dog': 2, 'panda': 3}
