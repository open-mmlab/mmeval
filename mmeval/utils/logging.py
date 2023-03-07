# Copyright (c) OpenMMLab. All rights reserved.
import logging

DEFAULT_LOGGER = logging.getLogger('mmeval')
DEFAULT_LOGGER.setLevel(logging.INFO)
DEFAULT_LOGGER.addHandler(logging.StreamHandler())
