# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
""" pyale: A Python implementation of activation likelihood estimation.
"""
from .dataset import import_neurosynth, import_sleuth
from .meta import ale, scale
from .version import __version__
