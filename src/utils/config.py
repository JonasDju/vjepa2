# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os


def expand_env_vars(obj):
    """Recursively expand ``$VAR`` / ``${VAR}`` and a leading ``~`` in every string
    value of a nested dict/list structure (e.g. a parsed YAML config).

    Undefined variables are left untouched (``os.path.expandvars`` behavior), so a
    literal ``$`` that is not a valid reference is harmless. Non-string leaves are
    returned unchanged.
    """
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(expand_env_vars(v) for v in obj)
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj
