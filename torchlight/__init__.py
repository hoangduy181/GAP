# Re-export from the inner torchlight package
# This allows 'from torchlight import DictAction' to work when torchlight is not installed as a package
import os

# Check if we're in the torchlight directory structure (not installed)
_torchlight_inner_path = os.path.join(os.path.dirname(__file__), 'torchlight')
if os.path.exists(_torchlight_inner_path) and os.path.isdir(_torchlight_inner_path):
    # Not installed - import from inner package using relative import
    from .torchlight.util import IO, DictAction, str2bool, str2dict, import_class
    from .torchlight.gpu import visible_gpu, occupy_gpu, ngpu
else:
    # Installed - import directly (this path won't be used when not installed)
    from .util import IO, DictAction, str2bool, str2dict, import_class
    from .gpu import visible_gpu, occupy_gpu, ngpu

__all__ = ['IO', 'DictAction', 'str2bool', 'str2dict', 'import_class', 'visible_gpu', 'occupy_gpu', 'ngpu']

