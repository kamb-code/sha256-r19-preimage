# Shim: this file name shadows the stdlib `enum` module when scripts run from this
# directory (the enumerator was renamed to enum_ctx.py).  Load the real stdlib enum.
import sys as _sys, importlib.util as _iu, sysconfig as _sc, os as _os
_spec = _iu.spec_from_file_location('enum', _os.path.join(_sc.get_paths()['stdlib'], 'enum.py'))
_mod = _iu.module_from_spec(_spec)
_sys.modules['enum'] = _mod
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith('__')})
