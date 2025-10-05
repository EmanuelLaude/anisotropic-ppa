# Route `import compot...` to third_party/compot/compot
import pathlib
_pkg = pathlib.Path(__file__).resolve().parents[1] / "third_party" / "compot" / "compot"
__path__ = [str(_pkg)]          # make subpackages (calculus, optimizer, ...) resolve
from third_party.compot.compot import *  # re-export top-level API
