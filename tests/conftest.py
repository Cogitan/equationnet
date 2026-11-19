import platform
import sys
from pathlib import Path

# Some Windows machines hang in platform's WMI query (numpy/scipy/torch call it on import).
platform._wmi_query = lambda *a, **k: (_ for _ in ()).throw(OSError("wmi disabled"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
