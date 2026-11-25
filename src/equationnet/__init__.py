"""equationnet: type an equation; six simple models are trained on your data and ranked on held-out points.

    import equationnet as en
    result = en.fit("u_tt + c*u_t + k*u = 0", "data.csv")
    print(result.table())
    result.best.save("best_model.py")
"""

from .fit import Candidate, Result, YourModel, custom, fit, generate
from .models import MODELS, VERSION, NotApplicable
from .parse import Equation, EquationError, parse_equation

__version__ = VERSION
__all__ = ["fit", "generate", "custom", "YourModel", "parse_equation", "Result", "Candidate", "Equation",
           "EquationError", "NotApplicable", "MODELS", "__version__"]
