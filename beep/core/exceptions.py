"""
Exceptions for BEEP
"""


class MoleculeNotSetError(Exception):
    """Error when molecule is not set."""
    
    def __init__(self, message = "Molecule is not set yet. Please use set_molecule(mol) method to set the desired molecule."
):
        self.message = message
        super().__init__(self.message)
        
class OptimizationMethodNotSetError(Exception):
    """Error when Optimization Method is not set."""
    
    def __init__(self, message = "Optimization Method is not set yet. Please use get_optimization_methods to list the available methods."
        ):
        self.message = message
        super().__init__(self.message)
        
        
        
class DataNotLoadedError(Exception):
    """Error when binding energy data is not loaded."""
    
    def __init__(self, message = "Binding energy data is not loaded yet. Please use load_data method to load the data for the desired molecule."
        ):
        self.message = message
        super().__init__(self.message)

# ---------------------------------------------------------------------------
# MBE (Many-Body Expansion) workflow exceptions
#
# Ported from the standalone beep-mbe package. Its ``ConfigError`` has no
# analogue here because config validation is handled by Pydantic
# (``pydantic.ValidationError``); the remaining errors map to the
# ``mbe`` / ``mbe_extract`` workflows.
# ---------------------------------------------------------------------------


class MbeError(Exception):
    """Base class for errors raised by the MBE workflows."""


class MbeFragmentationError(MbeError):
    """Raised when a molecule cannot be fragmented for the MBE."""


class MbeSubmissionError(MbeError):
    """Raised when MBE submission or monitoring fails."""


class MbeRecordMissingError(MbeError):
    """Raised when an expected optimization record or molecule is missing."""


class MbeExtractError(MbeError):
    """Raised when MBE binding energies cannot be assembled."""
