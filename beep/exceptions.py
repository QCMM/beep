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