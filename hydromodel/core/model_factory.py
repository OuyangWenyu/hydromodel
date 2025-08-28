from typing import Dict, Any
from traditional_model import TraditionalModel

# We need to handle the case where torch is not installed
try:
    from hydromodel.models.torch_model import PytorchModel

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def model_factory(model_config: Dict[str, Any], basin_config: Any = None) -> Any:
    """
    Factory function to instantiate a model based on its type.

    Args:
        model_config (Dict[str, Any]): The model configuration.
            It must contain a 'type' key, e.g., 'traditional' or 'pytorch'.
        basin_config (Any, optional): The basin configuration.

    Returns:
        An instance of a model wrapper (e.g., TraditionalModel, PytorchModel).
    """
    model_type = model_config.get("type", "traditional")  # Default to traditional for backward compatibility

    if model_type == "pytorch":
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please install it to use PyTorch models.")
        # Note: The PytorchModel class needs to be defined and accessible.
        # This assumes it's in the specified path.
        return PytorchModel(model_config, basin_config)

    elif model_type == "traditional":
        return TraditionalModel(model_config, basin_config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
