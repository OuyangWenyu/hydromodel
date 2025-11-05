r"""
Author: Wenyu Ouyang
Date: 2025-11-05
LastEditTime: 2025-11-05
LastEditors: Wenyu Ouyang
Description: Unified dataset mapping for all supported hydrological datasets
FilePath: /hydromodel/hydromodel/datasets/dataset_dict.py
Copyright (c) 2023-2026 Wenyu Ouyang. All rights reserved.
"""

from typing import Dict, Tuple, Optional

# Unified dataset mapping for all supported datasets
# Format: "data_type": ("module_path", "class_name", "dataset_category")
# dataset_category: "hydrodataset" for public datasets, "hydrodatasource" for custom datasets
DATASET_MAPPING: Dict[str, Tuple[str, str, str]] = {
    # ============================================================================
    # PUBLIC DATASETS - from hydrodataset
    # ============================================================================

    # CAMELS Series (16 datasets)
    # Catchment Attributes and Meteorology for Large-sample Studies
    "camels_aus": ("hydrodataset.camels_aus", "CamelsAus", "hydrodataset"),
    "camels_br": ("hydrodataset.camels_br", "CamelsBr", "hydrodataset"),
    "camels_ch": ("hydrodataset.camels_ch", "CamelsCh", "hydrodataset"),
    "camels_cl": ("hydrodataset.camels_cl", "CamelsCl", "hydrodataset"),
    "camels_col": ("hydrodataset.camels_col", "CamelsCol", "hydrodataset"),
    "camels_de": ("hydrodataset.camels_de", "CamelsDe", "hydrodataset"),
    "camels_deby": ("hydrodataset.camels_deby", "CamelsDeby", "hydrodataset"),
    "camels_dk": ("hydrodataset.camels_dk", "CamelsDk", "hydrodataset"),
    "camels_es": ("hydrodataset.camels_es", "CamelsEs", "hydrodataset"),
    "camels_fi": ("hydrodataset.camels_fi", "CamelsFi", "hydrodataset"),
    "camels_fr": ("hydrodataset.camels_fr", "CamelsFr", "hydrodataset"),
    "camels_gb": ("hydrodataset.camels_gb", "CamelsGb", "hydrodataset"),
    "camels_ind": ("hydrodataset.camels_ind", "CamelsInd", "hydrodataset"),
    "camels_lux": ("hydrodataset.camels_lux", "CamelsLux", "hydrodataset"),
    "camels_nz": ("hydrodataset.camels_nz", "CamelsNz", "hydrodataset"),
    "camels_se": ("hydrodataset.camels_se", "CamelsSe", "hydrodataset"),
    "camels_us": ("hydrodataset.camels_us", "CamelsUs", "hydrodataset"),

    # CAMELSH Series (2 datasets)
    # CAMELS extended with human impacts
    "camelsh": ("hydrodataset.camelsh", "Camelsh", "hydrodataset"),
    "camelsh_kr": ("hydrodataset.camelsh_kr", "CamelshKr", "hydrodataset"),

    # CARAVAN Series (3 datasets)
    # Large-sample dataset combining multiple CAMELS datasets
    "caravan": ("hydrodataset.caravan", "Caravan", "hydrodataset"),
    "caravan_dk": ("hydrodataset.caravan_dk", "CaravanDK", "hydrodataset"),
    "grdc_caravan": ("hydrodataset.grdc_caravan", "GrdcCaravan", "hydrodataset"),

    # LamaH Series (2 datasets)
    # Large-Sample Data for Hydrology and Environmental Sciences for Central Europe
    "lamah_ce": ("hydrodataset.lamah_ce", "LamahCe", "hydrodataset"),
    "lamah_ice": ("hydrodataset.lamah_ice", "LamahIce", "hydrodataset"),

    # Other Public Datasets (11 datasets)
    "hysets": ("hydrodataset.hysets", "Hysets", "hydrodataset"),  # Canadian dataset
    "mopex": ("hydrodataset.mopex", "Mopex", "hydrodataset"),  # US dataset
    "bull": ("hydrodataset.bull", "BULL", "hydrodataset"),  # French dataset
    "estreams": ("hydrodataset.estreams", "Estreams", "hydrodataset"),  # European dataset
    "hype": ("hydrodataset.hype", "Hype", "hydrodataset"),  # European hydrological model dataset
    "jialing": ("hydrodataset.jialingriverchina", "jialingriverchina", "hydrodataset"),  # Chinese regional dataset
    "simbi": ("hydrodataset.simbi", "simbi", "hydrodataset"),  # Brazilian dataset
    "waterbenchiowa": ("hydrodataset.waterbenchiowa", "waterbenchiowa", "hydrodataset"),  # Iowa, US
    "hyd_responses": ("hydrodataset.hyd_responses", "HydResponses", "hydrodataset"),  # Hydrological responses dataset

    # ============================================================================
    # CUSTOM DATASETS - from hydrodatasource
    # ============================================================================

    # Self-made hydro datasets
    # These require custom data directory structure
    "floodevent": ("hydrodatasource.reader.data_source", "SelfMadeHydroDataset", "hydrodatasource"),
    "selfmadehydrodataset": ("hydrodatasource.reader.data_source", "SelfMadeHydroDataset", "hydrodatasource"),
}


def get_supported_datasets(category: Optional[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """
    Get list of supported datasets.

    Parameters
    ----------
    category : str, optional
        Filter by category: "hydrodataset" or "hydrodatasource"
        If None, returns all datasets

    Returns
    -------
    Dict[str, Tuple[str, str, str]]
        Dictionary of dataset_name: (module_path, class_name, category)

    Examples
    --------
    >>> # Get all supported datasets
    >>> all_datasets = get_supported_datasets()
    >>> print(list(all_datasets.keys()))

    >>> # Get only public datasets from hydrodataset
    >>> public_datasets = get_supported_datasets(category="hydrodataset")

    >>> # Get only custom datasets from hydrodatasource
    >>> custom_datasets = get_supported_datasets(category="hydrodatasource")
    """
    if category is None:
        return DATASET_MAPPING.copy()
    else:
        return {
            name: info
            for name, info in DATASET_MAPPING.items()
            if info[2] == category
        }


def get_dataset_info(dataset_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Get information about a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    Optional[Tuple[str, str, str]]
        Tuple of (module_path, class_name, category) if dataset exists, None otherwise

    Examples
    --------
    >>> info = get_dataset_info("camels_us")
    >>> if info:
    ...     module_path, class_name, category = info
    ...     print(f"{dataset_name} is from {category}")
    """
    return DATASET_MAPPING.get(dataset_name)


def is_dataset_supported(dataset_name: str) -> bool:
    """
    Check if a dataset is supported.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    bool
        True if dataset is supported, False otherwise

    Examples
    --------
    >>> if is_dataset_supported("camels_us"):
    ...     print("CAMELS US is supported!")
    """
    return dataset_name in DATASET_MAPPING


def get_dataset_category(dataset_name: str) -> Optional[str]:
    """
    Get the category of a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    Optional[str]
        Category ("hydrodataset" or "hydrodatasource") if dataset exists, None otherwise

    Examples
    --------
    >>> category = get_dataset_category("camels_us")
    >>> print(f"CAMELS US is a {category} dataset")
    """
    info = DATASET_MAPPING.get(dataset_name)
    return info[2] if info else None
