"""
Basin configuration and management classes for hydrological modeling.

This module provides Basin class to encapsulate basin information and configurations,
supporting both lumped and semi-distributed modeling approaches.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class BasinInfo:
    """
    Basic basin information data class.

    This class holds fundamental basin attributes that are used across
    different modeling approaches.
    """

    basin_id: str
    basin_name: str
    basin_area: float
    location: Optional[str] = None
    description: Optional[str] = None

    # Geographical information
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    elevation_mean: Optional[float] = None
    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None

    # Hydrological characteristics
    main_river_length: Optional[float] = None
    main_river_slope: Optional[float] = None
    drainage_density: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert basin info to dictionary."""
        return {
            "basin_id": self.basin_id,
            "basin_name": self.basin_name,
            "basin_area": self.basin_area,
            "location": self.location,
            "description": self.description,
            "centroid_lat": self.centroid_lat,
            "centroid_lon": self.centroid_lon,
            "elevation_mean": self.elevation_mean,
            "elevation_min": self.elevation_min,
            "elevation_max": self.elevation_max,
            "main_river_length": self.main_river_length,
            "main_river_slope": self.main_river_slope,
            "drainage_density": self.drainage_density,
        }


class Basin:
    """
    Comprehensive basin configuration class for hydrological modeling.

    This class encapsulates all basin-related information and provides interfaces
    for both lumped and semi-distributed modeling approaches. It serves as the
    central hub for basin data management and model configuration decisions.

    Design Philosophy:
    - Single source of truth for basin information
    - Support for both lumped and semi-distributed models
    - Extensible for future modeling approaches
    - Integration with frontend basin configuration
    """

    def __init__(
        self,
        basin_info: Union[BasinInfo, Dict[str, Any]],
        modeling_approach: str = "lumped",
        **kwargs,
    ):
        """
        Initialize basin configuration.

        Parameters
        ----------
        basin_info : BasinInfo or Dict[str, Any]
            Basic basin information
        modeling_approach : str, default "lumped"
            Modeling approach: "lumped" or "semi_distributed"
        **kwargs
            Additional configuration parameters
        """
        # Handle basin_info input
        if isinstance(basin_info, dict):
            self.basin_info = BasinInfo(**basin_info)
        else:
            self.basin_info = basin_info

        # Modeling configuration
        self.modeling_approach = modeling_approach

        # Additional configuration
        self.config = kwargs

        # Initialize approach-specific configurations
        self._setup_modeling_approach()

    def _setup_modeling_approach(self):
        """Setup configuration based on modeling approach."""
        if self.modeling_approach == "lumped":
            self._setup_lumped_config()
        elif self.modeling_approach == "semi_distributed":
            self._setup_semi_distributed_config()
        else:
            raise ValueError(
                f"Unsupported modeling approach: {self.modeling_approach}"
            )

    def _setup_lumped_config(self):
        """Setup configuration for lumped modeling."""
        # For lumped models, we use single basin area
        self.n_units = 1
        self.unit_areas = np.array([self.basin_info.basin_area])
        self.unit_ids = [self.basin_info.basin_id]

    def _setup_semi_distributed_config(self):
        """Setup configuration for semi-distributed modeling."""
        # For semi-distributed models, we would have multiple units
        # This is a placeholder for future implementation
        sub_basins = self.config.get("sub_basins", [])

        if sub_basins:
            self.n_units = len(sub_basins)
            self.unit_areas = np.array(
                [sub["basin_area"] for sub in sub_basins]
            )
            self.unit_ids = [sub["basin_id"] for sub in sub_basins]
        else:
            # Fallback to lumped if no sub-basins defined
            self._setup_lumped_config()

    @property
    def basin_area(self) -> float:
        """Get total basin area in km²."""
        return self.basin_info.basin_area

    @property
    def main_river_length(self) -> float:
        """Get main river length in km."""
        return self.basin_info.main_river_length

    @property
    def basin_id(self) -> str:
        """Get basin ID."""
        return self.basin_info.basin_id

    @property
    def basin_name(self) -> str:
        """Get basin name."""
        return self.basin_info.basin_name

    def is_lumped(self) -> bool:
        """Check if this is a lumped model configuration."""
        return self.modeling_approach == "lumped"

    def is_semi_distributed(self) -> bool:
        """Check if this is a semi-distributed model configuration."""
        return self.modeling_approach == "semi_distributed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert basin configuration to dictionary."""
        return {
            "basin_info": self.basin_info.to_dict(),
            "modeling_approach": self.modeling_approach,
            "n_units": self.n_units,
            "unit_areas": self.unit_areas.tolist(),
            "unit_ids": self.unit_ids,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Basin":
        """Create Basin instance from dictionary."""
        basin_info = BasinInfo(**data["basin_info"])

        return cls(
            basin_info=basin_info,
            modeling_approach=data.get("modeling_approach", "lumped"),
            **data.get("config", {}),
        )

    @classmethod
    def from_config(cls, basin_data: Dict[str, Any]) -> "Basin":
        """
        Create Basin instance from configuration data.

        This is a flexible factory method that can handle various data formats
        and field name variations commonly used in basin configuration.

        Parameters
        ----------
        basin_data : Dict[str, Any]
            Basin configuration data, supporting various field name formats:
            - basin_id/basin_code, basin_name/name, basin_area
            - output_unit, time_step_hours, modeling_approach
            - geographical and hydrological attributes

        Returns
        -------
        Basin
            Configured Basin instance
        """
        # Extract basic basin info
        basin_info = BasinInfo(
            basin_id=basin_data.get(
                "basin_id", basin_data.get("basin_code", "unknown")
            ),
            basin_name=basin_data.get(
                "basin_name", basin_data.get("name", "Unknown Basin")
            ),
            basin_area=basin_data.get("basin_area", 0.0),
            main_river_length=basin_data.get("main_river_length"),
            location=basin_data.get("location"),
            description=basin_data.get("description"),
        )

        # Extract modeling configuration
        modeling_approach = basin_data.get("modeling_approach", "lumped")

        # Handle additional configuration (excluding basin info and modeling approach)
        config = {
            k: v
            for k, v in basin_data.items()
            if k
            not in [
                "basin_id",
                "basin_code",
                "basin_name",
                "name",
                "basin_area",
                "main_river_length",
                "location",
                "description",
                "modeling_approach",
            ]
        }

        return cls(
            basin_info=basin_info,
            modeling_approach=modeling_approach,
            **config,
        )

    def __repr__(self) -> str:
        """String representation of Basin."""
        return (
            f"Basin(id='{self.basin_id}', name='{self.basin_name}', "
            f"basin_area={self.basin_area}km², "
            f"main_river_length={self.main_river_length}km, "
            f"approach='{self.modeling_approach}')"
        )


def create_basin_from_attributes(
    basin_attributes: Dict[str, Any], **kwargs
) -> Basin:
    """
    Convenience function to create Basin from basin attributes format.

    This function provides easy integration with existing basin attribute
    dictionaries.

    Parameters
    ----------
    basin_attributes : Dict[str, Any]
        Basin attributes dictionary with keys like basin_code, basin_name, basin_area, etc.
    **kwargs
        Additional configuration parameters

    Returns
    -------
    Basin
        Configured Basin instance
    """
    return Basin.from_config({**basin_attributes, **kwargs})
