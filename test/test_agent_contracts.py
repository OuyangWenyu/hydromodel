from types import SimpleNamespace

import numpy as np
import pytest
import spotpy.objectivefunctions as spotpy_obj
import yaml

from hydromodel.models.model_config import (
    MODEL_PARAM_DICT,
    ParamRangeConfigError,
    attach_parameter_contract,
    denormalize_parameter_dict,
    inject_model_param_config,
    resolve_model_param_config,
)
from hydromodel.models.model_dict import (
    LOSS_DICT,
    describe_model,
    list_losses,
    list_models,
    resolve_loss_config,
)
from hydromodel.configs.config_manager import validate_config


def test_validate_config_requires_dataset():
    """A config without data_cfgs.dataset must fail validation."""
    config = {
        "data_cfgs": {"basin_ids": ["b1"]},  # missing dataset
        "model_cfgs": {"name": "xaj"},
        "training_cfgs": {"algorithm": "SCE_UA", "loss": "RMSE"},
    }
    result = validate_config(config)
    assert result["valid"] is False
    assert any("dataset" in err for err in result["errors"])


def test_validate_config_accepts_dataset():
    """A config with data_cfgs.dataset passes the dataset requirement."""
    config = {
        "data_cfgs": {"dataset": "camels_us", "basin_ids": ["b1"]},
        "model_cfgs": {"name": "xaj"},
        "training_cfgs": {"algorithm": "SCE_UA", "loss": "RMSE"},
    }
    result = validate_config(config)
    assert not any("dataset" in err for err in result["errors"])


def test_simulate_is_exported():
    """The documented top-level simulate(config) interface must exist."""
    import hydromodel

    assert callable(hydromodel.simulate)


def test_simulate_requires_specific_parameters():
    """Simulation needs concrete parameter values under model_cfgs.parameters."""
    import hydromodel

    config = {
        "data_cfgs": {"dataset": "camels_us", "basin_ids": ["01013500"]},
        "model_cfgs": {"name": "xaj", "params": {"source_type": "sources"}},
    }
    with pytest.raises(ValueError, match="parameters"):
        hydromodel.simulate(config)


def test_resolve_loss_config_maps_user_objectives_to_minimized_keys():
    nse = resolve_loss_config({"type": "time_series", "obj_func": "NSE"})
    kge = resolve_loss_config({"type": "time_series", "obj_func": "KGE"})

    assert nse["requested_obj_func"] == "NSE"
    assert nse["resolved_obj_func"] == "neg_nashsutcliffe"
    assert nse["obj_func"] == "neg_nashsutcliffe"
    assert kge["resolved_obj_func"] == "neg_kge"


def test_neg_kge_matches_flattened_spotpy_kge():
    obs = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
    sim = obs * 0.9

    expected = -spotpy_obj.kge(obs.reshape(-1), sim.reshape(-1))
    assert LOSS_DICT["neg_kge"](obs, sim) == pytest.approx(expected)


def test_spotpy_objectives_accept_3d_inputs():
    obs = np.array([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])
    sim = obs * 0.9

    assert np.isfinite(LOSS_DICT["spotpy_kge"](obs, sim))
    assert np.isfinite(LOSS_DICT["spotpy_nashsutcliffe"](obs, sim))
    assert np.isfinite(LOSS_DICT["spotpy_lognashsutcliffe"](obs, sim))


def test_direct_higher_is_better_spotpy_objective_warns():
    with pytest.warns(RuntimeWarning, match="higher-is-better"):
        resolved = resolve_loss_config(
            {"type": "time_series", "obj_func": "spotpy_kge"}
        )

    assert resolved["obj_func"] == "spotpy_kge"


def test_default_param_range_is_marked_and_warns():
    with pytest.warns(RuntimeWarning, match="param_range_file not provided"):
        resolved = resolve_model_param_config("gr4j")

    assert resolved["source"] == "default"
    assert resolved["source_path"] is None
    assert (
        resolved["model_param_config"]["param_name"]
        == MODEL_PARAM_DICT["gr4j"]["param_name"]
    )


def test_explicit_missing_param_range_file_fails(tmp_path):
    missing = tmp_path / "missing.yaml"

    with pytest.raises(ParamRangeConfigError):
        resolve_model_param_config(
            "gr4j", param_range_file=missing, strict=True
        )


def test_param_range_reorders_by_param_name(tmp_path):
    names = MODEL_PARAM_DICT["gr4j"]["param_name"]
    ranges = {
        name: [float(i), float(i + 1)]
        for i, name in enumerate(reversed(names), start=1)
    }
    path = tmp_path / "param_range.yaml"
    path.write_text(
        yaml.safe_dump(
            {"gr4j": {"param_name": names, "param_range": ranges}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    resolved = resolve_model_param_config(
        "gr4j", param_range_file=path, strict=True
    )

    assert list(resolved["model_param_config"]["param_range"].keys()) == names
    assert resolved["source"] == "explicit"


def test_param_range_missing_extra_and_invalid_ranges_fail(tmp_path):
    names = MODEL_PARAM_DICT["gr4j"]["param_name"]
    valid_ranges = {name: [0.0, 1.0] for name in names}

    cases = [
        {
            "param_name": names,
            "param_range": dict(list(valid_ranges.items())[1:]),
        },
        {
            "param_name": names,
            "param_range": {**valid_ranges, "EXTRA": [0.0, 1.0]},
        },
        {
            "param_name": names,
            "param_range": {**valid_ranges, names[0]: [1.0, 1.0]},
        },
    ]

    for index, case in enumerate(cases):
        path = tmp_path / f"bad_{index}.yaml"
        path.write_text(
            yaml.safe_dump({"gr4j": case}, sort_keys=False),
            encoding="utf-8",
        )
        with pytest.raises(ParamRangeConfigError):
            resolve_model_param_config(
                "gr4j", param_range_file=path, strict=True
            )


def test_injected_param_contract_supports_explicit_and_legacy_fields():
    with pytest.warns(RuntimeWarning):
        resolved = resolve_model_param_config("gr4j")
    injected = inject_model_param_config(
        {"source_type": "test"},
        "gr4j",
        resolved["model_param_config"],
    )

    assert injected["gr4j"] == resolved["model_param_config"]
    assert injected["param_config"]["gr4j"] == resolved["model_param_config"]
    assert (
        injected["param_name"] == resolved["model_param_config"]["param_name"]
    )
    assert (
        injected["param_range"]
        == resolved["model_param_config"]["param_range"]
    )


def test_calibration_result_contract_preserves_legacy_best_params():
    with pytest.warns(RuntimeWarning):
        resolved = resolve_model_param_config("gr4j")
    names = resolved["model_param_config"]["param_name"]
    normalized = {name: 0.5 for name in names}
    setup = SimpleNamespace(
        model_name="gr4j",
        parameter_names=names,
        model_param_config=resolved["model_param_config"],
        param_range_source="default",
        param_range_source_path=None,
        loss_config=resolve_loss_config(
            {"type": "time_series", "obj_func": "KGE"}
        ),
    )

    result = attach_parameter_contract(
        {
            "objective_value": np.float64(1.0),
            "best_params": {"gr4j": normalized.copy()},
        },
        setup,
    )
    expected_denorm = denormalize_parameter_dict(
        normalized, resolved["model_param_config"]
    )

    assert result["best_params"] == result["best_params_normalized"]
    assert result["parameter_format"] == "normalized"
    assert result["best_params_denormalized"]["gr4j"] == expected_denorm
    assert result["param_range_source"] == "default"
    assert result["loss_config"]["requested_obj_func"] == "KGE"
    assert result["loss_config"]["resolved_obj_func"] == "neg_kge"


def test_public_introspection_helpers_expose_contracts():
    assert "gr4j" in list_models()
    assert "KGE" in list_losses()["user_objectives"]
    assert describe_model("gr4j")["parameters"]["param_name"]
