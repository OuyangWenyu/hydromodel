import pytest
from hydromodel.trainers.calibrate_ga import calibrate_by_ga
from hydromodel.trainers.calibrate_sceua import calibrate_by_sceua


def test_calibrate_xaj_sceua(basins, p_and_e, qobs, warmup_length, tmp_path):
    # hyper-params are minimal to keep the test fast
    calibrate_by_sceua(
        basins,
        p_and_e,
        qobs,
        str(tmp_path / "sceua_xaj"),
        warmup_length,
        model={
            "name": "xaj_mz",
            "source_type": "sources",
            "source_book": "HF",
            "kernel_size": 15,
            "time_interval_hours": 24,
        },
        algorithm={
            "name": "SCE_UA",
            "random_seed": 1234,
            "rep": 5,
            "ngs": 7,
            "kstop": 3,
            "peps": 0.1,
            "pcento": 0.1,
        },
        loss={
            "type": "time_series",
            "obj_func": "RMSE",
            "events": None,
        },
    )


def test_calibrate_xaj_ga(p_and_e, qobs, warmup_length, tmp_path):
    calibrate_by_ga(
        p_and_e,
        qobs,
        deap_dir=str(tmp_path / "ga_xaj"),
        warmup_length=warmup_length,
    )
