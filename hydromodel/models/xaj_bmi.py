from typing import Tuple
from bmipy import Bmi
import numpy as np

import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

PRECISION = 1e-5


class xajBmi(Bmi):
    """Empty model wrapped in a BMI interface."""

    name = "hydro-model-xaj"
    input_var_names = ("precipitation", "ETp")
    output_var_names = ("ET", "discharge")
    var_units = {
        "precipitation": "mm/day",
        "ETp": "mm/day",
        "discharge": "mm/day",
        "ET": "mm/day",
    }

    def __init__(self):
        """Create a BmiHeat model that is ready for initialization."""
        self.time_step = 0

    def initialize(self, config_file):
        try:
            logger.info("xaj: initialize_model")
            config = configuration.read_config(config_file)
            forcing_data = pd.read_csv(config["forcing_file"])
            p_and_e_df, p_and_e = configuration.extract_forcing(forcing_data)
            p_and_e_warmup = p_and_e[0 : config["warmup_length"], :, :]
            params = np.tile([0.5], (1, 15))
            (
                self.q_sim_state,
                self.es_state,
                self.w0,
                self.w1,
                self.w2,
                self.s0,
                self.fr0,
                self.qi0,
                self.qg0,
            ) = configuration.warmup(p_and_e_warmup, params, config["warmup_length"])

            self._start_time_str, self._end_time_str, self._time_units = (
                configuration.get_time_config(config)
            )

            self.params = params
            self.warmup_length = config["warmup_length"]
            self.p_and_e_df = p_and_e_df
            self.p_and_e = p_and_e
            self.config = config
            self.basin_area = config["basin_area"]

        except:
            import traceback

            traceback.print_exc()
            raise

    def update(self):
        """Update model for a single time step."""

        self.time_step += 1
        # p_and_e_sim = self.p_and_e[self.warmup_length+1:self.time_step+self.warmup_length+1]
        p_and_e_sim = self.p_and_e[1 : self.time_step + 1]
        self.runoff_im, self.rss_, self.ris_, self.rgs_, self.es_runoff, self.rss = (
            xaj_runoff(
                p_and_e_sim,
                w0=self.w0,
                s0=self.s0,
                fr0=self.fr0,
                params_runoff=self.params,
                return_state=False,
            )
        )
        if self.time_step + self.warmup_length + 1 >= self.p_and_e.shape[0]:
            q_sim, es = xaj_route(
                p_and_e_sim,
                params_route=self.params,
                model_name="xaj",
                runoff_im=self.runoff_im,
                rss_=self.rss_,
                ris_=self.ris_,
                rgs_=self.rgs_,
                rss=self.rss,
                qi0=self.qi0,
                qg0=self.qg0,
                es=self.es_runoff,
            )
            self.p_sim = p_and_e_sim[:, :, 0]
            self.e_sim = p_and_e_sim[:, :, 1]
            q_sim = convert_unit(
                q_sim,
                unit_now="mm/day",
                unit_final=unit["streamflow"],
                basin_area=float(self.basin_area),
            )
            self.q_sim = q_sim
            self.es = es

    def update_until(self, time):
        while self.get_current_time() + 0.001 < time:
            self.update()

    def finalize(self) -> None:
        """Finalize model."""
        self.model = None

    def get_component_name(self) -> str:
        return "xaj"

    def get_input_item_count(self) -> int:
        return len(self.input_var_names)

    def get_output_item_count(self) -> int:
        return len(self.output_var_names)

    def get_input_var_names(self) -> Tuple[str]:
        return self.input_var_names

    def get_output_var_names(self) -> Tuple[str]:
        return self.output_var_names

    def get_var_grid(self, name: str) -> int:
        raise NotImplementedError()

    def get_var_type(self, name: str) -> str:
        return "float64"

    def get_var_units(self, name: str) -> str:
        return self.var_units[name]

    def get_var_itemsize(self, name: str) -> int:
        return np.dtype(self.get_var_type(name)).itemsize

    def get_var_nbytes(self, name: str) -> int:
        return self.get_value_ptr(name).nbytes

    def get_var_location(self, name: str) -> str:
        raise NotImplementedError()

    def get_start_time(self):
        return self.start_Time(self._start_time_str)

    def get_current_time(self):
        # return self.start_Time(self._start_time_str) + datetime.timedelta(self.time_step+self.warmup_length)
        if self._time_units == "hours":
            time_step = datetime.timedelta(hours=self.time_step)
        elif self._time_units == "days":
            time_step = datetime.timedelta(days=self.time_step)
        return self.start_Time(self._start_time_str) + time_step

    def get_end_time(self):
        return self.end_Time(self._end_time_str)

    def get_time_units(self) -> str:
        return self._time_units

    def get_time_step(self) -> float:
        return 1

    def get_value(self, name: str) -> None:
        logger.info("getting value for var %s", name)
        return self.get_value_ptr(name).flatten()

    def get_value_ptr(self, name: str) -> np.ndarray:
        if name == "discharge":
            return self.q_sim
        elif name == "ET":
            return self.es

    def get_value_at_indices(self, name: str, inds: np.ndarray) -> np.ndarray:

        return self.get_value_ptr(name).take(inds)

    def set_value(self, name: str, src: np.ndarray):

        val = self.get_value_ptr(name)
        val[:] = src.reshape(val.shape)

    def set_value_at_indices(
        self, name: str, inds: np.ndarray, src: np.ndarray
    ) -> None:
        val = self.get_value_ptr(name)
        val.flat[inds] = src

    # Grid information
    def get_grid_rank(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_size(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_type(self, grid: int) -> str:
        raise NotImplementedError()

    # Uniform rectilinear
    def get_grid_shape(self, grid: int, shape: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_spacing(self, grid: int, spacing: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_origin(self, grid: int, origin: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    # Non-uniform rectilinear, curvilinear
    def get_grid_x(self, grid: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_y(self, grid: int, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_z(self, grid: int, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_node_count(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_edge_count(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_face_count(self, grid: int) -> int:
        raise NotImplementedError()

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_grid_nodes_per_face(
        self, grid: int, nodes_per_face: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    def start_Time(self, _start_time_str):

        try:
            if " " in _start_time_str:
                date, time = _start_time_str.split(" ")
            else:
                date = _start_time_str
                time = None
            year, month, day = date.split("-")
            self._startTime = datetime.date(int(year), int(month), int(day))

            if time:
                hour, minute, second = time.split(":")
                # self._startTime = self._startTime.replace(hour=int(hour),
                #                                       minute=int(minute),
                #                                       second=int(second))
                self._startTime = datetime.datetime(
                    int(year), int(month), int(day), int(hour), int(minute), int(second)
                )
        except ValueError:
            raise ValueError("Invalid start date format!")

        return self._startTime

    def end_Time(self, _end_time_str):

        try:
            if " " in _end_time_str:
                date, time = _end_time_str.split(" ")
            else:
                date = _end_time_str
                time = None
            year, month, day = date.split("-")
            self._endTime = datetime.date(int(year), int(month), int(day))

            if time:
                hour, minute, second = time.split(":")
                self._endTime = datetime.datetime(
                    int(year), int(month), int(day), int(hour), int(minute), int(second)
                )
        except ValueError:
            raise ValueError("Invalid start date format!")
        return self._endTime
