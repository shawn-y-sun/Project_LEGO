# TECHNIC/mev.py

import os
import pandas as pd
from typing import Dict, Any, Tuple, Callable

# External default preprocessing function for MEV tables
default_load_and_preprocess: Callable[[str, str], Tuple[pd.DataFrame, Dict[str, str]]]  # type: ignore

def default_load_and_preprocess(workbook: str, sheet: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw = pd.read_excel(workbook, sheet_name=sheet)
    df = raw.copy()
    df_mev_prc = df.iloc[1:]
    df_mev_prc.columns = df_mev_prc.iloc[0]

    df_mev_prc.index = df_mev_prc.iloc[:, 0]
    df_mev_prc = df_mev_prc.iloc[1:, 1:]
    df_mev_prc = df_mev_prc.loc[:, df_mev_prc.columns.notna()].iloc[2:]

    mev_name_lst = df_mev_prc.columns.tolist()
    mev_name_lst = [mev.replace('Canada\n', '') for mev in mev_name_lst][1:]

    df_mev_prc.columns = df_mev_prc.iloc[0]
    df_mev_prc = df_mev_prc.iloc[1:, :]

    # Reformat index to timestamps at period end
    df_mev_prc.index = [i.replace(':', 'Q') for i in df_mev_prc.index]
    df_mev_prc.index = pd.PeriodIndex(df_mev_prc.index, freq='Q').to_timestamp(how='end')
    df_mev_prc.index = pd.to_datetime(df_mev_prc.index).normalize()

    mev_code_lst = df_mev_prc.columns.tolist()[1:]
    assert len(mev_name_lst) == len(mev_code_lst), \
        "Mismatch between code and name lists"
    mev_dict = dict(zip(mev_code_lst, mev_name_lst))

    return df_mev_prc, mev_dict

class MEVLoader:
    """
    Loader for Macro Economic Variables from Excel workbooks.

    - model_mev_source: dict with a single key (workbook path) and value (sheet name) for base MEV.
      Example: {"model_mev.xlsx": "ModelSheet"}

    - scen_mevs_source: dict mapping each workbook path to a dict of scenario names to sheet names.
      Example:
        {
            "scen_workbook1.xlsx": {"base": "BaseSheet", "adv": "AdverseSheet", "sev": "SevereSheet"},
            "scen_workbook2.xlsx": {"base": "Base2", "adv": "Adverse2", "sev": "Severe2"}
        }

    :param model_mev_source: single-entry mapping of workbook->sheet for base MEV.
    :param scen_mevs_source: mapping of workbook->(mapping of scenario->sheet).
    :param load_and_preprocess: function to load and preprocess each worksheet (workbook, sheet) -> (df, mapping).
    """
    def __init__(
        self,
        model_mev_source: Dict[str, str],
        scen_mevs_source: Dict[str, Dict[str, str]],
        load_and_preprocess: Callable[[str, str], Tuple[pd.DataFrame, Dict[str, str]]] = default_load_and_preprocess
    ):
        # Validate model_mev_source has exactly one entry
        if len(model_mev_source) != 1:
            raise ValueError("model_mev_source must contain exactly one workbook:sheet mapping.")
        self.model_mev_source = model_mev_source

        # Validate scen_mevs_source structure
        if not isinstance(scen_mevs_source, dict) or not all(
            isinstance(v, dict) for v in scen_mevs_source.values()
        ):
            raise ValueError(
                "scen_mevs_source must be a dict mapping workbook->(dict of scenario->sheet)."
            )
        self.scen_mevs_source = scen_mevs_source

        # Store preprocessing function
        self._preprocess_fn = load_and_preprocess

        # placeholders for loaded data
        self._model_mev: pd.DataFrame = None  # base MEV
        self._model_map: Dict[str, str] = {}
        self._scen_mevs: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._scen_maps: Dict[str, Dict[str, Dict[str, str]]] = {}

    def load(self) -> None:
        """
        Load and preprocess both base MEV and all scenario MEVs,
        storing results internally.
        """
        # load base model MEV
        workbook, sheet = next(iter(self.model_mev_source.items()))
        df, mapping = self._preprocess_fn(workbook, sheet)
        self._model_mev = df
        self._model_map = mapping

        # load scenario MEVs per workbook set
        for workbook, sheet_map in self.scen_mevs_source.items():
            key = os.path.splitext(os.path.basename(workbook))[0]
            df_dict: Dict[str, pd.DataFrame] = {}
            map_dict: Dict[str, Dict[str, str]] = {}
            for scenario, sheet in sheet_map.items():
                df_s, mapping_s = self._preprocess_fn(workbook, sheet)
                df_dict[scenario] = df_s
                map_dict[scenario] = mapping_s
            self._scen_mevs[key] = df_dict
            self._scen_maps[key] = map_dict

    @property
    def model_mev(self) -> pd.DataFrame:
        if self._model_mev is None:
            raise ValueError("Model MEV not loaded. Call load() first.")
        return self._model_mev

    @property
    def scen_mevs(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        return self._scen_mevs

    @property
    def model_map(self) -> Dict[str, str]:
        return self._model_map

    @property
    def scen_maps(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        return self._scen_maps

    def apply_to_all(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """
        Apply a transformation function to all loaded MEV DataFrames.
        """
        if self._model_mev is not None:
            self._model_mev = func(self._model_mev.copy())
        for key, df_dict in self._scen_mevs.items():
            for scen, df in df_dict.items():
                self._scen_mevs[key][scen] = func(df.copy())
