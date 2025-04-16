# TECHNIC/mev.py
import pandas as pd
from typing import List, Dict, Optional, Tuple

class MEVLoader:
    """Load and preprocess macro-economic variables (MEV) from Excel workbooks
    for modeling and scenario analyses."""

    def __init__(self,
                 model_workbook: str,
                 model_sheet: str,
                 scenario_workbooks: Optional[List[str]] = None,
                 scenario_sheets: Optional[Dict[str, str]] = None):
        self.model_workbook = model_workbook
        self.model_sheet = model_sheet
        self.scenario_workbooks = scenario_workbooks or []
        self.scenario_sheets = scenario_sheets or {}
        self._model_mev: Optional[pd.DataFrame] = None
        self._scenario_mevs: Dict[str, pd.DataFrame] = {}
        self._model_map: Dict[str, str] = {}
        self._scenario_maps: Dict[str, Dict[str, str]] = {}

    def load(self) -> None:
        self._model_mev, self._model_map = self._load_and_preprocess(
            self.model_workbook, self.model_sheet
        )
        for wb in self.scenario_workbooks:
            for scen, sheet in self.scenario_sheets.items():
                df, mapping = self._load_and_preprocess(wb, sheet)
                self._scenario_mevs[scen] = df
                self._scenario_maps[scen] = mapping

    def _load_and_preprocess(self, workbook: str, sheet: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        raw = pd.read_excel(workbook, sheet_name=sheet)
        # TODO: drop initial header rows, extract codes/names, build mapping
        df = raw.copy()
        df_mev_prc = df.iloc[1:,:]
        df_mev_prc.columns = df_mev_prc.iloc[0]

        df_mev_prc.index = df_mev_prc.iloc[:,0]
        df_mev_prc = df_mev_prc.iloc[1:,1:]
        df_mev_prc = df_mev_prc.loc[:, df_mev_prc.columns.notna()].iloc[2:]

        mev_name_lst = df_mev_prc.columns
        mev_name_lst = [mev.replace('Canada\n', '') for mev in mev_name_lst][1:]

        df_mev_prc.columns = df_mev_prc.iloc[0]
        df_mev_prc = df_mev_prc.iloc[1:,:]
 
        # Change index
        df_mev_prc.index = [i.replace(':', 'Q') for i in df_mev_prc.index]
        df_mev_prc.index = pd.PeriodIndex(df_mev_prc.index, freq='Q').to_timestamp(how='end').strftime('%Y-%m-%d')
        df_mev_prc.index = pd.to_datetime(df_mev_prc.index)

        mev_code_lst = df_mev_prc.columns.tolist()[1:]

        assert len(mev_name_lst) == len(mev_code_lst)
        mev_dict = dict(map(lambda i,j : (i,j) , mev_code_lst,mev_name_lst))

        return df_mev_prc, mev_dict

    @property
    def model_mev(self) -> pd.DataFrame:
        if self._model_mev is None:
            raise ValueError("Model MEV not loaded. Call load() first.")
        return self._model_mev

    @property
    def scenario_mevs(self) -> Dict[str, pd.DataFrame]:
        return self._scenario_mevs

    @property
    def model_map(self) -> Dict[str, str]:
        return self._model_map

    @property
    def scenario_maps(self) -> Dict[str, Dict[str, str]]:
        return self._scenario_maps
    
    def apply_to_all(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        self._model_mev = func(self._model_mev.copy())
        for scen, df in self._scenario_mevs.items():
            self._scenario_mevs[scen] = func(df.copy())