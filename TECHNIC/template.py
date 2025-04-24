# TECHNIC/template.py
from typing import List, Dict, Any, Type
import pandas as pd
from .writer import Val, ValueWriter, TemplateLoader, TemplateWriter
from .cm import CM
from abc import ABC, abstractmethod

class ValueMapEditor(ABC):
    """
    Abstract base class for constructing a mapping of template values
    from a dictionary of CM instances.
    """
    def __init__(self, cms: Dict[str, Any]):
        """
        :param cms: A dictionary of CM instances, keyed by model identifier.
        """
        self.cms = cms

    @property
    @abstractmethod
    def value_map(self) -> Dict[str, Any]:
        """
        Returns a nested dictionary mapping template workbook and sheet names
        to lists of Val instances (or similar), conforming to ExportTemplateBase
        requirements. Example structure:

            {
                'workbook1.xlsx': {
                    'Sheet1': [Val(...), Val(...), ...],
                    'Sheet2': [Val(...), ...],
                },
                'workbook2.xlsx': {
                    'Main': [Val(...), ...],
                }
            }
        """
        ...

class ExportTemplateBase:
    """
    Abstract base for Excel export templates. Handles loading template workbooks
    and writing data blocks into specified cells.

    Parameters
    ----------
    template_files : List[str]
        Paths to one or more Excel template workbooks.
    cms : Dict[str, CM]
        Candidate-model dict keyed by cm_id, providing measures and specs.
    mapping : Dict[str, Dict[str, List[Val]]]
        High-level mapping: template_file -> sheet_name -> list of Val instances.
    """
    def __init__(
        self,
        template_files: List[str],
        cms: Dict[str, CM],
        mapping: Dict[str, Dict[str, List[Val]]]
    ):
        self.loader = TemplateLoader(template_files)
        self.cms = cms
        self.mapping = mapping

    def export(self, output_map: Dict[str, str]) -> None:
        """
        Write all values into loaded templates and save to output paths.

        Parameters
        ----------
        output_map : Dict[str, str]
            Mapping from each input template path to its output path.
        """
        # Prepare sheet_tasks_map: template -> sheet -> list of ValueWriter
        sheet_tasks_map: Dict[str, Dict[str, List[ValueWriter]]] = {}
        for tmpl_path, sheets in self.mapping.items():
            wb = self.loader.get_workbook(tmpl_path)
            sheet_tasks_map[tmpl_path] = {}
            for sheet_name, vals in sheets.items():
                tasks: List[ValueWriter] = []
                for val in vals:
                    ws = wb[sheet_name]
                    writer = ValueWriter(
                        worksheet=ws,
                        data=val.data,
                        start_cell=val.start_cell,
                        orientation=val.orientation,
                        include_index=val.include_index,
                        include_header=val.include_header
                    )
                    tasks.append(writer)
                sheet_tasks_map[tmpl_path][sheet_name] = tasks

        # Write and save
        tmpl_writer = TemplateWriter(self.loader)
        tmpl_writer.write(sheet_tasks_map)
        tmpl_writer.save_all(output_map)


class PPNR_OLS_ValueMap(ValueMapEditor):
    """
    Concrete ValueMapEditor for PPNR OLS models. Gathers performance and
    parameter tables across all CM instances and maps them for Excel export.
    """
    def __init__(self, cms: Dict[str, Any]):
        """
        :param cms: A dictionary of CM instances.
        """
        super().__init__(cms)

    @property
    def perf_df(self) -> pd.DataFrame:
        """
        Combine in-sample performance tables from all CMs into a single DataFrame.
        """
        dfs = []
        for cm in self.cms.values():
            # Assumes cm.report_in.show_perf_tbl() returns a DataFrame
            df = cm.report_in.show_perf_tbl().copy()
            df.insert(0, 'model_id', cm.model_id)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    @property
    def params_df(self) -> pd.DataFrame:
        """
        Combine parameter tables from all CMs into a single DataFrame.
        """
        dfs = []
        for cm in self.cms.values():
            # Assumes cm.report_in.show_params_tbl() returns a DataFrame
            df = cm.report_in.show_params_tbl().copy()
            df.insert(0, 'model_id', cm.model_id)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    @property
    def value_map(self) -> Dict[str, Dict[str, List[Val]]]:
        """
        Map the performance and parameter DataFrames to Excel sheets via Val:
        'Performances' and 'Parameters' sheets in the PPNR OLS template.
        """
        return {
            'PPNR_OLS_Template.xlsx': {
                'Performances': [
                    Val(self.perf_df, 'A1')
                ],
                'Parameters': [
                    Val(self.params_df, 'A1')
                ],
            }
        }


class PPNR_OLS_ExportTemplate(ExportTemplateBase):
    """
    Export template for PPNR project using OLS models.
    Delegates mapping construction to a provided ValueMapEditor class.
    """
    def __init__(
        self,
        cms: Dict[str, CM],
        vm_cls: Type[ValueMapEditor] = PPNR_OLS_ValueMap,
        template_files: List[str] = None
    ):
        # Default template file if none provided
        if template_files is None:
            template_files = ['Template/PPNR_OLS_Template.xlsx']

        # Instantiate the value-map editor and retrieve mapping
        vm: ValueMapEditor = vm_cls(cms)
        mapping = vm.value_map

        super().__init__(template_files, cms, mapping)