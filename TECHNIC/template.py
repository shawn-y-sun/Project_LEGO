# TECHNIC/template.py
from typing import List, Dict, Any
import pandas as pd
from .writer import Val, ValueWriter, TemplateLoader, TemplateWriter
from .cm import CM

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

class PPNR_OLS_ExportTemplate(ExportTemplateBase):
    """
    Export template for PPNR project using OLS models.
    Defines mapping of CM measures/specs into Excel template locations.
    """
    def __init__(
        self,
        template_files: List[str],
        cms: Dict[str, CM]
    ):
        # Build mapping dict keyed by template filename
        mapping: Dict[str, Dict[str, List[Val]]] = {}

        # Example: single template workbook 'PPNR_OLS_Template.xlsx'
        tpl = template_files[0]
        # Specs go to 'Specs' sheet at A2
        specs_vals = [Val(cm.specs, 'A2') for cm in cms.values()]
        # In-sample performance to 'Perf' sheet at B4
        in_perf_df = pd.DataFrame([m.in_perf_measures for m in cms.values()])
        in_perf_df.insert(0, 'cm_id', list(cms.keys()))
        in_perf_vals = [Val(in_perf_df, 'B4', include_header=True)]
        # Out-sample performance to 'Perf' sheet at B20
        out_perf_df = pd.DataFrame([m.out_perf_measures for m in cms.values()])
        out_perf_df.insert(0, 'cm_id', list(cms.keys()))
        out_perf_vals = [Val(out_perf_df, 'B20', include_header=True)]
        # Parameters to 'Params' sheet at A2
        params_rows = []
        for cm_id, m in cms.items():
            for var, stats in m.param_measures.items():
                row = stats.copy()
                row['cm_id'] = cm_id
                row['Variable'] = var
                params_rows.append(row)
        params_df = pd.DataFrame(params_rows)
        params_df = params_df[['cm_id', 'Variable'] + [c for c in params_df.columns if c not in ['cm_id','Variable']]]
        params_vals = [Val(params_df, 'A2', include_header=True)]
        # Tests to 'Diagnostics' sheet at A2
        tests_df = pd.DataFrame([m.test_measures for m in cms.values()])
        tests_df.insert(0, 'cm_id', list(cms.keys()))
        tests_vals = [Val(tests_df, 'A2', include_header=True)]

        mapping[tpl] = {
            'Specs': specs_vals,
            'Perf':  in_perf_vals + out_perf_vals,
            'Params': params_vals,
            'Diagnostics': tests_vals
        }

        super().__init__(template_files, cms, mapping)
