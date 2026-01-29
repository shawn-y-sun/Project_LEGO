# # DEMO Part 1: Data Setup, Feature Engineering & Analysis, Segment Definition 

# In[ ]:


import pandas as pd
import Technic as tc
from Technic import TSFM, DumVar


# ## Prepare and Load Data

# In[ ]:


df_internal = pd.read_csv(r'Demo Data/housing_market.csv')
df_internal.head()


# In[ ]:


df_internal['home_price_GR1'] = df_internal['home_price_index'].pct_change().shift(-1)
df_internal['home_price_GR3'] = df_internal['home_price_index'].pct_change(3).shift(-1)


# In[ ]:


ID_config = {
    'in_sample_start':"2006-01-31",
    'in_sample_end':"2023-09-30",
    'full_sample_end':"2025-09-30",
}
int_ldr = tc.TimeSeriesLoader(
    in_sample_start="2006-01-31",
    in_sample_end="2023-09-30",
    full_sample_end="2025-09-30",
    scen_p0="2023-09-30",
)
int_ldr.load(df_internal, date_col='date')


# In[ ]:


int_ldr.internal_data


# ## MEV Loader

# In[ ]:


# Read Historical Macro Data
df_mev_qtr = pd.read_csv(r'Demo Data/macro_quarterly.csv')
df_mev_mth = pd.read_csv(r'Demo Data/macro_monthly.csv')
df_mev_mth.ffill(inplace=True)

# Read Scenario Macro Data
df_scen_mev_qtr_base = pd.read_excel(r'Demo Data/macro_scenarios_quarterly.xlsx', sheet_name='baseline').set_index('observation_date')
df_scen_mev_qtr_adv = pd.read_excel(r'Demo Data/macro_scenarios_quarterly.xlsx', sheet_name='adverse').set_index('observation_date')
df_scen_mev_qtr_sev = pd.read_excel(r'Demo Data/macro_scenarios_quarterly.xlsx', sheet_name='severely_adverse').set_index('observation_date')

df_scen_mev_mth_base = pd.read_excel(r'Demo Data/macro_scenarios_monthly.xlsx', sheet_name='baseline').set_index('observation_date')
df_scen_mev_mth_adv = pd.read_excel(r'Demo Data/macro_scenarios_monthly.xlsx', sheet_name='adverse').set_index('observation_date')
df_scen_mev_mth_sev = pd.read_excel(r'Demo Data/macro_scenarios_monthly.xlsx', sheet_name='severely_adverse').set_index('observation_date')


# In[ ]:


# Create loader for Macro Variables
mev_ldr = tc.MEVLoader()

# load quarterly macro data
mev_ldr.load(
    source=df_mev_qtr,
    date_col='observation_date'
)

# load monthly macro data
mev_ldr.load(
    source=df_mev_mth,
    date_col='observation_date'
)

# Load scenario quarterly macro data
mev_ldr.load_scens(
    {'Base': df_scen_mev_qtr_base,
     'Adv': df_scen_mev_qtr_adv,
     'Sev': df_scen_mev_qtr_sev},
    set_name='Scenario'
)

# Load scenario monthly macro data
mev_ldr.load_scens(
    {'Base': df_scen_mev_mth_base,
     'Adv': df_scen_mev_mth_adv,
     'Sev': df_scen_mev_mth_sev},
    set_name='Scenario'
)


# ## Data Manager

# In[ ]:


dm = tc.DataManager(
    int_ldr,
    mev_ldr
)


# In[ ]:


# Feature Engineering Function
def new_features(df_mev: pd.DataFrame, df_in: pd.DataFrame):
    # -------------------------------------------------------------------------
    # A) Rates & curve shape (term structure signals)
    # -------------------------------------------------------------------------
    df_mev['USYC10_2'] = df_mev['USGOV10Y'] - df_mev['USGOV2Y']
    df_mev['USYC10_1'] = df_mev['USGOV10Y'] - df_mev['USGOV1Y']
    df_mev['USYC10_6M'] = df_mev['USGOV10Y'] - df_mev['USGOV6M']
    df_mev['USYC30_10'] = df_mev['USGOV30Y'] - df_mev['USGOV10Y']
    df_mev['USYC5_2'] = df_mev['USGOV5Y'] - df_mev['USGOV2Y']
    df_mev['USREAL_TERM_PREM_PROXY'] = df_mev['USGOV10Y'] - df_mev['USGOVR10Y']

    # -------------------------------------------------------------------------
    # B) Credit conditions (spreads / wedges)
    # -------------------------------------------------------------------------
    df_mev['USCORP_SPRD_BAA_AAA'] = df_mev['USCORPBBB10Y'] - df_mev['USCORPAA10Y']
    df_mev['USCORP_SPRD_BAA_T10'] = df_mev['USCORPBBB10Y'] - df_mev['USGOV10Y']
    df_mev['USCORP_SPRD_AAA_T10'] = df_mev['USCORPAA10Y'] - df_mev['USGOV10Y']
    df_mev['USCP_FF_SPRD'] = df_mev['USCPF'] - df_mev['USFF']
    df_mev['USPRIME_FF_SPRD'] = df_mev['USPRIME'] - df_mev['USFF']
    df_mev['USIORB_FF_SPRD'] = df_mev['USIORB'] - df_mev['USFF']
    df_mev['USSOFR_FF_SPRD'] = df_mev['USSOFR'] - df_mev['USFF']

    # -------------------------------------------------------------------------
    # C) Mortgage pricing & household financing wedges
    # -------------------------------------------------------------------------
    df_mev['USMORT30_T10_SPRD'] = df_mev['USMORT30Y'] - df_mev['USGOV10Y']
    df_mev['USMORT15_T10_SPRD'] = df_mev['USMORT15Y'] - df_mev['USGOV10Y']
    df_mev['USARM_T2_SPRD'] = df_mev['USAM51'] - df_mev['USGOV2Y']
    df_mev['USMORT30_15_SPRD'] = df_mev['USMORT30Y'] - df_mev['USMORT15Y']

    # -------------------------------------------------------------------------
    # D) Liquidity / leverage ratios
    # -------------------------------------------------------------------------
    df_mev['USLIQ_M2_GDP'] = df_mev['USM2'] / df_mev['USNGDP']
    df_mev['USLIQ_M1_M2'] = df_mev['USM1'] / df_mev['USM2']
    df_mev['USCREDIT_CC_INC'] = df_mev['USCC'] / df_mev['USDI']
    df_mev['USCREDIT_CC_GDP'] = df_mev['USCC'] / df_mev['USNGDP']
    df_mev['USCONS_PCE_INC'] = df_mev['USNC'] / df_mev['USDI']
    df_mev['USSAV_DLR'] = df_mev['USDI'] - df_mev['USNC']

    # -------------------------------------------------------------------------
    # E) Nominal vs real “price-level wedges” (implied deflators)
    # -------------------------------------------------------------------------
    df_mev['USPCE_IMPLICIT_DEF'] = df_mev['USNC'] / df_mev['USRC']
    df_mev['USDPI_IMPLICIT_DEF'] = df_mev['USDI'] / df_mev['USRPDI']

    # -------------------------------------------------------------------------
    # F) Risk appetite & relative valuation proxies
    # -------------------------------------------------------------------------
    df_mev['USRISKON_SPX_VIX'] = df_mev['USSP500'] / df_mev['USVIXA']

    # -------------------------------------------------------------------------
    # G) Real estate cross-market relative pricing
    # -------------------------------------------------------------------------
    df_mev['USCRE_HOUS_REL'] = df_mev['USNCREIF'] / df_mev['USCSH']
    df_mev['USCP_GDP_SHARE'] = df_mev['USCP'] / df_mev['USNGDP']

    return df_mev, df_in

dm.apply_to_all(new_features)

var_update_dict = {
    'USYC10_2': {'type': 'rate', 'category': 'yield slope' },
    'USYC10_1': {'type': 'rate', 'category': 'yield slope' },
}


var_update_dict = {
    # A) Rates & curve shape
    'USYC10_2': {'type': 'rate', 'category': 'yield slope'},
    'USYC10_1': {'type': 'rate', 'category': 'yield slope'},
    'USYC10_6M': {'type': 'rate', 'category': 'yield slope'},
    'USYC30_10': {'type': 'rate', 'category': 'yield slope'},
    'USYC5_2': {'type': 'rate', 'category': 'yield slope'},
    'USREAL_TERM_PREM_PROXY': {'type': 'rate', 'category': 'real rate spread'},

    # B) Credit conditions
    'USCORP_SPRD_BAA_AAA': {'type': 'rate', 'category': 'credit spread'},
    'USCORP_SPRD_BAA_T10': {'type': 'rate', 'category': 'credit spread'},
    'USCORP_SPRD_AAA_T10': {'type': 'rate', 'category': 'credit spread'},
    'USCP_FF_SPRD': {'type': 'rate', 'category': 'funding spread'},
    'USPRIME_FF_SPRD': {'type': 'rate', 'category': 'bank pricing spread'},
    'USIORB_FF_SPRD': {'type': 'rate', 'category': 'policy spread'},
    'USSOFR_FF_SPRD': {'type': 'rate', 'category': 'funding spread'},

    # C) Mortgage wedges
    'USMORT30_T10_SPRD': {'type': 'rate', 'category': 'mortgage spread'},
    'USMORT15_T10_SPRD': {'type': 'rate', 'category': 'mortgage spread'},
    'USARM_T2_SPRD': {'type': 'rate', 'category': 'mortgage spread'},
    'USMORT30_15_SPRD': {'type': 'rate', 'category': 'mortgage spread'},

    # D) Liquidity / leverage ratios
    'USLIQ_M2_GDP': {'type': 'level', 'category': 'liquidity ratio'},
    'USLIQ_M1_M2': {'type': 'level', 'category': 'liquidity ratio'},
    'USCREDIT_CC_INC': {'type': 'level', 'category': 'leverage ratio'},
    'USCREDIT_CC_GDP': {'type': 'level', 'category': 'leverage ratio'},
    'USCONS_PCE_INC': {'type': 'level', 'category': 'consumption ratio'},
    'USSAV_DLR': {'type': 'level', 'category': 'savings level'},

    # E) Implied deflators
    'USPCE_IMPLICIT_DEF': {'type': 'level', 'category': 'price level'},
    'USDPI_IMPLICIT_DEF': {'type': 'level', 'category': 'price level'},

    # F) Risk / relative valuation
    'USRISKON_SPX_VIX': {'type': 'level', 'category': 'risk appetite'},

    # G) Real estate cross-market
    'USCRE_HOUS_REL': {'type': 'level', 'category': 'relative valuation'},
    'USCP_GDP_SHARE': {'type': 'level', 'category': 'income share'},
}

dm.update_var_map(var_update_dict)


# In[ ]:


# Build Segment for Home Price Growth
seg_config = {
    'segment_id': 'home_price_GR1',
    'target': 'home_price_GR1',
    'model_type': tc.Growth,
    'target_base': 'home_price_index',
    'data_manager': dm,
    'model_cls': tc.OLS
}

seg = tc.Segment(**seg_config)

