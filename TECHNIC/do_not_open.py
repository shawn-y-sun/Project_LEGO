# --- Test Harness for FeatureBuilder ---

# 1. Build your mapping and forced‐in TSFM
from QT import TSFM
import QT.transform as tf

# Sample transform map
mev_transMap = {
    'GDP':      'growthrate',
    'Unemp':    'diff',
    'DispInc':  'growthrate',
    'HPI':      'growthrate'
}

# Pre‐configure HPI at a large lag (36)
hpi_forced = TSFM(transform_fn=tf.QQGR, max_lag=36)
hpi_forced.mev = 'HPI'

# 2. Instantiate with freq='Q', force in Q1 and HPI, require GDP, max 3 vars, lag up to 1
fb = FeatureBuilder(
    mev_transMap=mev_transMap,
    freq='Q',
    max_var_num=3,
    forced_in=['Q1', hpi_forced],
    driver_pool=['GDP', 'Unemp', 'DispInc', 'Q1', 'HPI'],
    desired_pool=['GDP'],
    max_lag=1
)

# 3. Generate and display
combos = fb.generate_combinations()
for combo in combos:
    print(combo)

print(f"\nTotal combinations: {len(combos)}")
