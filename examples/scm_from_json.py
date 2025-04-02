from scm.base import SCM
from utils.plot import draw_scm

FILE_NAME = "example_SCM.json"

scm = SCM.from_json(FILE_NAME)
scm.visualize()

samples = scm.sample(100)
for var, values in samples.items():
    print(f"{var}: {values[:10]}")
