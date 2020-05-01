import pandas as pd
from pyspan.config import *

TASK_PATH = paths["synonyms_task_path"]

words = pd.read_csv(TASK_PATH + "synonyms.csv")
