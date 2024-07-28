from timer import Timer
from os.path import exists, join
from os import makedirs
from config import EXPLABEL, SEED, NGEN

class Folder:
    run_id = str(Timer.start.strftime('%s'))
    # DST = "runs/run_" + run_id
    DST = f"runs/label_{EXPLABEL}_seed_{SEED}_gen_{NGEN}_timestamp_{run_id}"
    if not exists(DST):
        makedirs(DST)
    DST_ARC = join(DST, "archive")
    DST_IND = join(DST, "inds")
