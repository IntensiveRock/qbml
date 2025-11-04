import sqlite3
from pathlib import Path
import pickle

from qbml.dynamics.spectraldensity import SpecDen, Lorentzians

def save_spddb(spdlist : list[list[SpecDen]], name : str, pth : Path):
    """
    Create a database to store spectral density parameters.
    Store those parameters.
    """
    conn = sqlite3.connect(pth / f"{name}.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE spdp(simnum, coupling, params BLOB)")
    coup = {0 : "x", 1 : "z"}
    for simnum, spdpair in enumerate(spdlist):
        for j, spd in enumerate(spdpair):
            pickled_dict = pickle.dumps(vars(spd))
            param_tuple = (simnum, coup[j], pickled_dict)
            cur.execute("INSERT INTO spdp VALUES(?, ?, ?)", param_tuple)
    conn.commit()
    conn.close()


def grab_spd_from_db(db_pth : Path, simnum : int) -> list[SpecDen]:
    """
    Grab specific spectral densities from a given database.
    """
    # First, select the rows where simnum=simnum
    # Second, reconstruct objects from params BLOB, need to unpickle that stuff
    conn = sqlite3.connect(db_pth)
    cur = conn.cursor()
    res = cur.execute(f"SELECT * FROM spdp WHERE simnum={simnum}")
    query_result = res.fetchall()
    # Returns a list of tuples containing coupling and parameters.
    spd_objs = []
    for spd in query_result:
        dummy_spd = Lorentzians([1],[1],[1],1)
        dummy_spd.__dict__ = pickle.loads(spd[2])
        spd_objs.append(dummy_spd)
    print([spd.centers for spd in spd_objs])
    return spd_objs
