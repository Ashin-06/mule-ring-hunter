import numpy as np

from mulehunter.elliptic import load


def test_elliptic_loader_splits_by_time_and_skips_unknown_labels(tmp_path):
    rng = np.random.default_rng(0)
    ids = np.arange(1000, 1060)
    steps = np.r_[np.full(20, 5), np.full(20, 32), np.full(20, 40)]
    feats = np.c_[ids, steps, rng.normal(size=(60, 4))]
    np.savetxt(tmp_path / "elliptic_txs_features.csv", feats, delimiter=",", fmt="%g")
    labels = ["1", "2", "unknown"] * 20
    (tmp_path / "elliptic_txs_classes.csv").write_text("txId,class\n" + "".join(f"{i},{c}\n" for i, c in zip(ids, labels)))
    (tmp_path / "elliptic_txs_edgelist.csv").write_text("txId1,txId2\n1000,1001\n1001,1059\n")

    ds, step = load(tmp_path)
    assert ds.x.shape == (60, 4)                      # id and time step are not features
    assert ds.edge_index.tolist() == [[0, 1], [1, 59]]
    assert set(ds.split[step == 5]) <= {"train", ""} and set(ds.split[step == 32]) <= {"val", ""}
    assert set(ds.split[step == 40]) <= {"test", ""}
    assert (ds.split[2::3] == "").all() and ds.y[0] == 1 and ds.y[1] == 0
