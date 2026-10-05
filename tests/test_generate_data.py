from src.generate_data import make_dataset


def test_dataset_is_reproducible():
    assert make_dataset().equals(make_dataset())


def test_dataset_size():
    assert len(make_dataset(repeats=4)) == 40


def test_damage_area_never_negative():
    assert (make_dataset()["delam_area_mm2"] >= 0).all()


def test_strength_drops_with_energy():
    df = make_dataset()
    means = df.groupby("impact_energy_J")["CAI_strength_MPa"].mean()
    assert means.loc[0.0] > means.loc[40.0]