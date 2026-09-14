import numpy as np
import pytest

from sta_forcemap.cli import main

SMALL = ["--z-max", "6", "--z-step", "0.5", "--pixels", "24", "--lateral-bins", "20",
         "--patch-size", "30", "-q"]


def test_cli_writes_html_and_npz(traj_file, tmp_path):
    out, npz = tmp_path / "map.html", tmp_path / "map.npz"
    main([str(traj_file), "-T", "300", "-o", str(out), "--save-npz", str(npz)] + SMALL)
    html = out.read_text()
    assert "zSlider" in html and "var atomTraces = [2, 3" in html
    data = np.load(npz)
    assert data["force_field"].shape == (20, 20, 20)
    assert data["force_profile"].shape == data["z"].shape == data["rho"].shape
    assert int(data["n_frames"]) == 40


def test_cli_no_overlay_and_frame_slicing(traj_file, tmp_path):
    out, npz = tmp_path / "map.html", tmp_path / "map.npz"
    main([str(traj_file), "-T", "300", "-o", str(out), "--no-overlay",
          "--start", "10", "--stride", "3", "--save-npz", str(npz)] + SMALL)
    assert "var atomTraces = [];" in out.read_text()
    assert int(np.load(npz)["n_frames"]) == 10


@pytest.mark.parametrize("args, expected", [
    (["--reference", "plane"], ("plane", "mean surface plane")),
    (["--reference-z", "9.5"], ("fixed", "Height above z = 9.5")),
])
def test_cli_reference_modes(traj_file, tmp_path, args, expected):
    out, npz = tmp_path / "map.html", tmp_path / "map.npz"
    main([str(traj_file), "-T", "300", "-o", str(out), "--save-npz", str(npz)] + args + SMALL)
    assert str(np.load(npz)["reference"]) == expected[0]
    assert expected[1] in out.read_text()


@pytest.mark.parametrize("args, message", [
    (["--reference", "fixed"], "needs --reference-z"),
    (["--reference", "local", "--reference-z", "3"], "cannot be combined"),
])
def test_cli_reference_errors(traj_file, tmp_path, args, message):
    with pytest.raises(SystemExit, match=message):
        main([str(traj_file), "-T", "300", "-o", str(tmp_path / "x.html")] + args + SMALL)


def test_cli_bad_selection(traj_file, tmp_path):
    with pytest.raises(SystemExit, match="matched no"):
        main([str(traj_file), "-T", "300", "-o", str(tmp_path / "x.html"),
              "--surface", "N"] + SMALL)
