# Quick start

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gui]"
silicon-sort gui
```

Pass a recording and a geometry file if you already know them:

```bash
silicon-sort gui --dat amplifier.dat --cfg electrode.cfg
```

The window opens filling the screen. The parameters dock is on the left. The log is a short panel under that dock, not a bar across the traces.

## A first session

These values match an Intan RHD2132 recording of an A1x32-Poly2 probe.

| Field | Value |
| --- | --- |
| Channels | 32 |
| Sampling rate | 30000 Hz |
| Dtype | int16 |
| Scale | 0.195 µV/bit |
| Highpass | 300 Hz |
| Lowpass | 6000 Hz |
| Threshold | −5 MAD |

1. On **Load**, browse to the `.dat`. The traces appear in file order, channel 1 through channel 32. Browse to `electrode.cfg` and the same traces reorder by depth.
2. Leave preprocessing off if the file is already the recording you want to sort. Mask bad channels on the **Mask** step.
3. Open **Detect**, run detection, and scroll the traces. Use **Suggest** on a channel, mark events good or bad, and run detection again.
4. Open **Clusters** and preview one site.
5. Open **Autosort**, run it, and export a Phy folder.

The cache for `amplifier.dat` is the folder `amplifier.silicon-sort` next to the file. A filtered cache is float32, about twice the size of an int16 recording.

## Geometry file

`electrode.cfg` starts with a name line. Each later line is `site channel x y`, in micrometers. Channel numbers start at 1 and follow the order of samples in the binary file. A line with three numbers is read as `channel x y`.

```
poly2
1 1 -21.65 0
2 2 21.65 25
```
