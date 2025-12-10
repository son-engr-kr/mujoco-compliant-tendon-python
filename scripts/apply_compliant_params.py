"""
Apply compliant-tendon fitted parameters to a MuJoCo muscle XML.

Usage:
    python scripts/apply_compliant_params.py \
        [xml_path] \
        [csv_path] \
        [out_path]

Defaults assume execution from repo root:
    xml_path:  myosim_convert/myo_sim/leg/assets/myolegs_muscle.xml
    csv_path:  mujoco_muscle_data/fitted_params_length_only.csv
    out_path:  overwrite xml_path unless a third argument is given

The script sets:
  - gainprm  -> nine fitted parameters (F_max ... E_REF)
  - gaintype -> "compliant_mtu"
Other attributes (e.g., lengthrange) are preserved.
"""

from pathlib import Path
import csv
import sys
import xml.etree.ElementTree as ET


def load_params(csv_path: Path):
    param_map = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            muscle = row["muscle"]
            if muscle == "":
                continue
            values = [
                float(row["F_max"]),
                float(row["l_opt"]),
                float(row["l_slack"]),
                float(row["v_max"]),
                float(row["W"]),
                float(row["C"]),
                float(row["N"]),
                float(row["K"]),
                float(row["E_REF"]),
            ]
            param_map[muscle] = values
    return param_map


def format_gain(values):
    return " ".join(f"{v:.12g}" for v in values)


def counterpart_name(name: str):
    if name.endswith("_l"):
        return name[:-2] + "_r"
    if name.endswith("_r"):
        return name[:-2] + "_l"
    return None


def update_xml(xml_path: Path, param_map, out_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    updated = 0
    missing = []
    mirrored = 0

    for general in root.findall(".//general"):
        name = general.attrib.get("name", "")
        values = None

        if name in param_map:
            values = param_map[name]
        else:
            mirror = counterpart_name(name)
            if mirror and mirror in param_map:
                values = param_map[mirror]
                mirrored += 1

        if values is None:
            missing.append(name)
            continue

        general.set("gainprm", format_gain(values))
        general.set("gaintype", "compliant_mtu")
        general.set("biasprm", "0")
        if "lengthrange" in general.attrib:
            general.attrib.pop("lengthrange")
        updated += 1

    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return updated, mirrored, missing


def main():
    root_dir = Path(__file__).resolve().parent.parent

    xml_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else root_dir / "myosim_convert/myo_sim/leg/assets/myolegs_muscle.xml"
    )
    csv_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else root_dir / "mujoco_muscle_data/fitted_params_length_only.csv"
    )
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else xml_path

    params = load_params(csv_path)
    updated, mirrored, missing = update_xml(xml_path, params, out_path)

    print(f"Applied compliant parameters to {updated} actuators.")
    print(f"Mirrored L/R pairs: {mirrored}")
    print(f"Output: {out_path}")
    if missing:
        print("Names not found in CSV or mirrored (left unchanged):")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()

