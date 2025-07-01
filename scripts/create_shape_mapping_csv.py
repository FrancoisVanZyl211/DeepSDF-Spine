import os
import argparse
import csv

def main():
    """
    create_shape_mapping_csv.py

    Scans a given folder for .obj files, sorts them by name, and writes a CSV
    mapping of shape_id to filename.

    Usage:
    python create_shape_mapping_csv.py --obj_folder "C:/Path/to/obj_files" \
                                        --out_csv "mapping.csv" \
                                        --max_shapes 25
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_folder", type=str, required=True,
                        help="Folder containing .obj files for ground truth.")
    parser.add_argument("--out_csv", type=str, default="shape_mapping.csv",
                        help="Where to write the CSV file.")
    parser.add_argument("--max_shapes", type=int, default=9999,
                        help="Maximum number of shapes to list in CSV.")
    args = parser.parse_args()

    # 1) Gather .obj files
    all_files = os.listdir(args.obj_folder)
    obj_files = [f for f in all_files if f.lower().endswith(".obj")]
    obj_files.sort()  # Alphabetically sorted

    # 2) Limit to --max_shapes if needed
    obj_files = obj_files[:args.max_shapes]

    if not obj_files:
        print(f"No .obj files found in: {args.obj_folder}")
        return

    # 3) Write CSV
    csv_path = args.out_csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shape_id", "obj_filename"])
        for idx, filename in enumerate(obj_files):
            writer.writerow([idx, filename])

    print(f"Wrote {len(obj_files)} entries to '{csv_path}'.")

if __name__ == "__main__":
    main()

#python create_shape_mapping_csv.py --obj_folder "C:/Users/GGPC/Downloads/SpineData/LumbarVertebraeDataSet/dataphotoscanfinalpublish" --out_csv "my_mapping.csv" --max_shapes 25
