import os
import glob
import numpy as np

def validate_pts_folder(folder, expected_columns=6, min_file_size=100):
    """
    Checks each .pts file in `folder` to see if it:
      1) Actually loads with numpy.loadtxt().
      2) Has the correct shape: (N, `expected_columns`).
      3) Is above some minimum file size in bytes (optional).
      4) Data ranges can also be checked if desired.
    Returns a string summarizing any errors or a success message.
    """
    pts_files = glob.glob(os.path.join(folder, "*.pts"))
    if not pts_files:
        return f"No .pts files found in {folder}."

    for f in pts_files:
        size_bytes = os.path.getsize(f)
        if size_bytes < min_file_size:
            return f"Error: File '{f}' is too small ({size_bytes} bytes). Possibly corrupt."

        try:
            data = np.loadtxt(f)
        except Exception as e:
            return f"Error: Failed to load '{f}' with numpy.loadtxt(): {str(e)}"

        if data.ndim != 2 or data.shape[1] != expected_columns:
            return (f"Error: File '{f}' has shape {data.shape}, "
                    f"expected (N, {expected_columns}).")

    return f"Successfully validated {len(pts_files)} .pts files in '{folder}'."

def validate_npy_folder(folder, expected_columns=7, min_file_size=100):
    """
    Checks each .npy file in `folder` to see if it:
      1) Actually loads with numpy.load().
      2) Has the correct shape: (N, `expected_columns`).
      3) Is above some minimum file size (optional).
      4) Potentially checks data range or shape IDs if desired.

    Returns a string summarizing results or a success message.
    """
    npy_files = glob.glob(os.path.join(folder, "*.npy"))
    if not npy_files:
        return f"No .npy files found in {folder}."

    for f in npy_files:
        size_bytes = os.path.getsize(f)
        if size_bytes < min_file_size:
            return f"Error: File '{f}' is too small ({size_bytes} bytes). Possibly corrupt."

        try:
            data = np.load(f)
        except Exception as e:
            return f"Error: Failed to load '{f}' with numpy.load(): {str(e)}"

        if data.ndim != 2 or data.shape[1] != expected_columns:
            return (f"Error: File '{f}' has shape {data.shape}, "
                    f"expected (N, {expected_columns}).")

    return f"Successfully validated {len(npy_files)} .npy files in '{folder}'."

def main():
    """
    Example usage. This can be run as a standalone test:
    python validate_files.py /path/to/folder
    """
    import sys
    if len(sys.argv) < 2:
        print("Usage: python validate_files.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    msg_pts = validate_pts_folder(folder, expected_columns=6, min_file_size=100)
    print(msg_pts)

    msg_npy = validate_npy_folder(folder, expected_columns=7, min_file_size=200)
    print(msg_npy)

if __name__ == "__main__":
    main()