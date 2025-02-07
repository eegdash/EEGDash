import os

def find_files_recursively(folder_path, extensions):
    """ Recursively find the first file of each specified extension in a folder. """
    first_files = {ext: None for ext in extensions}

    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            for ext in extensions:
                if filename.endswith(ext) and first_files[ext] is None:
                    first_files[ext] = os.path.join(dirpath, filename)
        
        # Stop searching if we found all required files
        if all(first_files.values()):
            break
    
    return first_files

def scan_folders(root_path):
    matching_folders = []

    # Scan only the first level of directories (non-recursive)
    with os.scandir(root_path) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name.startswith("ds00"):
                folder_path = os.path.join(root_path, entry.name)
                
                # Recursively find the first .set and .fdt files
                first_files = find_files_recursively(folder_path, [".set", ".fdt"])

                first_set_file = first_files[".set"]
                first_fdt_file = first_files[".fdt"]

                if first_set_file and not first_fdt_file:
                    matching_folders.append(folder_path)
                    print(f"{folder_path} - OK")
                else:
                    print(f"{folder_path} - NOT OK")

                # Print the first found .set and .fdt files if they exist
                if first_set_file:
                    print(f"  First .set file: {first_set_file}")
                if first_fdt_file:
                    print(f"  First .fdt file: {first_fdt_file}")

    return matching_folders

if __name__ == "__main__":
    root_directory = input("Enter the root directory to scan: ").strip()

    if not os.path.exists(root_directory):
        print("The specified directory does not exist.")
    else:
        result_folders = scan_folders(root_directory)

        if result_folders:
            print("\nMatching folders:")
            for folder in result_folders:
                print(folder)
        else:
            print("\nNo matching folders found.")