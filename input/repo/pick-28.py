import os
import shutil

# Mapping the 28 GBC languages to your available CSV files
# Some are mapped to None if they weren't visibly present in your 'ls' list
gbc_28_files = {
    # Column 1
    "Twi": "twi-eng.csv",
    "Fante": "fat-eng.csv",
    "Nzema": "nzi-eng.csv",
    "Ahanta": None, 
    "Ga": "gaa-eng.csv",
    "Dangme": "ada-eng.csv", # Ada is a dialect of Dangme
    "Ewe": "ewe-eng.csv",
    "Akoyede": None,
    "Wiase": None,
    
    # Column 2
    "Kaakye": None,
    "Sefwi": "sfw-eng.csv",
    "Aowin": "any-eng.csv", # Anyin/Aowin
    "Bono": None,
    "Mo (Deg)": "mzw-eng.csv", 
    "Gonja": "gjn-eng.csv",
    "Dagbanli": "dag-eng.csv",
    "Dwan": None,
    "Nawuri": "naw-eng.csv",
    "Hausa": None,
    
    # Column 3
    "Dagaare": "dga-eng.csv",
    "Sisaale": "sil-eng.csv",
    "Gurune": "gur-eng.csv",
    "Kusaal": "kus-eng.csv",
    "Kasem": "xsm-eng.csv",
    "Buli": "bwu-eng.csv",
    "Bisa": "bib-eng.csv", 
    "Baasare": None,
    "Nchumuru": "ncu-eng.csv"
}

# Directories
input_dir = "./"
output_dir = "gbc_28"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

copied_count = 0
missing_count = 0

print(f"Scanning for the 28 GBC languages in '{input_dir}'...\n")

for lang, filename in gbc_28_files.items():
    if filename:
        source_path = os.path.join(input_dir, filename)
        dest_path = os.path.join(output_dir, filename)
        
        # Check if the file exists in the input folder
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"✅ Copied {lang}: {filename}")
            copied_count += 1
        else:
            print(f"❌ Missing {lang}: {filename} (File not found)")
            missing_count += 1
    else:
        print(f"❌ Missing {lang}: No matching CSV file identified in your directory")
        missing_count += 1

print("\n--- Summary ---")
print(f"Successfully copied: {copied_count}")
print(f"Missing from input: {missing_count}")
print(f"All copied files are now in the '{output_dir}' folder.")
