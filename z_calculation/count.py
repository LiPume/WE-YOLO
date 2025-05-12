import os

def count_files(path):
    """Count the number of files in train and test subfolders."""
    counts = {}
    for subdir in ['train', 'val']:
        subfolder = os.path.join(path, subdir)
        if os.path.exists(subfolder):
            counts[subdir] = len(os.listdir(subfolder))
        else:
            counts[subdir] = 0
    return counts

# Define source paths
source_paths = [
    r"E:\AIpractice\detect\yolov11\ultralytics-main\dataset\split_g_1117",
    r"E:\AIpractice\detect\yolov11\ultralytics-main\dataset\split_s_1114",
    r"E:\AIpractice\detect\yolov11\ultralytics-main\dataset\split_z_0920"
]

# Count files in each source path
stats = {}
for source_path in source_paths:
    image_path = os.path.join(source_path, 'images')
    label_path = os.path.join(source_path, 'labels')

    stats[source_path] = {
        'images': count_files(image_path),
        'labels': count_files(label_path)
    }

# Output results
print("Stats for each source path:")
for source, stat in stats.items():
    print(f"{source}:")
    print(f"  Images - Train: {stat['images']['train']}, Test: {stat['images']['val']}")
    print(f"  Labels - Train: {stat['labels']['train']}, Test: {stat['labels']['val']}")
