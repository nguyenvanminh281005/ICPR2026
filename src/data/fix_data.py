import os

def find_folders(root_dir, threshold=11):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if len(filenames) > threshold:
            print(dirpath)

if __name__ == "__main__":
    root = r"/mnt/data/KHTN2023/Project/MultiFrame-LPR/data/new/train_pub/train/Scenario-A"  # đổi path nếu cần
    find_folders(root)