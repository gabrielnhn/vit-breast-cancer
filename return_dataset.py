import os
from datasets import Dataset, DatasetDict, ClassLabel

class_labels = ["Benign", "Malignant"]
label = ClassLabel(names=class_labels)

def read_images_from_folder(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                images.append(os.path.join(root, file))
    return images

import re
def super_split(input_string, splitters):
    pattern = '|'.join(map(re.escape, splitters))
    result = re.split(pattern, input_string)
    return result

def parse_filename(filename):
    # print(filename)
    parts = super_split(filename, ["-", "_"])

    details = {
        "BIOPSY_PROCEDURE": parts[0],
        "TUMOR_CLASS": parts[1],
        "TUMOR_TYPE": parts[2],
        "YEAR": parts[3],
        "SLIDE_ID": parts[4],
        "MAGNIFICATION": parts[5],
        "SEQ": parts[6].split('.')[0]
    }
    # labels = {"labels": 0 if details["TUMOR_CLASS"]=="B" else 1}
    labels = "Benign" if details["TUMOR_CLASS"]=="B" else "Malignant"
    return details, labels

def organize_dataset(root_path):
    dataset = {}

    for fold in range(1, 6):
        for split in ["train", "test"]:
            dataset[split] = {"image_paths": [], "prelabels": [], "details": []}
            for magnification in ["40X", "100X", "200X", "400X"]:
                folder_path = os.path.join(root_path, f"fold{fold}", split, magnification)
                # print(folder_path)
                images = read_images_from_folder(folder_path)
                dataset[split]["image_paths"].extend(images)

                details, labels = zip(*[parse_filename(os.path.basename(image)) for image in images])
                # dataset[split]["labels"].extend(labels)
                dataset[split]["prelabels"].extend(labels)
                dataset[split]["details"].extend(details)

           
            dataset[split]["labels"] = label.str2int(dataset[split]["prelabels"])
            dataset[split].pop("prelabels")


            

    return dataset

def main():
    root_path = "."
    dataset = organize_dataset(root_path)

    features = {
        "image_paths": "string",
        "labels": label,
        # "details": "dict"
    }

    dataset["train"] = Dataset.from_dict(dataset["train"])
    dataset["test"] = Dataset.from_dict(dataset["test"])
    hf_dataset = DatasetDict(dataset)


    print(hf_dataset)
    print(hf_dataset["train"].features)
    print(hf_dataset["train"]["labels"][:5])
    print(hf_dataset["train"]["image_paths"][:5])
    # hf_dataset.save_to_disk("/path/to/save/hf_dataset")

if __name__ == "__main__":
    main()
