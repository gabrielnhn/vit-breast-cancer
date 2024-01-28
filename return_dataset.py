import os
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
from PIL import Image

class_labels = ["Benign", "Malignant"]
label = ClassLabel(names=class_labels)

def read_images_from_folder(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                images.append(os.path.join(root, file))


    return images
    # return [Image.open(x).convert("RGB") for x in images], images

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
            dataset[split] = {"image": [], "prelabels": [], "details": []}
            for magnification in ["40X", "100X", "200X", "400X"]:
                folder_path = os.path.join(root_path, f"fold{fold}", split, magnification)
                # print(folder_path)
                # images, image_paths = read_images_from_folder(folder_path)
                image_paths = read_images_from_folder(folder_path)
                dataset[split]["image"].extend(image_paths)

                details, labels = zip(*[parse_filename(os.path.basename(image_path)) for image_path in image_paths])
                # dataset[split]["labels"].extend(labels)
                dataset[split]["prelabels"].extend(labels)
                dataset[split]["details"].extend(details)

           
            dataset[split]["labels"] = label.str2int(dataset[split]["prelabels"])
            dataset[split].pop("prelabels")
            

    return dataset

def main():
    root_path = "folds"
    dataset = organize_dataset(root_path)

    # features = Features({
    #     'image': Value(dtype='string', id=None),
    #     'details': {
    #         'BIOPSY_PROCEDURE': Value(dtype='string', id=None),
    #         'MAGNIFICATION': Value(dtype='string', id=None),
    #         'SEQ': Value(dtype='string', id=None),
    #         'SLIDE_ID': Value(dtype='string', id=None),
    #         'TUMOR_CLASS': Value(dtype='string', id=None),
    #         'TUMOR_TYPE': Value(dtype='string', id=None),
    #         'YEAR': Value(dtype='string', id=None)
    #         },
    #     'labels': Value(dtype='int64', id=None)}
    # ),
    features = Features({
        'image': Value(dtype='string'),
        'details': {
            'BIOPSY_PROCEDURE': Value(dtype='string'),
            'MAGNIFICATION': Value(dtype='string'),
            'SEQ': Value(dtype='string'),
            'SLIDE_ID': Value(dtype='string'),
            'TUMOR_CLASS': Value(dtype='string'),
            'TUMOR_TYPE': Value(dtype='string'),
            'YEAR': Value(dtype='string'),
        },
        'labels': ClassLabel(names=class_labels)
    })

    dataset["train"] = Dataset.from_dict(dataset["train"], features=features)
    dataset["test"] = Dataset.from_dict(dataset["test"], features=features)


    hf_dataset = DatasetDict(dataset)

    print(hf_dataset)
    print(hf_dataset["train"].features)
    print(hf_dataset["train"]["labels"][:5])
    print(hf_dataset["train"]["image"][:5])
    hf_dataset.save_to_disk("./breakhis.hf")

if __name__ == "__main__":
    main()
