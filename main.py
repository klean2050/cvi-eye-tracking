import os


if __name__ == "__main__":
    root = "/home/kavra/Datasets/medical/cvi_eyetracking/asc_data_v1/"

    ids = [i for i in os.listdir(root) if i.endswith(".asc")]
    ctrl_ids = [i.split(".")[0] for i in ids if i.split("_")[0].startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i not in ctrl_ids]
    names = ctrl_ids + cvi_ids

    trials_images_subset = [
        "Freeviewingstillimage_1.jpg",
        "Freeviewingstillimage_2.jpg",
        "Freeviewingstillimage_4.jpg",
        "Freeviewingstillimage_5.jpg",
        "Freeviewingstillimage_7.jpg",
        "Freeviewingstillimage_8.jpg",
        "Freeviewingstillimage_9.jpg",
        "Freeviewingstillimage_10.jpg",
        "Freeviewingstillimage_10_cutout.tif",
        "Moviestillimage_8.jpg",
        "Moviestillimage_6.jpg",
        "Freeviewingstillimage_50.jpg",
        "Freeviewingstillimage_88_cutout.tif",
    ]
    trial, vel = "Freeviewingstillimage_1.jpg", False

    compare_trials = [
        ["Freeviewingstillimage_36.jpg", "Freeviewingstillimage_36_cutout.tif"],
        ["Freeviewingstillimage_28.jpg", "Freeviewingstillimage_28_cutout.tif"],
        ["Freeviewingstillimage_93.jpg", "Freeviewingstillimage_93_cutout.tif"],
        ["Freeviewingstillimage_36.jpg", "Freeviewingstillimage_36_cutout.tif"],
        ["Freeviewingstillimage_10.jpg", "Freeviewingstillimage_10_cutout.tif"],
    ]
