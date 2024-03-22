from process import *
from saliency import *
from utils import *

from tqdm import tqdm


def GaussianMask(sizex, sizey, sigma=33, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:
            x0 = center[0]
            y0 = center[1]
        else:
            return np.zeros((sizey, sizex))

    # create a 2d gaussian centered at (x0, y0)
    return fix * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def Fixpos2Densemap(fix_arr, width, height, imgfile, alpha=0.5, threshold=0):
    """
    fix_arr   : fixation array number of subjects x 3(x,y,fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : marge rate imgfile and heatmap (optional)
    threshold : heatmap threshold(0~255)
    return heatmap
    """

    heatmap = np.zeros((height, width), np.float32)
    for n_subject in tqdm(range(fix_arr.shape[0])):
        heatmap += GaussianMask(
            width,
            height,
            31,
            (fix_arr[n_subject, 0], fix_arr[n_subject, 1]),
            fix_arr[n_subject, 2],
        )

    # Normalization
    heatmap = heatmap * 255 / np.amax(heatmap)
    heatmap = heatmap.astype("uint8")

    if imgfile.any():
        # Resize heatmap to imgfile shape
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Create mask
        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Marge images
        marge = imgfile * mask + heatmap_color * (1 - mask)
        marge = marge.astype("uint8")
        marge = cv2.addWeighted(imgfile, 1 - alpha, marge, alpha, 0)
        return marge

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap


def get_fixation_points(trial_name, root, ids):
    """
    trial_path : path to trial folder
    ids        : subject ids
    return fixation points array
    """
    fixations = []
    for i in ids:
        sub = Subject(root[i], i)
        fixs = sub.extract_fixations(trial_name)
        analyzer = FixationAnalyzer(root[i], fixs)

        fix_list = [fix["data"] for fix in analyzer.fixations if len(fix["data"]) > 50]
        fix_list = (
            [[np.mean(fix[:, 0]), np.mean(fix[:, 1])] for fix in fix_list]
            if "new_res" not in root[i]
            else [
                [np.mean(fix[:, 0]) - 320, np.mean(fix[:, 1]) - 240] for fix in fix_list
            ]
        )
        fixations.extend(fix_list)

    fix_arr = np.array(fixations)
    # convert to 3D array (compatible with Fixpos2Densemap)
    fix_arr = np.hstack((fix_arr, np.ones((fix_arr.shape[0], 1))))
    return fix_arr


if __name__ == "__main__":
    trial_name = "Freeviewingstillimage_10"

    # load trial image
    this_trial = ImageTrial(trial_name, "smaps")
    img = this_trial.load_trial_img()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    # subject ids
    ids1 = [i for i in os.listdir(DATA_ROOT1) if i.endswith(".asc")]
    ids = [i for i in os.listdir(DATA_ROOT2) if i.endswith(".asc")]

    # which path to use for each subject
    which_root = {
        name.split(".")[0]: DATA_ROOT1 if i < len(ids1) else DATA_ROOT2
        for i, name in enumerate(ids)
    }

    # distinguish controls from CVI subjects
    ctrl_ids = [i.split(".")[0] for i in ids if i.startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i.startswith("1")]

    # fixation points
    fix_arr_ctrl = get_fixation_points(trial_name, which_root, ctrl_ids)
    fix_arr_cvi = get_fixation_points(trial_name, which_root, cvi_ids)

    # plot heatmaps in same row
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    heatmap = Fixpos2Densemap(fix_arr_ctrl, W, H, img, 0.5, 15)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    plt.imshow(heatmap)

    plt.title(f"{trial_name} - CTRL")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    heatmap = Fixpos2Densemap(fix_arr_cvi, W, H, img, 0.5, 15)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    plt.imshow(heatmap)

    plt.title(f"{trial_name} - CVI")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.savefig(f"output/fmap_{trial_name}.png", dpi=300)
