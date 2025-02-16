import os
from pathlib import Path
import tifffile
import skimage
from skimage.morphology import binary_closing, binary_opening, binary_dilation, ball, cube
from skimage.segmentation import expand_labels
from scipy import ndimage
import numpy as np
from numba import njit
import time


def find_mitos(annotation, verbose=False):
    if verbose:
        print(f"{time.strftime('%H:%M:%S')} Starting to find separate mitos", flush=True)
    inner_structure = annotation > 1
    inner_structure = binary_opening(inner_structure, cube(3))
    mitos = skimage.measure.label(inner_structure)
    if verbose:
        print(f"{time.strftime('%H:%M:%S')} Starting to compute distance transform", flush=True)
    distance = ndimage.distance_transform_cdt(mitos > 0)
    if verbose:
        print(f"{time.strftime('%H:%M:%S')} Starting to compute watershed", flush=True)
    final = skimage.segmentation.watershed(-distance, mitos, mask=annotation > 0)
    return final


# Lets first extract the enclosing rectangel to make the closings more efficient
@njit
def find_ind_extreme_coords(img, n_mitos):
    # indices is going to hold the min and max coordinate. Each row corresponds to one mito label
    indices = np.zeros((n_mitos, 2, 3))
    indices[:, 0] = np.Inf
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                cur_label = img[i, j, k]
                if cur_label != 0:
                    cur_index = np.array([i, j, k], dtype="float64")
                    cur_min, cur_max = indices[cur_label]
                    indices[cur_label, 0] = np.minimum(cur_index, cur_min)
                    indices[cur_label, 1] = np.maximum(cur_index, cur_max)
    return indices


def rm_fragments(img, mitos=None, threshold=0.25):
    if mitos is None:
        mitos = skimage.measure.label(img > 0)
    unique_mitos = np.unique(mitos)

    n_mitos = unique_mitos.max() + 1
    indices = find_ind_extreme_coords(mitos, n_mitos).astype("uint32")

    fragmented_percantages = []
    counter = 0
    for c_mito_label in unique_mitos[1:]:
        cmin, cmax = indices[c_mito_label]
        cmax += 1  # Make sure to include the maximum pixel of the bounding box
        # Find the binary mask of the current mito
        c_mito_cub = mitos[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] == c_mito_label
        orig_cub = img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]]

        # Examinate the change that closing introduces to the current mitochondria
        c_mito_closing = binary_closing(c_mito_cub, ball(5))
        dif = np.logical_xor(c_mito_cub, c_mito_closing)
        prop = dif.sum() / c_mito_closing.sum()
        fragmented_percantages.append(prop)
        if prop > threshold:
            # Set the current mito to 0
            orig_cub[c_mito_cub] = 0

            # Insert the modified cuboid in its original place
            img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] = orig_cub
            counter += 1

    print(f"Removed {counter} from {len(unique_mitos)} components because fragmented.")
    return img, fragmented_percantages


def find_ind_to_remove(img, labels_small, n_indices):
    if len(img.shape) == 3:
        return _find_ind_to_remove3d(img, labels_small, n_indices)
    elif len(img.shape) == 2:
        return _find_ind_to_remove2d(img, labels_small, n_indices)
    else:
        raise ValueError("Image must be 2d or 3d")


@njit
def _find_ind_to_remove3d(img, labels_small, n_indices):
    """
    img to search through
    labels_small: boolean array encoding for every label whether its a small label
    n_indices: Number of small indices we

    Returns an np.array of shape [n_indices, 3] where every row corresponds to
    the coordinates of an index that should get removed.
    """
    counter = 0
    indices = np.zeros((n_indices, 3), dtype="uint16")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if labels_small[img[i, j, k]]:
                    indices[counter] = np.array([i, j, k], dtype="uint16")
                    counter += 1
    return indices


@njit
def _find_ind_to_remove2d(img, labels_small, n_indices):
    """
    img to search through
    labels_small: boolean array encoding for every label whether its a small label
    n_indices: Number of small indices we

    Returns an np.array of shape [n_indices, 2] where every row corresponds to
    the coordinates of an index that should get removed.
    """
    counter = 0
    indices = np.zeros((n_indices, 2), dtype="uint16")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if labels_small[img[i, j]]:
                indices[counter] = np.array([i, j], dtype="uint16")
                counter += 1
    return indices


def remove_small_parts(img, mitos=None, threshold=3200):
    """
    Searches for connected components in image and remove all of them
    which are smaller than <threshold> pixels
    """
    if mitos is None:
        img_binary = (img > 0).astype("uint8")
        mitos, n_mitos = skimage.measure.label(img_binary, return_num=True)
    _, mito_size = np.unique(mitos, return_counts=True)

    labels_small = mito_size < threshold
    n_indices = np.sum(mito_size[mito_size < threshold])
    print(f"Removing {np.sum(labels_small)} from {len(mito_size)} connected components because size < {threshold}")

    indices = find_ind_to_remove(mitos, labels_small, n_indices)
    indices = (indices[:, 0], indices[:, 1], indices[:, 2])

    img[indices] = 0
    mitos[indices] = 0
    return img, mitos


def remove_small_parts_slice(img, threshold=150):
    """
    Searches for connected components in each slice and remove all of them
    which are smaller than <threshold> pixels
    """
    for slice_n in range(img.shape[0]):
        slice = img[slice_n]
        components = skimage.measure.label(slice > 0, connectivity=1)  # Ignore diagonal connections
        _, components_size = np.unique(components, return_counts=True)

        labels_small = components_size < threshold
        n_indices = np.sum(components_size[components_size < threshold])
        # print(f'Removing {np.sum(labels_small)} from {len(components_size)} connected components because size < {threshold}')

        indices = find_ind_to_remove(components, labels_small, n_indices)
        indices = (indices[:, 0], indices[:, 1])

        slice[indices] = 0
        img[slice_n] = slice
    return img


def remove_boundary_noise(img, n_pixel):
    """Sometimes voxel at the very boundy would show artifacts which gets removed here"""
    img[:, :n_pixel] = 0
    img[:, -n_pixel:] = 0
    img[:, :, :n_pixel] = 0
    img[:, :, -n_pixel:] = 0
    return img


def remove_mitos_with_only_or_no_intermembrane(img, mitos, percentage_threshold_upper, percentage_threshold_lower):
    """
    Remove artifacts which contain no intermembrane space or only consisten of
    intermembrane space which is both biologically not meaningful.
    """
    unique_mitos = np.unique(mitos)

    n_mitos = unique_mitos.max() + 1
    indices = find_ind_extreme_coords(mitos, n_mitos).astype("uint32")
    # If there are some mitos deleted and len(unique_mitos) is less than n_mitos
    # I will get RuntimeWarning: invalid value encountered in cast
    # as there will be some inf values in find_ind_extreme_coords) which will be casted to 0
    # However, I won't use them later, so it doesnt matter.

    counter = 0
    for c_mito_label in unique_mitos[1:]:
        cmin, cmax = indices[c_mito_label]
        cmax += 1  # Make sure to include the maximum pixel of the bounding box
        # Find the binary mask of the current mito
        c_mito_cube = mitos[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] == c_mito_label
        annotated_cube = img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]].copy()

        # Calculate the percentage of intermembrane pixels
        annotated_cube[np.logical_not(c_mito_cube)] = 0  # Set all non-mito pixels to 0
        label_dist = np.bincount(annotated_cube.flatten(), minlength=4)
        percentage_intermembrane = label_dist[1] / label_dist[1:].sum()

        if (
            percentage_intermembrane > percentage_threshold_upper
            or percentage_intermembrane < percentage_threshold_lower
        ):
            # Set the current mito to 0
            orig_cube = img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]]
            orig_cube[c_mito_cube] = 0

            orig_label = mitos[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]]
            orig_label[c_mito_cube] = 0

            # Insert the modified cuboid in its original place
            img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] = orig_cube
            mitos[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] = orig_label
            counter += 1

    print(f"Removed {counter} from {len(unique_mitos)} components because no inner structure.")
    return img, mitos


def close_intermembrane_holes(img, mitos=None):
    """
    When we have small pieces of intermembrane space within the mito, we replace it with cristae
    """
    img = img.copy()
    if mitos is None:
        mitos = skimage.measure.label(img > 0)
    unique_mitos = np.unique(mitos)

    n_mitos = unique_mitos.max() + 1
    indices = find_ind_extreme_coords(mitos, n_mitos).astype("uint32")

    for c_mito_label in unique_mitos[1:]:
        cmin, cmax = indices[c_mito_label]
        cmax += 1  # Make sure to include the maximum pixel of the bounding box
        # Find the binary mask of the current mito
        c_mito_cub = mitos[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] == c_mito_label
        orig_cub = img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]].copy()
        orig_cub[c_mito_cub == 0] = 0

        # Find segments of intermembrane space and their boundaries
        ib_seg, n_seg = skimage.measure.label(orig_cub == 1, return_num=True)
        ib_seg_boundary = expand_labels(ib_seg, distance=2)
        ib_seg_boundary[ib_seg > 0] = 0

        # Set the segments to zero where the extended boundary mostly extends into the mito
        boundary_pixels = ib_seg_boundary > 0
        seg_boundary_labels = orig_cub[boundary_pixels] > 1

        for ib_seg_label in range(1, n_seg):
            cur_boundary_labels = seg_boundary_labels[ib_seg_boundary[boundary_pixels].flatten() == ib_seg_label]
            if np.sum(cur_boundary_labels) / len(cur_boundary_labels) > 0.9:
                orig_cub[ib_seg == ib_seg_label] = 2  # Set to cristae

        # Replace the original with the new one
        final_cub = img[
            cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]
        ].copy()  # Copy the whole cube to make sure we have the right other mitos there
        final_cub[c_mito_cub == 1] = orig_cub[c_mito_cub == 1]  # Change the current mito
        img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] = final_cub

    return img


def close_background_holes(img, mitos=None):
    """
    When we have small pieces of background within the mito, we replace it with a watershed
    """
    img = img.copy()
    if mitos is None:
        mitos = skimage.measure.label(img > 0)
    unique_mitos = np.unique(mitos)

    n_mitos = unique_mitos.max() + 1
    indices = find_ind_extreme_coords(mitos, n_mitos).astype("uint32")

    for c_mito_label in unique_mitos[1:]:
        cmin, cmax = indices[c_mito_label]
        cmax += 1  # Make sure to include the maximum pixel of the bounding box
        # Find the binary mask of the current mito
        c_mito_cub = mitos[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] == c_mito_label
        orig_cub = img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]].copy()
        orig_cub[c_mito_cub == 0] = 0

        # TODO: Find and replace background holes smaller than some threshold

        # Replace the original with the new one
        final_cub = img[
            cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]
        ].copy()  # Copy the whole cube to make sure we have the right other mitos there
        final_cub[c_mito_cub == 1] = orig_cub[c_mito_cub == 1]  # Change the current mito
        img[cmin[0] : cmax[0], cmin[1] : cmax[1], cmin[2] : cmax[2]] = final_cub

    return img


def add_IB_coating(pred, radius=10):
    """Matrix and Cristae should not directly touch background without intermembrane space around it"""
    dil_pred = binary_dilation(pred > 1, ball(radius))
    mask = np.logical_and(pred == 0, dil_pred > 0)
    pred[mask] = 1  # Make sure every mitochondrion is covered in intermembrane space
    return pred


def process_img(img, fragment_threshold=0.25, plot_hist=False):
    img = remove_boundary_noise(img, n_pixel=5)
    mitos = find_mitos(img)
    img, mitos = remove_small_parts(img, mitos, threshold=25000)
    img, mitos = remove_mitos_with_only_or_no_intermembrane(
        img, mitos, percentage_threshold_upper=0.80, percentage_threshold_lower=0.25
    )
    img, frag_perc = rm_fragments(img, mitos, threshold=fragment_threshold)
    img, _ = remove_small_parts(
        img, threshold=10000
    )  # Somehow some small mitos survive (might be because they get removed in the opnening)
    img = remove_small_parts_slice(img, threshold=250)

    img = add_IB_coating(img, 3)
    img = close_intermembrane_holes(img)

    return img


def process_file(file, path_source, path_target, fragment_threshold=0.25, plot_hist=False):
    time.sleep(os.getpid() % 10 / 10)  # To avoid that all processes start at the same time
    output_file = os.path.join(path_target, file)
    if os.path.exists(output_file):
        print(f"File {file} already exists and is skipped.")
        return
    else:
        Path(output_file).touch()
    print(time.strftime("%H:%M:%S"), "Started: ", file, f"with threshold {fragment_threshold}", flush=True)
    img = tifffile.imread(os.path.join(path_source, file))
    img = process_img(img, fragment_threshold=fragment_threshold, plot_hist=plot_hist)
    tifffile.imwrite(
        output_file,
        img,
        imagej=True,
        metadata={"axes": "ZYX"},
        compression="zlib",
    )

    print(time.strftime("%H:%M:%S"), "Finished:", file, flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path_source", type=str)
    argparser.add_argument("--path_target", type=str)

    args = argparser.parse_args()

    files = sorted([f for f in os.listdir(args.path_source) if f.endswith(".tif") and not f.startswith(".")])
    print(f"Files to process: {len(files)}")

    if not os.path.exists(args.path_target):
        os.makedirs(args.path_target, exist_ok=True)

    # All files
    def process_file_wrapper(file):
        process_file(file, args.path_source, args.path_target, fragment_threshold=0.25, plot_hist=False)

    with mp.Pool(4) as pool:
        pool.map(process_file_wrapper, files)
