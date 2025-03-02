import torch
import tifffile
import os
from pathlib import Path
import time
from einops import rearrange


def majority_vote_v1(preds):
    return -1 * (torch.mode(-torch.cat(preds, dim=0), dim=0))[0]


def majority_vote(preds):
    # Decide whether its background or matrix
    mito = torch.mode(preds > 0, dim=0)[0]

    # Base prediction with the tie breaking we want (for non conflict areas)
    mode_preds = -1 * (torch.mode(-preds, dim=0))[0]

    # Find the areas in which the mode predicts background even though we want mito
    conflict = torch.logical_and(mode_preds == 0, mito == 1)

    # We count the number of times each class appears and give a 0.5 boost to M and C if A
    total_counts = torch.zeros(5, *preds.shape[1:], device=preds.device)
    for label in [0, 1, 2, 3, 4]:
        counts = torch.sum(preds == label, dim=0)
        total_counts[label] = counts
    total_counts[3] += 0.5 * total_counts[4]
    total_counts[2] += 0.5 * total_counts[4]
    inner_pred = torch.argmax(total_counts[1:], dim=0) + 1.0  # Ignore the background

    # Change these conflict areas to the inner prediction
    mode_preds[conflict] = inner_pred[conflict]
    return mode_preds


def gpu_majorityVote(preds, batch_size=10):
    mv = []
    for slice in range(0, preds.shape[1], batch_size):
        mv.append(majority_vote(preds[:, slice : (slice + batch_size)].cuda()).cpu())
    return torch.cat(mv, dim=0)


def process_runs(path_runs, path_target):
    """
    Load in the predictions from all 5 runs and compute the majority vote
    """
    run_dirs = [f for f in os.listdir(os.path.dirname(path_runs)) if f.startswith("Run") and not f.startswith(".")]
    all_runs = list(range(1, len(run_dirs) + 1))
    # all_runs = [1,2,3,4,5,6]
    if not os.path.exists(path_target):
        os.makedirs(path_target, exist_ok=True)
    path_run = lambda run: f"{path_runs}{run}"
    dataset_names = sorted([f for f in os.listdir(path_run(1)) if f.endswith(".tif") and not f.startswith(".")])

    for dataset in dataset_names:
        time.sleep(os.getpid() % 10 / 10)  # To avoid that all processes start at the same time
        output_name = os.path.join(path_target, dataset)
        if os.path.exists(output_name) or not all(
            [os.path.exists(os.path.join(path_run(run), dataset)) for run in all_runs]
        ):
            continue
        else:
            Path(output_name).touch()

        print(f"{time.strftime('%d/%m/%Y %H:%M:%S')} Processing {dataset}")
        ind_preds_files = [os.path.join(path_run(run), dataset) for run in all_runs]
        preds = [torch.Tensor(tifffile.imread(ind_pred).astype("float"))[None] for ind_pred in ind_preds_files]
        preds = torch.cat(preds, dim=0)

        # result =majority_vote(preds)
        result = gpu_majorityVote(preds)
        tifffile.imwrite(
            output_name,
            result.numpy().astype("uint8"),
            imagej=True,
            metadata={"axes": "ZYX"},
            compression="zlib",
        )


def process_Avoid_Ambig(path_runs, path_mv, path_target):
    """
    Loads in the predictions of the models that are trained to avoid the ambiguous class
    and loads the majority vote as well.
    Whenever the majority vote of the dice-trainemodels is 'Ambiguous', we check the models
    trained to avoid ambiguous and replace the ambiguous values with the majority vote between
    matrix and cristae of the confident models.
    """

    def decide_matrix_vs_cristae(preds):
        total_counts = torch.zeros(5, *preds.shape[1:])
        # Only count the matrix or cristae votes
        for label in [2, 3, 4]:
            counts = torch.sum(preds == label, dim=0)
            total_counts[label] = counts
        total_counts[3] += 0.5 * total_counts[4]
        total_counts[2] += 0.5 * total_counts[4]

        inner_pred = torch.argmax(total_counts[1:], dim=0) + 1.0  # Ignore the background

        # if there are 0 votes for matrix or cristae and the inner_pred 1,
        # we return stick with ambiguous
        inner_pred[inner_pred == 1] = 4

        return inner_pred

    def merge_MV_Ambig(preds, mv):
        ambiguous_mask = mv == 4  # Shape (1, Z, Y, X)
        # But preds is shape (5, Z, Y, X)
        ambiguous_mask = rearrange(ambiguous_mask, "b z y x -> b (z y x)")
        preds = rearrange(preds, "b z y x -> b (z y x)")
        preds_shape = preds.shape
        preds = preds[ambiguous_mask.expand(*preds.shape)].reshape(preds_shape[0], -1)
        confident_mv = decide_matrix_vs_cristae(preds)
        mv[mv == 4] = confident_mv  # Ambiguous_mask has been reshaped, use mv ==4 instead
        return mv[0]

    path_run = lambda run: f"{path_runs}{run}"
    dataset_names = sorted([f for f in os.listdir(path_run(1)) if f.endswith(".tif") and not f.startswith(".")])

    for dataset in dataset_names:
        print(f"{time.strftime('%d/%m/%Y %H:%M:%S')} Processing {dataset}")
        preds = [
            torch.Tensor(tifffile.imread(os.path.join(path_run(run), dataset)).astype("float"))[None]
            for run in [1, 2, 3, 4, 5]
        ]
        preds = torch.cat(preds, dim=0)
        mv = torch.Tensor(tifffile.imread(os.path.join(path_mv, dataset)).astype("float"))[None]
        result = merge_MV_Ambig(preds, mv)
        tifffile.imwrite(
            os.path.join(path_target, dataset),
            result.numpy().astype("uint8"),
            imagej=True,
            metadata={"axes": "ZYX"},
            compression="zlib",
        )


if __name__ == "__main__":
    output_dir = "YOUR_OUTPUT_FOLDER"

    process_runs(
        path_runs=f"{output_dir}/Run",
        path_target=f"{output_dir}/MajorityVote",
    )
