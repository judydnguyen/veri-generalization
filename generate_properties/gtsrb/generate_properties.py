import argparse
import numpy as np
import os
import shutil
import onnxruntime as rt
import re
from PIL import Image
import csv

DEFAULT_EPSILON = [1, 3, 5, 10, 15]
DEFAULT_NETWORK = ['./3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx',
                   './3_48_48_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_BN_Dense_256_BN_Dense_43_ep_30.onnx',
                   './3_64_64_QConv_32_5_MP_2_BN_QConv_64_5_MP_2_BN_QConv_64_3_MP_2_BN_Dense_1024_BN_Dense_43_ep_30.onnx']


def read_csv(file_path):
    """
    Read GTSRB Test.csv file.
    Expected format: Filename, Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId
    Returns: (labels, image_paths)
    """
    labels = []
    image_paths = []

    with open(file_path, 'r', newline='') as csv_file:
        sample = csv_file.read(2048)
        csv_file.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            dialect = csv.excel
        csv_reader = csv.reader(csv_file, dialect)
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            if len(row) >= 8:
                # GTSRB Test.csv format: Filename, Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId
                # row[0] = Filename, row[7] = ClassId
                image_paths.append(row[0])  # Filename
                labels.append(int(row[7]))  # ClassId

    return np.array(labels), np.array(image_paths)


def get_testing_dataset(img_size, dataset_dir='GTSRB_dataset', csv_name='Test.csv'):
    """
    Load GTSRB test dataset and preprocess images.
    
    Args:
        img_size: Target image size (height, width)
        dataset_dir: Directory containing Test.csv and images
    
    Returns:
        X_test: numpy array of shape (N, H, W, C) with values in [0, 255]
        y_test: numpy array of labels
    """
    csv_path = os.path.join(dataset_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GTSRB Test.csv not found at: {csv_path}")
    
    y_test, img_paths = read_csv(csv_path)

    data = []
    image_roots = [
        dataset_dir,
        os.path.join(dataset_dir, "GTSRB", "Final_Test", "Images"),
        os.path.join(dataset_dir, "Final_Test", "Images"),
        os.path.join(dataset_dir, "Images"),
    ]
    image_root = None
    for root in image_roots:
        if len(img_paths) > 0 and os.path.exists(os.path.join(root, img_paths[0])):
            image_root = root
            break
    if image_root is None:
        raise FileNotFoundError(
            f"Could not locate GTSRB images for {csv_path}. Tried roots: {image_roots}"
        )

    for img_path in img_paths:
        full_path = os.path.join(image_root, img_path)
        if not os.path.exists(full_path):
            print(f"Warning: Image not found: {full_path}")
            continue
        image = Image.open(full_path).convert("RGB")
        image = image.resize(img_size)
        # Convert to numpy array: shape (H, W, C) with values in [0, 255]
        data.append(np.array(image))
    
    if len(data) == 0:
        raise ValueError(f"No images loaded from {dataset_dir}")
    
    X_test = np.array(data).astype(np.float32)  # Shape: (N, H, W, C), values in [0, 255]

    return X_test, y_test


def write_vnn_spec(x_ds, y_ds, index, eps, dir_path="./", prefix="spec", data_lb=0, data_ub=1, n_class=10, mean=0.0, std=1.0, negate_spec=False):
    x = x_ds[index]  # Shape: (H, W, C), values in [0, 255]
    y = y_ds[index]
    x = np.array(x)
    
    # Apply epsilon perturbation in [0, 255] range
    x_lb = np.clip(x - eps, data_lb, data_ub)
    x_ub = np.clip(x + eps, data_lb, data_ub)
    
    # Normalize if needed (mean=0, std=1 means no normalization)
    if not (np.all(mean == 0.0) and np.all(std == 1.0)):
        x_lb = (x_lb - mean) / std
        x_ub = (x_ub - mean) / std
    
    # Flatten to 1D: (H, W, C) -> (H*W*C,)
    x_lb = x_lb.reshape(-1)
    x_ub = x_ub.reshape(-1)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Convert epsilon to fraction format (e.g., 1 -> 1over255, 3 -> 3over255)
    eps_num = int(eps)
    eps_str = f"{eps_num}over255"

    if np.all(mean == 0.) and np.all(std == 1.):
        spec_name = f"{prefix}_idx_{index}_{eps_str}.vnnlib"
    else:
        existing_specs = os.listdir(dir_path)
        competing_norm_ids = [int(re.match(f"{prefix}_idx_{index}_{eps_str}_n([0-9]+).vnnlib", spec).group(
            1)) for spec in existing_specs if spec.startswith(f"{prefix}_idx_{index}_{eps_str}_n")]
        norm_id = 1 if len(competing_norm_ids) == 0 else max(
            competing_norm_ids)+1
        spec_name = f"{prefix}_idx_{index}_{eps_str}_n{norm_id}.vnnlib"

    spec_path = os.path.join(dir_path, spec_name)

    with open(spec_path, "w") as f:
        # Convert epsilon to fraction format for display
        eps_fraction = eps / 255.0
        f.write(f"; Spec for sample id {index} and epsilon {eps_fraction:.6f} ({eps}/255)\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

        f.write(f"\n; Definition of output constraints\n")
        if negate_spec:
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"(assert (<= Y_{i} Y_{y}))\n")
        else:
            f.write(f"(assert (or\n")
            for i in range(n_class):
                if i == y:
                    continue
                f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
            f.write(f"))\n")
    return spec_name


def get_sample_idx(n, seed=42, n_max=10000):
    np.random.seed(seed)
    assert n <= n_max, f"only {n_max} samples are available"
    idx = list(np.random.choice(n_max, n, replace=False))
    return idx


def get_all_spec(n, seed, x_test, y_test, sess, input_name, img_size, epsilon, negate_spec, dont_extend, network, instances, new_instances, time_out):
    mean = np.array(0.0).reshape((1, -1, 1, 1)).astype(np.float32)
    std = np.array(1.0).reshape((1, -1, 1, 1)).astype(np.float32)

    idxs = get_sample_idx(n, seed, n_max=len(x_test))
    spec_path = os.path.dirname(os.path.abspath(__file__)) + "/gtsrb"

    i = 0
    ii = 1
    n_ok = 0
    with open(instances, "w" if new_instances else "a") as f:
        while i < len(idxs):
            idx = idxs[i]
            i += 1
            x = x_test[idx]  # Shape: (H, W, C), values in [0, 255]
            y = y_test[idx]
            
            # This model expects (batch, height, width, channels) format
            # Keep as (H, W, C) and add batch dimension
            x_new = x[np.newaxis, ...]  # (1, H, W, C)
            
            # Values are in [0, 255] range - keep as is (model expects this based on data_lb=0, data_ub=255)
            pred_onx = sess.run(None, {input_name: x_new})[0]
            y_pred = np.argmax(pred_onx, axis=-1)
            n_ok += (y == y_pred)

            if y == y_pred:
                if epsilon == None:
                    for eps in DEFAULT_EPSILON:
                        spec_i = write_vnn_spec(x_test, y_test, idx, eps, dir_path=spec_path, prefix="model_"+str(img_size),
                                                data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=negate_spec)
                        spec_abs = os.path.abspath(os.path.join(spec_path, spec_i))
                        f.write(f"{spec_abs}\n")
                else:
                    spec_i = write_vnn_spec(x_test, y_test, idx, epsilon, dir_path=spec_path, prefix="model_"+str(img_size),
                                            data_lb=0, data_ub=255, n_class=43, mean=mean, std=std, negate_spec=negate_spec)
                    spec_abs = os.path.abspath(os.path.join(spec_path, spec_i))
                    f.write(f"{spec_abs}\n")
            elif not dont_extend:
                # only sample idxs while there are still new samples to be found
                if len(idxs) < len(x_test):
                    tmp_idx = get_sample_idx(
                        1, seed=seed+ii, n_max=len(x_test))
                    ii += 1
                    while tmp_idx in idxs:
                        tmp_idx = get_sample_idx(
                            1, seed=seed + ii, n_max=len(x_test))
                        ii += 1
                    idxs.append(*tmp_idx)
        print(f"{len(idxs)-n_ok} samples were misclassified{''if dont_extend else ' and replacement samples drawn'}.")


def get_img_size(network):
    pattern = r"\d+_(\d+)_(\d+)"
    match = re.search(pattern, network)

    if match:
        return int(match.group(2))
    else:
        return None


def process_network(network, n, seed, epsilon, negate_spec, dont_extend, instances, new_instances, time_out):
    instances_dir = os.path.dirname(instances)
    if not os.path.isdir(instances_dir):
        os.mkdir(instances_dir)
        
    if new_instances and os.path.exists('./vnnlib'):
        shutil.rmtree('./vnnlib')

    sess = rt.InferenceSession(network)
    input_name = sess.get_inputs()[0].name

    img_size = get_img_size(network)

    print(
        f"Generating {n} random specs using seed {seed}. Model: {network}")
    
    # Try to find GTSRB dataset in common locations
    dataset_dirs = [
        'GTSRB_dataset',
        '../GTSRB_dataset',
        '../../GTSRB_dataset',
        '../../datasets/GTSRB',
        '../../datasets/GTSRB/Test/gtsrb',
    ]
    dataset_dir = None
    csv_name = None
    for d in dataset_dirs:
        for candidate in ('Test.csv', 'GT-final_test.csv', 'GT-final_test.test.csv'):
            if os.path.exists(os.path.join(d, candidate)):
                dataset_dir = d
                csv_name = candidate
                break
        if dataset_dir is not None:
            break
    
    if dataset_dir is None:
        raise FileNotFoundError(
            f"Could not find GTSRB dataset. Tried: {dataset_dirs}\n"
            f"Please ensure GTSRB_dataset/Test.csv exists in one of these locations."
        )
    
    print(f"Using GTSRB dataset from: {dataset_dir} ({csv_name})")
    x_test, y_test = get_testing_dataset((img_size, img_size), dataset_dir, csv_name=csv_name)

    get_all_spec(n, seed, x_test, y_test, sess,
                 input_name, img_size, epsilon, negate_spec, dont_extend, network, instances, new_instances, time_out)


def main():
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epsilon', type=float, default=None,
                        help='The epsilon for L_infinity perturbation')
    parser.add_argument('--n', type=int, default=3,
                        help='The number of specs to generate')
    parser.add_argument('seed', type=int, default=42,
                        help='Random seed for idx generation')
    parser.add_argument("--network", type=str, default=None,
                        help="Network to evaluate as .onnx file.")
    parser.add_argument('--negate_spec', action="store_true", default=False,
                        help='Generate spec that is violated for correct certification')
    parser.add_argument('--dont_extend', action="store_true", default=False,
                        help='Do not filter for naturally correctly classified images')
    parser.add_argument("--instances", type=str,
                        default="./instances.csv", help="Path to instances file")
    parser.add_argument("--new_instances", action="store_true",
                        default=True, help="Overwrite old instances.csv")
    parser.add_argument('--time_out', type=float, default=480.0,
                        help='time out')

    args = parser.parse_args()

    if args.network is not None:
        process_network(args.network, args.n, args.seed,
                        args.epsilon, args.negate_spec, args.dont_extend, args.instances, args.new_instances, args.time_out)
    else:
        first_time = True
        for network in DEFAULT_NETWORK:
            process_network(network, args.n, args.seed,
                            args.epsilon, args.negate_spec, args.dont_extend, args.instances, (args.new_instances and first_time), args.time_out)
            first_time = False


if __name__ == "__main__":
    main()