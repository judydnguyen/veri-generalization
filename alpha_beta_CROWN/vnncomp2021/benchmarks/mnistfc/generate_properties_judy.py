import argparse
import os
import sys
import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader, sampler
import numpy as np
import onnxruntime as onnxrun


# def load_data(data_dir: str = "./tmp", num_imgs: int = 25, random: bool = True) -> tuple:
#     """
#     Loads MNIST data and returns correctly classified images across multiple ONNX models.
#     """
#     if not os.path.isdir(data_dir):
#         os.makedirs(data_dir, exist_ok=True)

#     trns_norm = trans.ToTensor()
#     mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)

#     if random:
#         loader_test = DataLoader(mnist_test, batch_size=10000,
#                                  sampler=sampler.SubsetRandomSampler(range(10000)))
#     else:
#         loader_test = DataLoader(mnist_test, batch_size=10000)

#     images, labels = next(iter(loader_test))

#     selected_images, selected_labels = [], []
#     num_selected = 0

#     # Load all model sessions
#     sessions = [
#         onnxrun.InferenceSession("./mnist-net_256x2.onnx"),
#         onnxrun.InferenceSession("./mnist-net_256x4.onnx"),
#         onnxrun.InferenceSession("./mnist-net_256x6.onnx")
#     ]

#     i = -1
#     while num_selected < num_imgs:
#         i += 1
#         correctly_classified = True
#         img_np = images[i].numpy().reshape(1, 784, 1)
#         lbl = labels[i].item()

#         for sess in sessions:
#             input_name = sess.get_inputs()[0].name
#             pred = np.argmax(sess.run(None, {input_name: img_np})[0])
#             if pred != lbl:
#                 correctly_classified = False
#                 break

#         if correctly_classified:
#             selected_images.append(images[i])
#             selected_labels.append(labels[i])
#             num_selected += 1

#     return selected_images, selected_labels


def load_data(
    data_dir: str = "./tmp",
    num_imgs: int = 25,
    random: bool = True,
    ensure_all_classes: bool = True,
    num_classes: int = 10,
) -> tuple:
    """
    Loads MNIST data and returns images that are correctly classified
    by all ONNX models. Optionally ensures every class appears at least once.
    """

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    trns_norm = trans.ToTensor()
    mnist_test = dset.MNIST(data_dir, train=False, download=True, transform=trns_norm)

    loader_test = DataLoader(mnist_test, batch_size=10000)
    images, labels = next(iter(loader_test))

    # -----------------------
    # Run all models once
    # -----------------------
    sessions = [
        onnxrun.InferenceSession("./mnist-net_256x2.onnx"),
        onnxrun.InferenceSession("./mnist-net_256x4.onnx"),
        onnxrun.InferenceSession("./mnist-net_256x6.onnx"),
    ]

    correct_indices_per_class = {c: [] for c in range(num_classes)}

    imgs_np = images.view(len(images), -1, 1).numpy()  # (N, 784, 1)
    lbls_np = labels.numpy()

    for idx in range(len(images)):
        x = imgs_np[idx:idx+1]  # shape (1, 784, 1)
        y_true = int(lbls_np[idx])

        all_correct = True
        for sess in sessions:
            input_name = sess.get_inputs()[0].name
            pred = np.argmax(sess.run(None, {input_name: x})[0])
            if pred != y_true:
                all_correct = False
                break

        if all_correct:
            correct_indices_per_class[y_true].append(idx)

    # -----------------------
    # Select indices
    # -----------------------
    selected_indices = []

    if ensure_all_classes:
        if num_imgs < num_classes:
            raise ValueError(
                f"num_imgs={num_imgs} but ensure_all_classes=True requires at least {num_classes}."
            )

        # 1) one sample per class (if possible)
        for c in range(num_classes):
            inds = correct_indices_per_class[c]
            if len(inds) == 0:
                print(f"⚠️ Warning: no jointly-correct examples found for class {c}.")
                continue
            chosen = np.random.choice(inds) if random else inds[0]
            selected_indices.append(chosen)

        # 2) remaining samples from the union pool
        remaining_needed = num_imgs - len(selected_indices)
        if remaining_needed > 0:
            all_inds = sorted({i for inds in correct_indices_per_class.values() for i in inds})
            remaining_pool = [i for i in all_inds if i not in selected_indices]
            if random:
                extra = np.random.choice(remaining_pool, size=remaining_needed, replace=False)
            else:
                extra = remaining_pool[:remaining_needed]
            selected_indices.extend(list(extra))
    else:
        # original behavior: just take any jointly-correct samples
        all_inds = []
        for c_inds in correct_indices_per_class.values():
            all_inds.extend(c_inds)
        all_inds = sorted(all_inds)
        if random:
            all_inds = list(np.random.permutation(all_inds))
        selected_indices = all_inds[:num_imgs]

    # -----------------------
    # Build tensors
    # -----------------------
    selected_images = [images[i] for i in selected_indices]
    selected_labels = [labels[i] for i in selected_indices]

    return selected_images, selected_labels


def create_input_bounds(img: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Creates lower/upper input bounds for a given image and epsilon.
    """
    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = torch.clip((img - eps), 0, 1)
    bounds[..., 1] = torch.clip((img + eps), 0, 1)
    return bounds.view(-1, 2)


def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):
    """
    Saves a classification property in VNN-LIB format.
    """
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)

    with open(spec_path, "w") as f:
        f.write(f"; MNIST property with label: {label}\n\n")

        # Declare input variables
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Input constraints
        f.write("; Input constraints\n")
        for i in range(input_bounds.shape[0]):
            lo, hi = input_bounds[i, 0], input_bounds[i, 1]
            f.write(f"(assert (<= X_{i} {hi}))\n")
            f.write(f"(assert (>= X_{i} {lo}))\n\n")

        # Output constraints
        f.write("; Output constraints\n")
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")


def create_instances_csv(num_props: int, epsilons, save_dir: str):
    """
    Creates an instances.csv file for all models and generated VNN-LIBs.
    """
    nets = ["mnist-net_256x2.onnx", "mnist-net_256x4.onnx", "mnist-net_256x6.onnx"]

    props = []
    for eps in epsilons:
        props.extend([f"prop_{i}_{eps:.2f}.vnnlib" for i in range(num_props)])

    csv_path = os.path.join(save_dir, "mnistfc_instances.csv")
    with open(csv_path, "w") as f:
        for net in nets:
            timeout = 120 if net == "mnist-net_256x2.onnx" else 300
            for prop in props:
                # line = f"{net},{prop},{timeout}\n"
                line = f"{prop}\n"
                f.write(line)

    print(f"✅ Instances CSV saved to {csv_path}")


def generate_vnnlib_dataset(
    epsilons,
    save_dir: str,
    num_images: int = 15,
    random: bool = True,
    seed: int | None = None,
):
    """
    Main entry: generates VNNLIB property files and instance CSV.

    Args:
        epsilons: list of float or single float
        save_dir: directory to save VNNLIB files and CSV
        num_images: number of MNIST samples
        random: whether to select images randomly
        seed: random seed for reproducibility
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(epsilons, (float, int)):
        epsilons = [float(epsilons)]

    if seed is not None:
        torch.random.manual_seed(seed)

    print(f"📦 Generating MNIST VNNLIB dataset in '{save_dir}' for eps={epsilons}")
    # images, labels = load_data(num_imgs=num_images, random=random)
    
    images, labels = load_data(
        num_imgs=num_images,
        random=True,
        ensure_all_classes=True
    )

    for eps in epsilons:
        for i in range(num_images):
            image, label = images[i], labels[i]
            input_bounds = create_input_bounds(image, eps)
            spec_path = os.path.join(save_dir, f"prop_{i}_{eps:.2f}.vnnlib")
            save_vnnlib(input_bounds, label, spec_path)

    create_instances_csv(num_images, epsilons, save_dir)
    print("✅ Done generating VNNLIB dataset.")


if __name__ == "__main__":
    # Example usage: python generate_vnnlib_dataset.py 123
    args = argparse.ArgumentParser()
    args.add_argument("--epsilons", type=float, default=[0.03, 0.05])
    args.add_argument("--save_dir", type=str, default="./mnist_vnnlib_out")
    args.add_argument("--num_images", type=int, default=20)
    args.add_argument("--random", type=bool, default=True)
    args.add_argument("--seed", type=int, default=42)
    args = args.parse_args()

    generate_vnnlib_dataset(
        epsilons=args.epsilons,
        save_dir=args.save_dir,
        num_images=args.num_images,
        random=args.random,
        seed=args.seed,
    )
    
    # example usage: python generate_properties_judy.py --epsilons 0.03 --save_dir ./vnnlib_props_0.03 --num_images 20 --random True --seed 42


# generate_vnnlib_dataset([0.03, 0.05], save_dir="./vnnlib_props", num_images=100, seed=42)
# generate_vnnlib_dataset([0.03], save_dir="./vnnlib_props_0.03", num_images=100, seed=42)

# python generate_properties_judy.py --epsilons 0.03 --save_dir ./vnnlib_props_0.03_mnist --num_images 20 --random True --seed 42