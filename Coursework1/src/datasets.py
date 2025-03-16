import numpy as np
from sklearn.datasets import make_blobs, make_circles
import os
import urllib.request
import tarfile
import glob
import pickle
from PIL import Image
import scipy.io
from tensorflow.keras.datasets import mnist
import bz2

def synthetic_dataset_1(cluster_density=0.75, random_state=42):
    n_samples = int(400*cluster_density)
    
    # Generate two dense clusters
    X1, _ = make_blobs(n_samples=n_samples, centers=[[0, -0.33]], cluster_std=0.1, random_state=random_state)
    X2, _ = make_blobs(n_samples=n_samples, centers=[[0, 0.33]], cluster_std=0.1, random_state=random_state)

    # Generate circular pattern
    X_circle, _ = make_circles(n_samples=n_samples*2, factor=0.99, noise=0.02, random_state=random_state)

    # Generate sparse noise around
    np.random.seed(random_state)
    X_noise = np.random.uniform(-1.5, 1.5, (n_samples*2, 2))

    # Remove points inside the circumference centered at (0,0) with radius 1.2
    distance_from_center = np.sqrt(np.sum(X_noise**2, axis=1))
    X_noise = X_noise[distance_from_center > 1.3]

    # Assign labels to each cluster
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    y_circle = np.full(X_circle.shape[0], 2)
    y_noise = np.full(X_noise.shape[0], 3)

    # Combine all points and labels
    X = np.vstack((X1, X2, X_circle, X_noise))
    Y = np.concatenate((y1, y2, y_circle, y_noise))

    return X, Y

def synthetic_dataset_2(cluster_density=0.75, random_state=42):
    cluster_density = 0.75
    n_samples = int(400*cluster_density)

    Pi = np.pi

    # Generate two dense clusters
    X1, _ = make_blobs(n_samples=n_samples, centers=[[-3/4*Pi+0.1, 0]], cluster_std=0.2, random_state=random_state)
    X2, _ = make_blobs(n_samples=n_samples, centers=[[-1/4*Pi-0.1, 0]], cluster_std=0.2, random_state=random_state)

    # Generate sinusoidal cluster
    np.random.seed(random_state)
    X_sin_raw = np.random.uniform(-Pi, Pi, n_samples*3)
    sin_noise = np.random.normal(0, 0.05, n_samples*3)
    X_sin = np.column_stack((X_sin_raw, -np.sin(X_sin_raw)*Pi/2 + sin_noise))

    # Generate line cluster
    X_line_raw = np.random.uniform(1/5*Pi, 1/3*Pi, n_samples)
    line_noise = np.random.normal(0, 0.05, n_samples)
    X_line = np.column_stack((X_line_raw, 5 * X_line_raw - Pi) + line_noise)

    # Generate square cluster
    X_square = np.random.uniform(0, 2/5*Pi, (n_samples, 2))
    X_square[:, 0] = X_square[:, 0] + 0.4*Pi
    X_square[:, 1] = X_square[:, 1] - 0.1*Pi

    # Assign labels to each cluster
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    y_sin = np.full(X_sin.shape[0], 2)
    y_line = np.full(X_line.shape[0], 3)
    y_square = np.full(X_square.shape[0], 4)

    # Combine all points and labels
    X = np.vstack((X1, X2, X_sin, X_line, X_square))
    Y = np.concatenate((y1, y2, y_sin, y_line, y_square))

    return X, Y

def synthetic_dataset_3(cluster_density=0.75, random_state=42):
    n_samples = int(400*cluster_density)
    
    # Generate two dense clusters
    X1, _ = make_blobs(n_samples=n_samples*2, centers=[[-0.5, 0]], cluster_std=0.05, random_state=random_state)
    X2, _ = make_blobs(n_samples=n_samples*2, centers=[[0.5, 0]], cluster_std=0.05, random_state=random_state)

    # Generate sparse noise around
    np.random.seed(random_state)
    X_noise = np.random.uniform(-1, 1, (n_samples, 2))

    # Remove points inside the circumferences centered at X1 with radius 0.25
    distance_from_X1 = np.sqrt(np.sum((X_noise - [-0.5, 0])**2, axis=1))
    X_noise = X_noise[distance_from_X1 > 0.25]

    # Remove points inside the circumferences centered at X1 with radius 0.25
    distance_from_X2 = np.sqrt(np.sum((X_noise - [0.5, 0])**2, axis=1))
    X_noise = X_noise[distance_from_X2 > 0.25]

    # Assign labels to each cluster
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    y_noise = np.full(X_noise.shape[0], 2)

    # Combine all points and labels
    X = np.vstack((X_noise, X1, X2))
    Y = np.concatenate((y_noise, y1, y2))

    return X, Y

def download_mnist():
    """
    Loads MNIST dataset and selects only the testing images with digits 0-4.
    Returns (X, Y_true) from the filtered test set.
    """
    (_, _), (x_test, y_test) = mnist.load_data()
    mask = y_test < 5
    X = x_test[mask].reshape(-1, 28*28)
    Y_true = y_test[mask]
    return X, Y_true

def prepare_usps(raw_data_path):
    """
    Loads the USPS dataset from the provided bz2 compressed files and returns (X, Y_true).
    """
    def load_bz2(filepath):
        """
        Loads USPS data from a bz2 file in LIBSVM format.
        Returns (X, Y_true).
        """
        X, Y_true = [], []
        with bz2.BZ2File(filepath, 'rb') as f:  # 'rb' for binary mode
            for line in f:
                line = line.decode('utf-8').strip()  # Decode each line
                if not line:
                    continue
                parts = line.split()
                # First element is the label
                Y_true.append(int(parts[0]))
                # Remaining are feature:value pairs
                features = np.zeros(256)  # USPS images are 16x16
                for item in parts[1:]:
                    index, value = item.split(':')
                    features[int(index) - 1] = float(value)  # LIBSVM indexing starts from 1
                X.append(features)
        return np.array(X), np.array(Y_true)



    train_file = os.path.join(raw_data_path, 'usps.bz2')
    test_file = os.path.join(raw_data_path, 'usps.t.bz2')
    
    X_train, Y_train = load_bz2(train_file)
    X_test, Y_test = load_bz2(test_file)
    
    # Combine train and test sets
    X = np.vstack([X_train, X_test])
    Y_true = np.hstack([Y_train, Y_test])
    
    return X, Y_true

def prepare_caltech256(raw_data_path):
    """
    Loads the Caltech-256 dataset from the tar file, extracts the specified categories,
    and selects the first 100 images from each category. Returns (X, Y_true).
    """
    tar_filename = os.path.join(raw_data_path, '256_ObjectCategories.tar')
    extract_path = os.path.join(raw_data_path, '256_ObjectCategories')
    
    if not os.path.exists(extract_path):
        print("Extracting Caltech-256 dataset...")
        with tarfile.open(tar_filename) as tar:
            tar.extractall()
    
    categories = ['hibiscus', 'ketch-101', 'leopards-101', 'motorbikes-101', 'airplanes-101', 'faces-easy-101']
    label_dict = {cat: idx for idx, cat in enumerate(categories)}
    X, Y_true = [], []

    def find_category_folder(extract_path, category):
        # Search for folders that match the pattern 'XXX.category'
        for folder in os.listdir(extract_path):
            if folder.endswith(f'.{category}'):
                return os.path.join(extract_path, folder)
        raise ValueError(f"Category '{category}' not found in Caltech-256 dataset.")
    
    for cat in categories:
        cat_folder = find_category_folder(extract_path, cat)
        img_files = sorted(glob.glob(os.path.join(cat_folder, '*')))
        for img_path in img_files[:100]:
            try:
                img = Image.open(img_path).convert('L').resize((60, 70))
                X.append(np.array(img).flatten())
                Y_true.append(label_dict[cat])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    X = np.array(X)
    Y_true = np.array(Y_true)
    return X, Y_true


def prepare_datasets(data_path, raw_data_path):
    """
    Downloads and processes USPS, MNIST, and Caltech-256 datasets.
    Saves each dataset as a pickle file in the specified data_path.
    """
    datasets = {
        'MNIST': download_mnist(),
        'USPS': prepare_usps(raw_data_path),
        'Caltech-256': prepare_caltech256(raw_data_path)
    }
    os.makedirs(data_path, exist_ok=True)
    for name, data in datasets.items():
        with open(os.path.join(data_path, f'{name}.pkl'), 'wb') as f:
            pickle.dump(data, f)

def load_dataset(data_path, dataset_name):
    """
    Loads a specified dataset from a pickle file.
    Returns (X, Y_true) for the requested dataset.
    """
    filepath = os.path.join(data_path, f'{dataset_name}.pkl')
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found at {data_path}.")
    with open(filepath, 'rb') as f:
        return pickle.load(f)
