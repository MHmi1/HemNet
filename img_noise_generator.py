#!/usr/bin/env python3

#HemNet denoising framework - 2025 

"""
noisy_img_gen.py: A module to add various types of noise to images for data augmentation in denoising tasks.

This script supports multiple noise types including Gaussian, Salt & Pepper, Periodic, Speckle, and Poisson.
It processes all images in the input directory and saves noisy versions to the output directory.
It also supports preserving hierarchical folder structures if in_path_analyzing is enabled.
Multiple noise types can be applied by specifying a comma-separated list.

Usage:
    python noisy_img_gen.py --input_path ./images --output_path ./noisy_images --noise_type gaussian,salt_pepper --mean 0.0 --var 0.01 --in_path_analyzing True
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple


class NoiseGenerator:
    """
    A class to generate various types of noise and apply them to images.
    """
    def __init__(self, noise_type: str, **kwargs):
        """
        Initialize the NoiseGenerator with the specified noise type and parameters.

        Args:
            noise_type (str): Type of noise to apply. Options: gaussian, salt_pepper, periodic, speckle, poisson.
            **kwargs: Additional parameters for specific noise types (mean, var, amount, etc.).
        """
        self.noise_type = noise_type.lower()
        self.params = kwargs

    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to the image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Noisy image.
        """
        mean = self.params.get('mean', 0.0)
        var = self.params.get('var', 0.01)
        sigma = var ** 0.5
        if len(image.shape) == 2:  # Grayscale image
            gauss = np.random.normal(mean, sigma, image.shape)
            noisy = image + gauss
        else:  # RGB image
            gauss = np.random.normal(mean, sigma, image.shape)
            noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_salt_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Salt & Pepper noise to the image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Noisy image.
        """
        amount = self.params.get('amount', 0.05)
        noisy = image.copy()
        num_salt = np.ceil(amount * image.size * 0.5)
        num_pepper = np.ceil(amount * image.size * 0.5)
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        if len(image.shape) == 3:
            noisy[coords[0], coords[1], :] = 255
        else:
            noisy[coords[0], coords[1]] = 255
        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        if len(image.shape) == 3:
            noisy[coords[0], coords[1], :] = 0
        else:
            noisy[coords[0], coords[1]] = 0
        return noisy

    def add_periodic_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add periodic noise to the image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Noisy image.
        """
        amplitude = self.params.get('amplitude', 20)
        frequency = self.params.get('frequency', 0.1)
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        noise = amplitude * np.sin(2 * np.pi * frequency * x + 2 * np.pi * frequency * y)
        if len(image.shape) == 3:
            noisy = image + noise[:, :, np.newaxis]
        else:
            noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add speckle noise to the image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Noisy image.
        """
        var = self.params.get('var', 0.05)
        noise = np.random.normal(0, var ** 0.5, image.shape[:2])
        if len(image.shape) == 3:
            noise = np.stack([noise] * 3, axis=-1)
        noisy = image + image * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Poisson noise to the image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Noisy image.
        """
        lam = self.params.get('lam', 1.0)
        noisy = np.random.poisson(image * lam) / lam
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the specified noise type to the image.

        Args:
            image (np.ndarray): Input image as a numpy array.

        Returns:
            np.ndarray: Noisy image.
        """
        noise_functions = {
            'gaussian': self.add_gaussian_noise,
            'salt_pepper': self.add_salt_pepper_noise,
            'periodic': self.add_periodic_noise,
            'speckle': self.add_speckle_noise,
            'poisson': self.add_poisson_noise
        }
        if self.noise_type not in noise_functions:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return noise_functions[self.noise_type](image)


def analyze_folder_structure(input_path: str) -> List[Tuple[Path, Path]]:
    """
    Analyze the hierarchical structure of the input directory and return a list of input-output path pairs
    to preserve the structure in the output directory.

    Args:
        input_path (str): Path to the input directory.

    Returns:
        List[Tuple[Path, Path]]: List of tuples containing (input_file_path, output_file_path) for each image.
    """
    input_dir = Path(input_path)
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = []
    for ext in supported_extensions:
        image_files.extend(input_dir.rglob(f'*{ext}'))
        image_files.extend(input_dir.rglob(f'*{ext.upper()}'))

    if not image_files:
        print(f"No images found in {input_path} or its subdirectories")
        return []

    path_pairs = []
    for img_path in image_files:
        # Compute the relative path to preserve hierarchy
        relative_path = img_path.relative_to(input_dir)
        output_filename = relative_path.parent / f"noisy_{relative_path.name}"
        path_pairs.append((img_path, output_filename))
    return path_pairs


def process_images(input_path: str, output_path: str, noise_types: List[str], noise_params: dict, in_path_analyzing: bool = False) -> None:
    """
    Process all images in the input directory, apply each specified noise type, and save to the output directory.
    If in_path_analyzing is True, preserve the hierarchical structure of the input directory.
    Each noise type will have its own subdirectory in the output path.

    Args:
        input_path (str): Path to the directory containing input images.
        output_path (str): Path to save noisy images.
        noise_types (List[str]): List of noise types to apply.
        noise_params (dict): Dictionary of noise parameters.
        in_path_analyzing (bool): Flag to enable hierarchical structure preservation.
    """
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    if in_path_analyzing:
        # Analyze folder structure and get input-output path pairs
        path_pairs = analyze_folder_structure(input_path)
        if not path_pairs:
            return

        for noise_type in noise_types:
            # Create a noise-specific subdirectory in output
            noise_output_dir = output_dir / noise_type
            noise_output_dir.mkdir(parents=True, exist_ok=True)
            # Initialize noise generator for this type
            noise_gen = NoiseGenerator(noise_type=noise_type, **noise_params)
            print(f"Processing images with {noise_type} noise...")

            for img_path, rel_output_path in path_pairs:
                # Read image
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                # Apply noise
                noisy_image = noise_gen.apply_noise(image)

                # Create output directory structure
                output_filename = noise_output_dir / rel_output_path
                output_filename.parent.mkdir(parents=True, exist_ok=True)

                # Save noisy image
                cv2.imwrite(str(output_filename), noisy_image)
                print(f"Saved noisy image: {output_filename}")
    else:
        # Process images in a flat structure (only top-level directory)
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in input_dir.glob('*') if f.suffix.lower() in supported_extensions]

        if not image_files:
            print(f"No images found in {input_path}")
            return

        for noise_type in noise_types:
            # Create a noise-specific subdirectory in output
            noise_output_dir = output_dir / noise_type
            noise_output_dir.mkdir(parents=True, exist_ok=True)
            # Initialize noise generator for this type
            noise_gen = NoiseGenerator(noise_type=noise_type, **noise_params)
            print(f"Processing images with {noise_type} noise...")

            for img_path in image_files:
                # Read image
                image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"Failed to load image: {img_path}")
                    continue

                # Apply noise
                noisy_image = noise_gen.apply_noise(image)

                # Save noisy image
                output_filename = noise_output_dir / f"noisy_{img_path.name}"
                cv2.imwrite(str(output_filename), noisy_image)
                print(f"Saved noisy image: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate noisy images for denoising tasks.")
    parser.add_argument('--input_path', type=str, required=True,
                        help="Path to the directory containing input images.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to save noisy images.")
    parser.add_argument('--noise_type', type=str, default='gaussian',
                        help="Comma-separated list of noise types to apply (e.g., gaussian,salt_pepper). Options: gaussian, salt_pepper, periodic, speckle, poisson.")
    parser.add_argument('--in_path_analyzing', type=bool, default=False,
                        help="Flag to enable hierarchical input path structure analysis and preservation (default: False).")
    # Gaussian noise parameters
    parser.add_argument('--mean', type=float, default=0.0,
                        help="Mean of Gaussian noise (default: 0.0).")
    parser.add_argument('--var', type=float, default=0.01,
                        help="Variance of Gaussian noise (default: 0.01).")
    # Salt & Pepper noise parameters
    parser.add_argument('--amount', type=float, default=0.05,
                        help="Amount of Salt & Pepper noise (default: 0.05).")
    # Periodic noise parameters
    parser.add_argument('--amplitude', type=float, default=20.0,
                        help="Amplitude of periodic noise (default: 20.0).")
    parser.add_argument('--frequency', type=float, default=0.1,
                        help="Frequency of periodic noise (default: 0.1).")
    # Poisson noise parameters
    parser.add_argument('--lam', type=float, default=1.0,
                        help="Lambda parameter for Poisson noise (default: 1.0).")

    args = parser.parse_args()

    # Parse noise types from comma-separated input
    noise_types = [nt.strip() for nt in args.noise_type.split(',')]
    valid_noise_types = {'gaussian', 'salt_pepper', 'periodic', 'speckle', 'poisson'}
    for nt in noise_types:
        if nt not in valid_noise_types:
            raise ValueError(f"Invalid noise type: {nt}. Must be one of {valid_noise_types}")

    # Prepare noise parameters
    noise_params = {
        'mean': args.mean,
        'var': args.var,
        'amount': args.amount,
        'amplitude': args.amplitude,
        'frequency': args.frequency,
        'lam': args.lam
    }

    # Process images for each noise type
    print(f"Applying noise types {noise_types} to images in {args.input_path}...")
    if args.in_path_analyzing:
        print("Hierarchical structure analysis enabled. Preserving folder structure in output.")
    process_images(args.input_path, args.output_path, noise_types, noise_params, args.in_path_analyzing)
    print("Done!")


if __name__ == "__main__":
    main()