import pandas as pd
import numpy as np
import cv2

from osgeo import gdal
import rasterio

import tensorflow as tf
import keras
from keras.utils import PyDataset
from sklearn.model_selection import train_test_split

import os
import random
import warnings

warnings.filterwarnings("ignore")

from config import *


def get_patch_list(base_dir, patch_type, sites=None, distances=None):
    """
    Get a list of all patch files in the specified patch type directory, optionally filtering by site and distance.

    Args:
        base_dir (str): The root directory (train_patches, test_patches, or val_patches).
        patch_type (str): The type of patch directory to look for (e.g., 'dsm_patches', 'dtm_patches').
        sites (list, optional): A list of site names to filter by (e.g., ['fao', 'roudoudour', 'timbertiere']).
                                If None, no site filtering is applied.
        distances (list, optional): A list of distances to filter by (e.g., [40, 70, 100] or ['40m', '70m']).
                                    If None, no distance filtering is applied.

    Returns:
        list: A list of paths to the patch files.
    """
    # Format distances to ensure they end with 'm' (e.g., 40 -> '40m')
    formatted_distances = None
    if distances:
        formatted_distances = [f"{d}m" if not str(d).endswith('m') else str(d) for d in distances]

    patch_list = []
    for root, dirs, files in os.walk(base_dir):
        # Check if the directory matches the patch type
        if os.path.basename(root) == patch_type:
            
            # 1. Site Filter
            valid_site = True
            if sites:
                valid_site = any(site in root for site in sites)
                
            # 2. Distance Filter
            valid_distance = True
            if formatted_distances:
                # Check if the distance folder name is present in the path
                path_parts = root.split(os.sep)
                valid_distance = any(dist in path_parts for dist in formatted_distances)
            
            # If both conditions are met, add the files
            if valid_site and valid_distance:
                patch_list.extend([os.path.join(root, file) for file in files if file.endswith(".tif")])
                
    return patch_list


def get_labels(df_labels, img_paths, order): 
    """
    Retrieves labels from a DataFrame based on specified image paths, order, site, and campaign.
    
    Args:
        df_labels (pd.DataFrame): DataFrame containing columns 'id', 'site', 'campaign', and label columns for each order.
        img_paths (list): List of image file paths used to extract IDs.
        order (str): Specifies the insect order for label retrieval (options: 'ephemeroptera', 'plecoptera', 'trichoptera').
        
    Returns:
        list: List of labels corresponding to the specified order.

    Raises:
        ValueError: If `order` is not among the allowed options.
    """

    valid_orders = ['plecoptera', 'trichoptera']
    if order not in valid_orders:
        raise ValueError(f"Invalid order '{order}'. Must be one of {valid_orders}.")

    ids_order = [int(os.path.basename(i).split('_')[-1].split('.')[0]) for i in img_paths]
    sites_order = [str(os.path.basename(i).split('_')[1]) for i in img_paths]
    campaigns_order = [str(os.path.basename(i).split('_')[0]) for i  in img_paths]
    
    order_df = pd.DataFrame({
        'id': ids_order,
        'site': sites_order,
        'campaign': campaigns_order
    })
    
    df_ordered = pd.merge(order_df, df_labels, on=['id', 'site', 'campaign'], how='left')
    df_ordered.drop_duplicates(inplace=True)
    labels = df_ordered[order].values.tolist()
    return labels


def patch_minmaxnormalisation(input_image, band_by_band=True):
    """
    Applies min-max normalization to an image patch, either band-by-band or globally.

    Args:
        input_image (np.array): The image array to normalize.
        band_by_band (bool): If True, normalize each channel separately.

    Returns:
        np.array: The normalized image.
    """
    image = input_image.copy()
    if band_by_band:
        if len(image.shape) > 2:
            for b in range(image.shape[-1]):
                band = image[:,:,b]
                max_ = tf.reduce_max(band)
                min_ = tf.reduce_min(band)
                if max_ - min_ > 0:
                    band = (band - min_) / (max_ - min_)  
                image[:,:,b] = band
        else:
            max_ = tf.reduce_max(image)
            min_ = tf.reduce_min(image)
            if max_ - min_ > 0:
                image = (image - min_) / (max_ - min_)
                image = image.numpy()
    else:
        max_ = tf.reduce_max(image)
        min_ = tf.reduce_min(image)
        if max_ - min_ > 0:
            image = (image - min_) / (max_ - min_)
            image = image.numpy()
            
    return image


def random_rotate(image, angle):
    """
    Apply a random rotation around the image center.

    Args:
        image (tf.Tensor): Input image tensor.

    Returns:
        tf.Tensor: Rotated image.
    """
    radians = angle * np.pi / 180.0 
    height = tf.cast(tf.shape(image)[0], tf.float32)
    width = tf.cast(tf.shape(image)[1], tf.float32)
    cx, cy = width / 2.0, height / 2.0

    def get_rotation_matrix(angle, cx, cy):
        cos_a = tf.math.cos(angle)
        sin_a = tf.math.sin(angle)
        return tf.reshape(
            [cos_a, -sin_a, cx - cos_a * cx + sin_a * cy,
             sin_a,  cos_a, cy - sin_a * cx - cos_a * cy,
             0.0,    0.0],
            [1, 8]
        )

    transform_matrix = get_rotation_matrix(radians, cx, cy)
    image = tf.expand_dims(image, axis=0)  

    image_rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform_matrix,
        output_shape=tf.shape(image)[1:3],
        interpolation="NEAREST",
        fill_value=0.0
    )

    image_rotated = tf.squeeze(image_rotated, axis=0)
    return image_rotated




class MultimodalDataGenerator(PyDataset):
    """
    Data generator supporting both Single Modality and Two-Branch Late Fusion.
    """
    def __init__(self, 
                 img_paths, 
                 batch_size, 
                 labels=None, 
                 band_by_band_normalisation=True, 
                 augment=False, 
                 dominant_class="present", 
                 use_mixup=False,
                 ms_size=None,
                 dem_size=None,
                 single_modality=False,
                 shuffle=True, 
                 workers=1,
                 use_multiprocessing=False):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing)
        
        self.img_paths = img_paths
        self.labels = labels
        self.band_by_band_normalisation = band_by_band_normalisation
        self.batch_size = batch_size
        self.augment = augment
        self.use_mixup = use_mixup
        self.dominant_class = dominant_class
        self.ms_size = ms_size
        self.dem_size = dem_size
        self.single_modality = single_modality
        self.indices = np.arange(len(self.img_paths))
        self.shuffle = shuffle
        
        # We need a default fallback size to avoid undefined variable errors
        self.default_resize = {"MS": 128, "DEM": 128} 
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):        
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.img_paths))
        batch_indices = self.indices[start_idx:end_idx]
        
        if self.single_modality:
            batch_features, batch_labels = self.__data_generation_single(batch_indices)
            
            if self.use_mixup:
                batch_features, batch_labels = self.__apply_mixup_single(
                    batch_features, batch_labels, dominant_class=self.dominant_class
                )
            if self.augment:
                batch_features = self.__classic_augment_batch_single(batch_features)
                
            return np.array(batch_features), np.array(batch_labels)
                
        else:
            (batch_spectral, batch_topo), batch_labels = self.__data_generation_multi(batch_indices)

            if self.use_mixup:
                batch_spectral, batch_topo, batch_labels = self.__apply_mixup_multi(
                    batch_spectral, batch_topo, batch_labels, dominant_class=self.dominant_class
                )
            if self.augment:
                batch_spectral, batch_topo = self.__classic_augment_batch_multi(batch_spectral, batch_topo)

            return (np.array(batch_spectral), np.array(batch_topo)), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation_single(self, batch_indices):
        batch_features = []
        batch_labels = []
        
        for idx in batch_indices:
            paths = self.img_paths[idx]
            
            if len(paths) == 1:
                # Drone MS or Super-Resolution (MS) 
                feat = self.load_and_preprocess_image(paths[0], spectral=True, default_resize=self.default_resize["MS"])
                
            elif len(paths) == 2:
                # Check if it's S2 (MS+RE) or DEM (DSM+DTM)
                is_dem = any(x in paths[0].lower() for x in ["dsm", "dtm"])
                
                if is_dem:
                    dsm = self.load_and_preprocess_image(paths[0], spectral=False, default_resize=self.default_resize["DEM"])
                    dtm = self.load_and_preprocess_image(paths[1], spectral=False, default_resize=self.default_resize["DEM"])
                    feat = np.concatenate([dsm, dtm], axis=-1)
                else:
                    s2_ms = self.load_and_preprocess_image(paths[0], spectral=True, default_resize=self.default_resize["MS"])
                    s2_re = self.load_and_preprocess_image(paths[1], spectral=True, default_resize=self.default_resize["MS"])
                    feat = np.concatenate([s2_ms, s2_re], axis=-1)
            else:
                raise ValueError(f"For single modality, expected 1 or 2 paths, got {len(paths)}")
                
            label = 1.0 if self.labels[idx] > 0 else 0.0
            
            batch_features.append(feat)
            batch_labels.append(label)
            
        return np.array(batch_features), np.array(batch_labels)

    def __data_generation_multi(self, batch_indices):
        batch_spectral = []
        batch_topo = []
        batch_labels = []
        
        for idx in batch_indices:
            paths = self.img_paths[idx]
            if len(paths) == 3:
                spectral = self.load_and_preprocess_image(paths[0], spectral=True, default_resize=self.default_resize["MS"])
                dsm = self.load_and_preprocess_image(paths[1], spectral=False, default_resize=self.default_resize["DEM"])
                dtm = self.load_and_preprocess_image(paths[2], spectral=False, default_resize=self.default_resize["DEM"])
                topo = np.concatenate([dsm, dtm], axis=-1)
                
            elif len(paths) == 4:
                s2_ms = self.load_and_preprocess_image(paths[0], spectral=True, default_resize=self.default_resize["MS"])
                s2_re = self.load_and_preprocess_image(paths[1], spectral=True, default_resize=self.default_resize["MS"])
                spectral = np.concatenate([s2_ms, s2_re], axis=-1)

                dsm = self.load_and_preprocess_image(paths[2], spectral=False, default_resize=self.default_resize["DEM"])
                dtm = self.load_and_preprocess_image(paths[3], spectral=False, default_resize=self.default_resize["DEM"])
                topo = np.concatenate([dsm, dtm], axis=-1)
            else:
                raise ValueError(f"Expected 3 or 4 modalities, got {len(paths)}")
            
            label = 1.0 if self.labels[idx] > 0 else 0.0
            
            batch_spectral.append(spectral)
            batch_topo.append(topo)
            batch_labels.append(label)
            
        return (np.array(batch_spectral), np.array(batch_topo)), np.array(batch_labels)

    def ensure_string_path(self, path):
        if isinstance(path, bytes):
            return path.decode('utf-8')
        return path

    def remove_nan(self, image, nan=-32767.0):
        image = np.nan_to_num(image, nan=nan)
        return image

    def load_and_preprocess_image(self, image_path, spectral, default_resize, expand_dims=False):
        image_path = self.ensure_string_path(image_path)  
        basename = os.path.basename(image_path)
        split = image_path.split('\\')
        
        try:
            type_ = split[2]
            site = split[-1].split('_')[1]
            dist = split[1]
        except IndexError:
            type_, site, dist = "unknown", "unknown", "unknown"
            
        dataset = gdal.Open(str(image_path))
        if dataset is None:
            raise FileNotFoundError(f"Failed to open image file: {image_path}")

        bands = dataset.RasterCount
        image = np.stack([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=-1)

        if spectral and (dist == "40m"):
            expected_possible_bands = expected_bands_dict[type_]
            if (bands not in expected_possible_bands):
                if type_ == "S2_MS_patches": bands = 4
                elif type_ == "S2_RE_patches": bands = 1
                else: bands = 5 if site == "fao" else 10
                
                total_elements = image.size
                true_height = image.shape[1] 
                true_width = total_elements // (bands * true_height)
                image = image.flatten().reshape((bands, true_height, true_width))
                image = np.transpose(image, (1, 2, 0))
                
        
        image = image.astype(np.float32)
        
        if spectral:
            target_size = self.ms_size if self.ms_size else default_resize
        else:
            target_size = self.dem_size if self.dem_size else default_resize
            
        if image.shape[0] != target_size or image.shape[1] != target_size:
            image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        image = self.remove_nan(image)
        image = image.clip(min=0)
        image = patch_minmaxnormalisation(image, self.band_by_band_normalisation)
   
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            
        return image

    def __classic_augment_batch_single(self, batch_features):
        batch_features = list(batch_features)
        for i in range(len(batch_features)):
            if random.random() > 0.4: batch_features[i] = tf.image.flip_left_right(batch_features[i])
            if random.random() > 0.4: batch_features[i] = tf.image.flip_up_down(batch_features[i])
            if random.random() > 0.4:
                angle = tf.random.uniform([], minval=-20, maxval=20, dtype=tf.float32)
                batch_features[i] = random_rotate(batch_features[i], angle)
        return batch_features

    def __classic_augment_batch_multi(self, batch_spectral, batch_topo):
        batch_spectral = list(batch_spectral)
        batch_topo = list(batch_topo)
        for i in range(len(batch_spectral)):
            flip_lr = random.random() > 0.4
            flip_ud = random.random() > 0.4
            rotate = random.random() > 0.4
            
            if flip_lr:
                batch_spectral[i] = tf.image.flip_left_right(batch_spectral[i])
                batch_topo[i] = tf.image.flip_left_right(batch_topo[i])
            if flip_ud:
                batch_spectral[i] = tf.image.flip_up_down(batch_spectral[i])
                batch_topo[i] = tf.image.flip_up_down(batch_topo[i])
            if rotate:
                angle = tf.random.uniform([], minval=-20, maxval=20, dtype=tf.float32)
                batch_spectral[i] = random_rotate(batch_spectral[i], angle)
                batch_topo[i] = random_rotate(batch_topo[i], angle)
        return batch_spectral, batch_topo

    def __apply_mixup_single(self, batch_features, batch_labels, alpha=0.25, beta=0.25, dominant_class="present"):
        batch_size = batch_features.shape[0]
        absent_indices = np.where(batch_labels == 0)[0]
        present_indices = np.where(batch_labels == 1)[0]
        
        target_indices = absent_indices if dominant_class == "present" else present_indices
        mixed_indices = np.random.choice(target_indices, size=batch_size, replace=True) if len(target_indices) > 0 else np.random.permutation(batch_size)

        features_shuffled = batch_features[mixed_indices]
        labels_shuffled = batch_labels[mixed_indices]
        
        l = self.sample_beta_distribution(batch_size, alpha, beta)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size,))
        
        mixed_features = batch_features * x_l + features_shuffled * (1 - x_l)
        mixed_labels = batch_labels * y_l + labels_shuffled * (1 - y_l)
        
        return np.array(mixed_features), np.array(mixed_labels)

    def __apply_mixup_multi(self, batch_spectral, batch_topo, batch_labels, alpha=0.25, beta=0.25, dominant_class="present"):
        batch_size = batch_spectral.shape[0]
        absent_indices = np.where(batch_labels == 0)[0]
        present_indices = np.where(batch_labels == 1)[0]
        
        target_indices = absent_indices if dominant_class == "present" else present_indices
        mixed_indices = np.random.choice(target_indices, size=batch_size, replace=True) if len(target_indices) > 0 else np.random.permutation(batch_size)

        spectral_shuffled = batch_spectral[mixed_indices]
        topo_shuffled = batch_topo[mixed_indices]
        labels_shuffled = batch_labels[mixed_indices]
        
        l = self.sample_beta_distribution(batch_size, alpha, beta)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size,))
        
        mixed_spectral = batch_spectral * x_l + spectral_shuffled * (1 - x_l)
        mixed_topo = batch_topo * x_l + topo_shuffled * (1 - x_l)
        mixed_labels = batch_labels * y_l + labels_shuffled * (1 - y_l)
        
        return np.array(mixed_spectral), np.array(mixed_topo), np.array(mixed_labels)

    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2, min_=0.1, max_=0.9):
        gamma_1_sample = tf.random.gamma([size], concentration_1)
        gamma_2_sample = tf.random.gamma([size], concentration_0)
        beta = gamma_1_sample / (gamma_1_sample + gamma_2_sample)
        return tf.clip_by_value(beta, min_, max_)


class CombinedDataGenerator(PyDataset):
    """
    A Keras PyDataset generator that combines data from two generators.
    Automatically handles single and multi-modality combinations.
    """
    def __init__(self, gen1, gen2, single_modality=False, shuffle=True, workers=1, use_multiprocessing=False):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing)
        self.gen1 = gen1
        self.gen2 = gen2
        self.single_modality = single_modality
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return min(len(self.gen1), len(self.gen2))

    def __getitem__(self, index):
        if self.single_modality:
            feat1, labels1 = self.gen1[index]
            feat2, labels2 = self.gen2[index]
            
            feat_combined = np.concatenate([feat1, feat2], axis=0)
            labels_combined = np.concatenate([labels1, labels2], axis=0)
            
            if self.shuffle:
                indices = np.arange(feat_combined.shape[0])
                np.random.shuffle(indices)
                feat_combined = feat_combined[indices]
                labels_combined = labels_combined[indices]
                
            return feat_combined, labels_combined
        else:
            (spec1, topo1), labels1 = self.gen1[index]
            (spec2, topo2), labels2 = self.gen2[index]
            
            spec_combined = np.concatenate([spec1, spec2], axis=0)
            topo_combined = np.concatenate([topo1, topo2], axis=0)
            labels_combined = np.concatenate([labels1, labels2], axis=0)
            
            if self.shuffle:
                indices = np.arange(spec_combined.shape[0])
                np.random.shuffle(indices)
                spec_combined = spec_combined[indices]
                topo_combined = topo_combined[indices]
                labels_combined = labels_combined[indices]
                
            return (spec_combined, topo_combined), labels_combined

    def on_epoch_end(self):
        if hasattr(self.gen1, 'on_epoch_end'):
            self.gen1.on_epoch_end()
        if hasattr(self.gen2, 'on_epoch_end'):
            self.gen2.on_epoch_end()


def get_combined_generator(img_paths, labels, 
                           batch_size, 
                           band_by_band_normalisation=True,
                           augment_first_gen=False, 
                           augment_second_gen=True, 
                           mixup_first_gen=False, 
                           mixup_second_gen=False, 
                           dominant_class="absent", 
                           ms_size=None, 
                           dem_size=None,
                           single_modality=False, 
                           workers=1,
                           use_multiprocessing=False):

    train_gen_non_augmented = MultimodalDataGenerator(
        img_paths=img_paths, 
        batch_size=batch_size // 2, 
        labels=labels, 
        augment=augment_first_gen, 
        use_mixup=mixup_first_gen,
        dominant_class=dominant_class,
        band_by_band_normalisation=band_by_band_normalisation,
        ms_size=ms_size,
        dem_size=dem_size,
        single_modality=single_modality,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

    augmented_shuffling_indices_A = random.sample(range(len(img_paths)), len(img_paths))
    image_paths_B = [img_paths[i] for i in augmented_shuffling_indices_A]
    labels_B = [labels[i] for i in augmented_shuffling_indices_A]

    train_gen_augmented = MultimodalDataGenerator(
        img_paths=image_paths_B, 
        batch_size=batch_size // 2, 
        labels=labels_B, 
        augment=augment_second_gen, 
        use_mixup=mixup_second_gen, 
        dominant_class=dominant_class,
        band_by_band_normalisation=band_by_band_normalisation,
        ms_size=ms_size,
        dem_size=dem_size,
        single_modality=single_modality,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

    train_gen_combined = CombinedDataGenerator(
        train_gen_augmented, 
        train_gen_non_augmented,
        single_modality=single_modality,
        workers=workers,
        use_multiprocessing=use_multiprocessing
    )

    return train_gen_combined
