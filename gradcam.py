import tensorflow as tf
import keras
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, target_class="presence", apply_relu=False):
    """
    Generates a Grad-CAM heatmap for a binary classification model.
    
    Args:
        target_class (str): "presence" (default) or "absence".
    """
    grad_model = keras.models.Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    if target_class.lower() == "absence":
        heatmap = -heatmap
    
    # Optional ReLU
    if apply_relu:
        heatmap = tf.maximum(heatmap, 0) 
    else:
        # If we don't ReLU, we shift the heatmap so the lowest negative value becomes 0 
        # just so the colormap can display it properly from 0 to 1.
        heatmap = heatmap - tf.math.reduce_min(heatmap)
    
    # Normalize
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        print("WARNING: Heatmap is completely zero after ReLU!")
        
    heatmap = heatmap / (max_val + 1e-10)
    
    return heatmap.numpy()



def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET, threshold=0):
    """
    Overlays a continuous heatmap on a 1-channel or 3-channel image.
    
    Args:
        img: NumPy array of the background image (H, W), (H, W, 1), or (H, W, 3).
        heatmap: NumPy array of the heatmap (H, W) with values between 0 and 1.
        alpha: Blending factor. 0.0 = only image, 1.0 = only heatmap.
        colormap: OpenCV colormap to apply to the heatmap.
        threshold: Percentage (0-100). E.g., 80 means only the top 20% of 
                   activated pixels will be overlaid. 0 shows the full heatmap.
        
    Returns:
        NumPy array of the overlaid image (H, W, 3) in uint8 format.
    """
    # 1. Format the Background Image
    img = np.array(img)

    if len(img.shape) == 2:
        rgb = np.stack([img, img, img], axis=-1)
    elif len(img.shape) == 3:
        num_bands = img.shape[-1]
        if num_bands == 1:
            gray = img[:, :, 0]
            rgb = np.stack([gray, gray, gray], axis=-1)
        elif num_bands == 3:
            rgb = img.copy()
        else:
            raise ValueError("Unsupported number of bands. Expected 1 or 3.")
    else:
        raise ValueError("Unsupported image shape.")

    # Normalize the RGB image to 0-255 uint8
    rgb = np.clip(rgb, 0, 255)
    if rgb.max() > rgb.min():  
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
    rgb = rgb.astype(np.uint8)

    # 2. Format the Heatmap
    heatmap = np.array(heatmap)
    
    # Resize heatmap to match image dimensions if necessary
    if heatmap.shape != rgb.shape[:2]:
        heatmap = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]))
        
    # Ensure heatmap is normalized between 0 and 1
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # 3. Calculate Threshold Mask
    # Determine the cutoff value based on the requested percentile
    cutoff_value = np.percentile(heatmap, threshold)
    
    # Create a boolean mask of pixels that are above the cutoff
    mask = heatmap >= cutoff_value

    # 4. Apply Colormap and Blend
    # Convert heatmap to 8-bit (0-255) to apply colormap
    heatmap_8bit = np.uint8(255 * heatmap)
    
    # Apply colormap (OpenCV outputs BGR, so we convert it to RGB)
    heatmap_color = cv2.applyColorMap(heatmap_8bit, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # cv2.addWeighted calculates the blended image
    blended = cv2.addWeighted(rgb, 1 - alpha, heatmap_color, alpha, 0)
    
    # 5. Apply the Mask
    # Where the mask is True, use the blended image. Where False, use the original RGB image.
    # mask[..., None] expands the mask from (H, W) to (H, W, 1) so it broadcasts across RGB channels
    overlay = np.where(mask[..., None], blended, rgb)
    
    return overlay


def min_max_norm(a):
    min_ = a.min()
    max_ = a.max()
    a_norm = (a - min_)/(max_ - min_ + 1e-10)

    return a_norm

def calculate_spatial_binning_agreement(heatmaps, grid_blocks=25, target_shape=None, normalize=True):
    """
    Calculates spatial agreement between heatmaps by dividing them into a grid of blocks
    and correlating the mean activations.
    
    Args:
        heatmaps (list of np.array): List of heatmaps.
        grid_blocks (int or tuple): Number of total blocks (e.g., 25 for a 5x5 grid), 
                                    or a specific tuple like (5, 5).
        target_shape (tuple): (height, width) to resize all heatmaps to a common grid.
        
    Returns:
        dict: Contains the mean correlation, full correlation matrix, and the binned values.
    """
    if target_shape is None:
        target_shape = heatmaps[0].shape[:2]
        
    # Determine rows and columns for the grid
    if isinstance(grid_blocks, int):
        grid_dim = int(np.sqrt(grid_blocks))
        rows, cols = grid_dim, grid_dim
    else:
        rows, cols = grid_blocks
        
    num_blocks = rows * cols
    binned_heatmaps = []
    
    for hm in heatmaps:
        # Ensure 2D
        if hm.ndim == 3:
            hm = hm[:, :, 0]
        elif hm.ndim > 3:
            hm = np.squeeze(hm)

        hm_norm = min_max_norm(hm)
            
        # Resize to common spatial grid
        if hm.shape != target_shape:
            hm_resized = cv2.resize(hm_norm.astype(float), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            hm_resized = hm_norm.copy().astype(float)
            
        # 1. Binning: Calculate mean activation in each block
        h, w = target_shape
        block_h = h / rows
        block_w = w / cols
        
        binned = np.zeros(num_blocks)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                # Pixel coordinates for the block
                r_start, r_end = int(r * block_h), int((r + 1) * block_h)
                c_start, c_end = int(c * block_w), int((c + 1) * block_w)
                
                # Extract block and calculate mean
                block = hm_resized[r_start:r_end, c_start:c_end]
                binned[idx] = np.mean(block)
                idx += 1
                
        # Add a tiny epsilon to avoid division by zero in correlation if heatmap is completely blank
        if np.std(binned) == 0:
            binned += 1e-10 
            
        binned_heatmaps.append(binned)
        
    # Shape: (num_models, num_blocks)
    binned_matrix = np.array(binned_heatmaps) 
    
    # 2. Calculate Correlation
    # np.corrcoef compares each row (model) with every other row
    corr_matrix = np.corrcoef(binned_matrix)
    
    # Extract only the upper triangle (pairwise comparisons, ignoring self-comparisons)
    upper_tri_indices = np.triu_indices(corr_matrix.shape[0], k=1)
    pairwise_correlations = corr_matrix[upper_tri_indices]
    
    # The final single metric
    mean_correlation = np.mean(pairwise_correlations)
    
    return {
        "mean_pairwise_correlation": mean_correlation,
        "pairwise_correlations": pairwise_correlations,
        "correlation_matrix": corr_matrix,
        "binned_arrays": binned_matrix
    }