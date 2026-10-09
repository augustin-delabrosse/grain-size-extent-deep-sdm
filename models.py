import tensorflow as tf
from tensorflow.keras import layers, Model

def cerberuscnn_extractor(input_shape, name="cerberuscnn"):
    inputs = layers.Input(shape=input_shape, name=f"{name}_input")

    # First scale: Small receptive field
    x1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Second scale: Medium receptive field
    x2 = layers.Conv2D(32, (5, 5), activation="relu", padding="same")(inputs)
    x2 = layers.MaxPooling2D((4, 4))(x2)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Conv2D(64, (5, 5), activation="relu", padding="same")(x2)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Third scale: Large receptive field
    x3 = layers.Conv2D(32, (7, 7), activation="relu", padding="same")(inputs)
    x3 = layers.MaxPooling2D((8, 8))(x3)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Conv2D(64, (7, 7), activation="relu", padding="same")(x3)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x3 = layers.GlobalAveragePooling2D()(x3)

    # Concatenate multi-scale features (Output size: 192)
    merged = layers.Concatenate()([x1, x2, x3])
    
    return Model(inputs, merged, name=name)

def tinycerberuscnn_extractor(input_shape, name="tinycerberuscnn"):
    inputs = layers.Input(shape=input_shape, name=f"{name}_input")

    # Scale 1: Pixel-level spectral signatures (1x1 Conv)
    s1 = layers.Conv2D(64, (1, 1), activation="relu", padding="same")(inputs)
    s1 = layers.BatchNormalization()(s1)
    s1 = layers.GlobalAveragePooling2D()(s1)

    # Scale 2: Local neighborhood (3x3 Conv)
    s2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    s2 = layers.BatchNormalization()(s2)
    s2 = layers.GlobalAveragePooling2D()(s2)

    # Scale 3: Global Patch Context 
    # Directly pool the input and pass through a Dense layer
    s3 = layers.GlobalAveragePooling2D()(inputs)
    s3 = layers.Dense(64, activation="relu")(s3)
    s3 = layers.BatchNormalization()(s3)

    # Concatenate multi-scale features (Output size: 192)
    merged = layers.Concatenate()([s1, s2, s3])
    
    return Model(inputs, merged, name=name)

def build_model(single_sensor=True, ms_shape=None, lidar_shape=None, num_classes=1):

    if single_sensor:
        # Feature extractor
        if ((ms_shape is not None) and (lidar_shape is not None)) or (ms_shape is None) and (lidar_shape is None):
            raise ValueError("Exactly one of parameters ms_shape and lidar_shape should receive a valid shape when using a single-sensor model")
            
        if ms_shape is not None:
            if ms_shape[0] > 100:
                feat_extractor = cerberuscnn_extractor(ms_shape)
            else:
                feat_extractor = tinycerberuscnn_extractor(ms_shape)
        else:
            feat_extractor = cerberuscnn_extractor(lidar_shape)


        # MLP classifier
        x = layers.Dense(128, activation='relu')(feat_extractor.output)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='sigmoid', name="presence_absence")(x)

        return Model(inputs=feat_extractor.input, outputs=outputs)
    

    else:
        # Feature extractor
        if (ms_shape is None) or (lidar_shape is None):
            raise ValueError("Both parameters ms_shape and lidar_shape must receive a valid shape when using a dual-sensor model")

        if ms_shape[0] > 100:
            ms_extractor = cerberuscnn_extractor(ms_shape)
        else:
            ms_extractor = tinycerberuscnn_extractor(ms_shape)

        lidar_extractor = cerberuscnn_extractor(lidar_shape)
    
        # Concatenate embeddings (192 + 192 = 384 features)
        combined = layers.Concatenate()([ms_extractor.output, lidar_extractor.output])
        
        # Fusion head
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='sigmoid', name="presence_absence")(x)
    
        return Model(inputs=[ms_extractor.input, lidar_extractor.input], outputs=outputs)

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import ResNet50

# def build_multiscale_cnn(input_shape=(128, 128, 4), num_classes=1):
#     """
#     Build a multi-scale Convolutional Neural Network (CNN) for image classification.

#     This model extracts features at multiple spatial scales using parallel convolutional
#     branches with different receptive fields. Features from each scale are concatenated
#     before passing through fully-connected layers.

#     Args:
#         input_shape (tuple): Shape of the input images, including channels (H, W, C).
#         num_classes (int): Number of output classes. If 1, uses sigmoid activation for binary classification.

#     Returns:
#         tf.keras.Model: Compiled Keras model.
#     """
#     inputs = layers.Input(shape=input_shape)

#     # First scale: Small receptive field
#     x1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
#     x1 = layers.MaxPooling2D((2, 2))(x1)
#     x1 = layers.BatchNormalization()(x1)
#     x1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
#     x1 = layers.MaxPooling2D((2, 2))(x1)
#     x1 = layers.GlobalAveragePooling2D()(x1)  # Flatten after final conv layer

#     # Second scale: Medium receptive field
#     x2 = layers.Conv2D(32, (5, 5), activation="relu", padding="same")(inputs)
#     x2 = layers.MaxPooling2D((4, 4))(x2)
#     x2 = layers.BatchNormalization()(x2)
#     x2 = layers.Conv2D(64, (5, 5), activation="relu", padding="same")(x2)
#     x2 = layers.MaxPooling2D((2, 2))(x2)
#     x2 = layers.GlobalAveragePooling2D()(x2)

#     # Third scale: Large receptive field
#     x3 = layers.Conv2D(32, (7, 7), activation="relu", padding="same")(inputs)
#     x3 = layers.MaxPooling2D((8, 8))(x3)
#     x3 = layers.BatchNormalization()(x3)
#     x3 = layers.Conv2D(64, (7, 7), activation="relu", padding="same")(x3)
#     x3 = layers.MaxPooling2D((2, 2))(x3)
#     x3 = layers.GlobalAveragePooling2D()(x3)

#     # Concatenate multi-scale features
#     merged = layers.Concatenate()([x1, x2, x3])

#     # Fully connected layers
#     dense = layers.Dense(256, activation="relu")(merged)
#     dense = layers.Dropout(0.5)(dense)
#     dense = layers.Dense(128, activation="relu")(dense)
#     dense = layers.Dropout(0.3)(dense)

#     # Output layer
#     output = layers.Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(dense)

#     model = models.Model(inputs, output)
#     return model

# def build_resnet(input_shape=(128, 128, 4), num_classes=1):
#     """
#     Build a ResNet50-based model for image classification.

#     Args:
#         input_shape (tuple): Shape of the input images (H, W, C).
#         num_classes (int): Number of output classes. If 1, uses sigmoid activation.

#     Returns:
#         tf.keras.Model: Compiled Keras model with ResNet50 backbone.
#     """
#     # Input layer definition
#     input_tensor = layers.Input(shape=input_shape)
    
#     # Build ResNet50 backbone 
#     resnet = ResNet50(input_tensor=input_tensor, weights=None, include_top=False)
    
#     # Add classification head
#     x = layers.GlobalAveragePooling2D()(resnet.output)
#     x = layers.Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(x)
#     model = models.Model(inputs=input_tensor, outputs=x)
    
#     return model

# def define_mlp(input_shape=(None, ), num_classes=1):
#     """
#     Define a simple Multi-Layer Perceptron (MLP) for binary classification.

#     Args:
#         input_shape (tuple): Shape of the input features.
    
#     Returns:
#         tf.keras.Model: Compiled Keras MLP model.
#     """
#     # Note: Input shape should be provided for the first layer
#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Dense(128, activation="relu", kernel_regularizer="l2"),
#         layers.Dense(64, activation="relu", kernel_regularizer="l2"),
#         layers.Dense(32, activation="relu", kernel_regularizer="l2"),
#         layers.Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")
#     ])
#     return model


# def build_high_res_extractor(input_shape, name="drone_ign_extractor"):
#     inputs = layers.Input(shape=input_shape, name=f"{name}_input")

#     # First scale: Small receptive field
#     x1 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
#     x1 = layers.MaxPooling2D((2, 2))(x1)
#     x1 = layers.BatchNormalization()(x1)
#     x1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
#     x1 = layers.MaxPooling2D((2, 2))(x1)
#     x1 = layers.GlobalAveragePooling2D()(x1)

#     # Second scale: Medium receptive field
#     x2 = layers.Conv2D(32, (5, 5), activation="relu", padding="same")(inputs)
#     x2 = layers.MaxPooling2D((4, 4))(x2)
#     x2 = layers.BatchNormalization()(x2)
#     x2 = layers.Conv2D(64, (5, 5), activation="relu", padding="same")(x2)
#     x2 = layers.MaxPooling2D((2, 2))(x2)
#     x2 = layers.GlobalAveragePooling2D()(x2)

#     # Third scale: Large receptive field
#     x3 = layers.Conv2D(32, (7, 7), activation="relu", padding="same")(inputs)
#     x3 = layers.MaxPooling2D((8, 8))(x3)
#     x3 = layers.BatchNormalization()(x3)
#     x3 = layers.Conv2D(64, (7, 7), activation="relu", padding="same")(x3)
#     x3 = layers.MaxPooling2D((2, 2))(x3)
#     x3 = layers.GlobalAveragePooling2D()(x3)

#     # Concatenate multi-scale features (Output size: 192)
#     merged = layers.Concatenate()([x1, x2, x3])
    
#     return Model(inputs, merged, name=name)


# def build_low_res_extractor(input_shape, name="s2_extractor"):
#     inputs = layers.Input(shape=input_shape, name=f"{name}_input")

#     # Scale 1: Pixel-level spectral signatures (1x1 Conv)
#     s1 = layers.Conv2D(64, (1, 1), activation="relu", padding="same")(inputs)
#     s1 = layers.BatchNormalization()(s1)
#     s1 = layers.GlobalAveragePooling2D()(s1)

#     # Scale 2: Local neighborhood (3x3 Conv)
#     s2 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
#     s2 = layers.BatchNormalization()(s2)
#     s2 = layers.GlobalAveragePooling2D()(s2)

#     # Scale 3: Global Patch Context 
#     # Directly pool the input and pass through a Dense layer
#     s3 = layers.GlobalAveragePooling2D()(inputs)
#     s3 = layers.Dense(64, activation="relu")(s3)
#     s3 = layers.BatchNormalization()(s3)

#     # Concatenate multi-scale features (Output size: 192)
#     # This perfectly matches the 192 output size of your high-res extractor!
#     merged = layers.Concatenate()([s1, s2, s3])
    
#     return Model(inputs, merged, name=name)

# def build_late_fusion_model(high_res_shape, low_res_shape, num_classes=1):
    
#     high_res_branch = build_high_res_extractor(high_res_shape, name="high_res_branch")
#     low_res_branch = build_low_res_extractor(low_res_shape, name="low_res_branch")
    
#     # Concatenate embeddings (192 + 192 = 384 features)
#     combined = layers.Concatenate()([high_res_branch.output, low_res_branch.output])
    
#     # # Fusion head
#     # x = layers.Dense(128, activation='relu')(combined)
#     # x = layers.Dropout(0.5)(x)
#     # outputs = layers.Dense(num_classes, activation='sigmoid', name="presence_absence")(x)

    
#     # Fusion head
#     dense = layers.Dense(256, activation="relu")(combined)
#     dense = layers.Dropout(0.5)(dense)
#     dense = layers.Dense(128, activation="relu")(dense)
#     dense = layers.Dropout(0.3)(dense)

#     # Output layer
#     output = layers.Dense(num_classes, activation="sigmoid" if num_classes == 1 else "softmax")(dense)
    
#     return Model(inputs=[high_res_branch.input, low_res_branch.input], outputs=outputs)