import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import NUM_CLASSES_RETINANET, MODEL_INPUT_SIZE_RETINANET, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, IMAGENET_WEIGHTS_PATH

# Helper function to create a convolution block
def conv_block(filters, kernel_size, strides=1, padding='same', activation='relu'):
    def apply(x):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                          kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        if activation:
            x = layers.Activation(activation)(x)
        return x
    return apply

# Feature Pyramid Network (FPN)
def build_fpn(backbone_outputs):
    """
    Builds a Feature Pyramid Network (FPN) on top of the ResNet backbone outputs.
    :param backbone_outputs: Dictionary of C3, C4, C5 outputs from ResNet.
    """
    C3, C4, C5 = backbone_outputs['C3'], backbone_outputs['C4'], backbone_outputs['C5']

    # Lateral connections
    P5 = conv_block(256, 1)(C5)
    P4 = conv_block(256, 1)(C4)
    P3 = conv_block(256, 1)(C3)

    # Upsample and add
    P4 = layers.Add()([P4, layers.UpSampling2D(size=(2, 2))(P5)])
    P3 = layers.Add()([P3, layers.UpSampling2D(size=(2, 2))(P4)])

    # Smooth P3, P4, P5
    P3 = conv_block(256, 3)(P3)
    P4 = conv_block(256, 3)(P4)
    P5 = conv_block(256, 3)(P5)

    # P6 and P7 (for larger receptive fields)
    P6 = conv_block(256, 3, strides=2)(P5) # Downsample P5
    P7 = conv_block(256, 3, strides=2)(P6) # Downsample P6

    return [P3, P4, P5, P6, P7]

# Subnet for classification and regression
def build_subnet(num_filters, num_layers, output_filters):
    """Builds a classification or regression subnet."""
    def apply(x):
        for _ in range(num_layers):
            x = conv_block(num_filters, 3)(x)
        x = layers.Conv2D(output_filters, 3, padding='same',
                          kernel_initializer=tf.random_normal_initializer(stddev=0.01), bias_initializer='zeros')(x)
        return x
    return apply

# Focal Loss implementation
def focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA):
    def focal_loss_fixed(y_true, y_pred):
        # y_true is one-hot encoded, y_pred is probabilities
        # y_true can also contain -1 for padded labels
        
        # Filter out padded instances where y_true == -1
        mask = tf.where(tf.not_equal(y_true, -1), True, False)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(tf.abs(1 - y_pred), gamma) * cross_entropy
        
        return tf.reduce_sum(loss, axis=-1) # Sum across classes for each instance
    return focal_loss_fixed

# Bounding box regression loss (Smooth L1)
def smooth_l1_loss(sigma=1.0):
    sigma_squared = sigma ** 2
    def _smooth_l1_loss_fixed(y_true, y_pred):
        # y_true: [batch, num_anchors, 4] (ground truth box deltas)
        # y_pred: [batch, num_anchors, 4] (predicted box deltas)
        
        # We need to consider background/foreground and only compute loss for foreground anchors
        # This implementation assumes y_true has 0 for background and actual deltas for foreground
        # A more robust implementation would use anchor labels (positive/negative/neutral)
        
        # For simplicity, assuming y_true contains actual deltas for positive anchors and 0s for others
        # And we only care about non-zero targets.
        
        diff = tf.abs(y_true - y_pred)
        
        less_than_one = tf.cast(tf.less(diff, 1.0 / sigma_squared), tf.float32)
        
        loss = (less_than_one * 0.5 * sigma_squared * tf.pow(diff, 2) +
                (1 - less_than_one) * (diff - 0.5 / sigma_squared))
        
        # Only compute loss for instances where y_true is not all zeros (i.e., there's a target box)
        object_mask = tf.cast(tf.reduce_any(tf.not_equal(y_true, 0.0), axis=-1), tf.float32)
        loss = loss * tf.expand_dims(object_mask, -1) # Apply mask
        
        # Sum across the 4 box coordinates, then average across active anchors
        return tf.reduce_sum(loss, axis=-1) / (tf.reduce_sum(object_mask) + tf.keras.backend.epsilon())
    return _smooth_l1_loss_fixed


def build_retinanet(input_shape=MODEL_INPUT_SIZE_RETINANET + (3,), num_classes=NUM_CLASSES_RETINANET, num_anchors_per_location=9):
    """
    Builds the RetinaNet model.
    :param input_shape: Input image shape (height, width, channels).
    :param num_classes: Number of object classes (excluding background).
    :param num_anchors_per_location: Number of anchor boxes at each pyramid level pixel.
    """
    input_tensor = keras.Input(shape=input_shape, name='input_image')

    # 1. Backbone: ResNet50
    # Use ResNet50 without the top classification layers
    resnet50 = keras.applications.ResNet50(
        include_top=False, weights=IMAGENET_WEIGHTS_PATH, input_tensor=input_tensor
    )
    # Get outputs from different stages for FPN
    # C3: output of layer 'conv3_block4_out'
    # C4: output of layer 'conv4_block6_out'
    # C5: output of layer 'conv5_block3_out'
    backbone_outputs = {
        'C3': resnet50.get_layer('conv3_block4_out').output,
        'C4': resnet50.get_layer('conv4_block6_out').output,
        'C5': resnet50.get_layer('conv5_block3_out').output,
    }

    # 2. FPN
    fpn_features = build_fpn(backbone_outputs) # List of P3, P4, P5, P6, P7

    # 3. Classification and Regression Subnets
    classification_subnet = build_subnet(256, 4, num_anchors_per_location * num_classes)
    regression_subnet = build_subnet(256, 4, num_anchors_per_location * 4) # 4 for bbox deltas (tx, ty, tw, th)

    classification_outputs = []
    regression_outputs = []

    for feature_map in fpn_features:
        classification_outputs.append(classification_subnet(feature_map))
        regression_outputs.append(regression_subnet(feature_map))

    # Concatenate all outputs across feature maps
    classification_output = layers.Concatenate(axis=1, name='classification_output')([
        layers.Reshape((-1, num_classes))(o) for o in classification_outputs
    ])
    regression_output = layers.Concatenate(axis=1, name='regression_output')([
        layers.Reshape((-1, 4))(o) for o in regression_outputs
    ])

    # Apply sigmoid to classification outputs for probabilities
    classification_output = layers.Activation('sigmoid')(classification_output)

    model = keras.Model(inputs
