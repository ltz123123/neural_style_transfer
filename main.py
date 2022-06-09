import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.utils import get_file
from vgg_model import build_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def preprocess_image(image_path, n_row, n_col):
    img = img_to_array(load_img(image_path, target_size=(n_row, n_col)))
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return tf.convert_to_tensor(img)


def postprocess_image(x):
    output = np.squeeze(x, axis=0)
    output[:, :, 0] += 103.939
    output[:, :, 1] += 116.779
    output[:, :, 2] += 123.68
    output = output[:, :, ::-1]
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output


def load_mask(path, n_row, n_col):
    mask = img_to_array(load_img(path, color_mode="grayscale", target_size=(n_row, n_col)))

    return mask.astype(np.float32)


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)

    return G


def style_loss(target, combination):
    n_row, n_col, channels = target.shape
    size = n_row * n_col

    target_gram = gram_matrix(target, size, channels)
    combination_gram = gram_matrix(combination, size, channels)

    return tf.reduce_sum(tf.square(target_gram - combination_gram)) / (4.0 * size * size * channels * channels)


def sum_style_loss(target_features, features, image_ws, layer_names, layer_ws, mask=None):
    loss = tf.zeros(shape=())

    for idx, img_weight in enumerate(image_ws):
        for layer_w, layer_name in zip(layer_ws, layer_names):
            combination_layer_features = tf.squeeze(features[layer_name], axis=0)
            target_layer_features = target_features[layer_name][idx, :, :, :]

            if mask is None:
                loss += img_weight * layer_w * style_loss(
                    combination_layer_features, target_layer_features
                )
            else:
                n_row, n_col, channel = combination_layer_features.shape
                resized_mask = tf.image.resize(mask[..., np.newaxis], (n_row, n_col), method="area")
                resized_mask = tf.clip_by_norm(resized_mask, 1.0)
                resized_mask = tf.broadcast_to(tf.cast(resized_mask, tf.float32), [n_row, n_col, channel])
                loss += img_weight * layer_w * style_loss(
                    combination_layer_features * resized_mask, target_layer_features * resized_mask
                )

    return loss


def content_loss(target, combination):
    n_row, n_col, channels = target.shape

    return tf.reduce_sum(tf.square(target - combination)) / (2.0 * n_row * n_col * channels)


def total_variation_loss(x):
    _, row, col, channel = x.shape
    a = tf.square(x[:, : row - 1, : col - 1, :] - x[:, 1:, : col - 1, :])
    b = tf.square(x[:, : row - 1, : col - 1, :] - x[:, : row - 1, 1:, :])

    return tf.reduce_sum(tf.pow(a + b, 1.25))


def vgg_layers(pooling="avg"):
    vgg = build_model(pooling=pooling)
    vgg.trainable = False
    outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

    return Model(inputs=vgg.inputs, outputs=outputs_dict)


def convert_to_original_colors(content_img, stylized_img):
    content_img = content_img.astype(np.float32) / 255.0
    stylized_img = stylized_img.astype(np.float32) / 255.0

    cvt_type = cv2.COLOR_BGR2YUV
    inv_cvt_type = cv2.COLOR_YUV2BGR

    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)

    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)

    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = np.clip(dst * 255.0, 0, 255).astype(np.int)

    return dst


# Image path
# style_image_paths = [get_file("starry_night.jpg", "https://i.imgur.com/9ooB60I.jpg")]
# content_image_path = get_file("paris.jpg", "https://i.imgur.com/F28w3Ac.jpg")
style_image_paths = ["style_img/unnamed.jpg"]
content_image_path = "content_img/tuebingen.jpg"


# Image size
width, height = load_img(content_image_path).size
img_nrows = 512
img_ncols = int(width * img_nrows / height)


# Load image
content_image = preprocess_image(content_image_path, img_nrows, img_ncols)
style_image = tf.concat([preprocess_image(path, img_nrows, img_ncols) for path in style_image_paths], axis=0)
combination_image = tf.Variable(preprocess_image(content_image_path, img_nrows, img_ncols))
# img_mask = np.zeros((img_nrows, img_ncols))
# img_mask[:, img_ncols//2:] += 1
# img_mask = np.squeeze(load_mask("content_img/mask.jpg", img_nrows, img_ncols), axis=-1)
img_mask = None


# Weights
layer_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
layer_weights /= np.sum(layer_weights)
style_image_weights = [1.0]
style_loss_weight = 1e-3
content_loss_weight = 1e-8  # 7 to 9
total_variation_weight = 1e-8


# Model
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
content_layer_name = "block5_conv2"
extractor = vgg_layers(pooling="avg")
optimizer = Adam(learning_rate=1)


# Target layer features for loss calculation
content_target = extractor(content_image)
style_target = extractor(style_image)


def compute_loss(combination):
    combination_features = extractor(combination)
    loss = tf.zeros(shape=())

    # Content loss
    content_features = tf.squeeze(content_target[content_layer_name], axis=0)
    combination_content_features = tf.squeeze(combination_features[content_layer_name], axis=0)
    loss += content_loss_weight * content_loss(content_features, combination_content_features)

    # Style loss
    loss += style_loss_weight * sum_style_loss(
        style_target,
        combination_features,
        style_image_weights,
        style_layer_names,
        layer_weights,
        mask=img_mask
    )

    # Total variation loss
    # loss += total_variation_weight * tf.image.total_variation(tf.squeeze(combination, axis=0))
    loss += total_variation_weight * total_variation_loss(combination)

    return loss


@tf.function
def compute_loss_and_grads(combination):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image)

    return loss, tape.gradient(loss, combination)


# out = cv2.VideoWriter("video_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (img_ncols, img_nrows))

iterations = 2000
for i in tqdm(range(1, iterations + 1)):
    total_loss, grads = compute_loss_and_grads(combination_image)
    optimizer.apply_gradients([(grads, combination_image)])

    # print("Iteration %d: loss=%.2f" % (i, total_loss))
    # output_img = postprocess_image(combination_image.numpy())
    # out.write(cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
# out.release()

output_img = postprocess_image(combination_image.numpy())
# output_img = convert_to_original_colors(
#     img_to_array(load_img(content_image_path, target_size=(img_nrows, img_ncols))),
#     output_img
# )
save_img("358.jpg", output_img)
