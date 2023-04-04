import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import cv2
import ast
import pandas as pd

# load csv with labels (polars = rho and theta; labels = x and y pixel coordinates)
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['polars'] = df['polars'].apply(lambda x: ast.literal_eval(x))
    df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x))
    return df

# preparing images for centernet
def prepare_data(df, img_folder):
    images = []
    boxes = []
    for index, row in df.iterrows():
        img_file = os.path.join(img_folder, row['name'] + '-ht.png')
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Erro ao carregar a imagem: {img_file}")
            continue

        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
        image_tensor = tf.cast(image_tensor * 255, dtype=tf.uint8)
        images.append(image_tensor)

        labels = row['labels']
        box = [labels[0][1] / image.shape[1], labels[0][0] / image.shape[0],
               labels[0][1] / image.shape[1], labels[0][0] / image.shape[0]]
        boxes.append(box)

    return images, boxes

def input_pipeline(csv_path, img_folder):
    df = load_csv(csv_path)
    images, boxes = prepare_data(df, img_folder)
    return images, boxes

def train_model(images, boxes):
    # loading pretrained tensorflow model
    model_url = "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1"
    model = hub.KerasLayer(model_url, trainable=True)

    # creating a new keras model with an initial layer and the pretrained model
    input_layer = tf.keras.layers.Input(shape=(None, None, 3))
    output_layer = model(input_layer)
    finetuned_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # compiling the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = tf.keras.losses.MeanSquaredError()
    finetuned_model.compile(optimizer=optimizer, loss=loss)

    # preparing training data
    X_train = np.stack(images)
    y_train = np.array(boxes)

    # training and saving the model
    epochs = 10
    batch_size = 8
    finetuned_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    finetuned_model.save("finetuned_model.h5")

    return finetuned_model

def show_original_image_with_bounding_box(finetuned_model, img_file):
    # loading original image
    original_image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # normalizing the image and expanding dimensions for the batch
    image_tensor = tf.convert_to_tensor(original_image, dtype=tf.float32) / 255.0
    image_tensor = tf.cast(image_tensor * 255, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # inferring the image
    output = finetuned_model(image_tensor)
    boxes, classes, scores = output['detection_boxes'], output['detection_classes'], output['detection_scores']

    # threshold for detected peaks
    score_threshold = 0.5

    # loop over detections
    for box, cls, score in zip(boxes[0], classes[0], scores[0]):
        if score >= score_threshold:
            ymin, xmin, ymax, xmax = [int(x) for x in (box * np.array([original_image.shape[0], original_image.shape[1], original_image.shape[0], original_image.shape[1]]))]
            cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # return original ht image with bounding box around the peak
    cv2.imshow("Detections", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    csv_path = r'ht-peaks-dataset/1l_1n/1l_1n.csv'  # add the path to the csv file with labels
    img_folder = r'ht-peaks-dataset/1l_1n'  # add the path to the folder with dataset images
    images, boxes = input_pipeline(csv_path, img_folder)
    finetuned_model = train_model(images, boxes)
    img_file = r'ht-image.png'  # add the path to a new ht image for peak extraction
    show_original_image_with_bounding_box(finetuned_model, img_file)

if __name__ == '__main__':
    main()
