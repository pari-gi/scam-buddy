import os
import cv2
import re
import numpy as np
import tensorflow as tf
import re as regex
from PIL import Image
from PIL import UnidentifiedImageError
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
import ssl
import pytesseract

# Download stopwords
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def clean_text(text):
    text = text.replace('\n', ' ')
    text = regex.sub(r'[^0-9A-Za-z%.$-~ ]', ' ', text)
    text = regex.sub(r'\s+', ' ', text).strip()
    return text

def process_images(directory):
    supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif']
    unsupported_formats = ['webp', 'avif']

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            extension = file.split('.')[-1].lower()

            # Skip hidden files
            if file.startswith('.'):
                print(f"Skipping hidden file: {file_path}")
                continue

            # Convert unsupported formats to jpg
            if extension in unsupported_formats:
                jpg_path = os.path.splitext(file_path)[0] + '.jpg'
                try:
                    with Image.open(file_path) as img:
                        img.convert('RGB').save(jpg_path, 'JPEG')
                        print(f"Converted {file_path} to {jpg_path}")
                    os.remove(file_path)  # Delete original unsupported file
                except Exception as e:
                    print(f"Error converting {file_path}: {e}")
                    continue

            # Check if file is a valid image
            if extension not in supported_formats:
                print(f"Skipping unsupported file: {file_path}")
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check for valid image file
                    img = tf.io.read_file(file_path)
                    img = tf.image.decode_image(img)

                    # Ensure correct number of channels
                    if img.shape[-1] not in [1, 3]:
                        #print(f"Unsupported number of channels in image: {file_path}")
                        continue

                #print(f"Successfully processed image: {file_path}")
            except (UnidentifiedImageError, Exception) as e:
                print(f"Deleting invalid image: {file_path}, Error: {e}")
                os.remove(file_path)
                

def load_images_with_filenames(directory):
    image_paths = []
    labels = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            extension = file.split('.')[-1].lower()
            if extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                image_paths.append(file_path)
                label = 1 if 'spam' in root else 0
                labels.append(label)
    
    return image_paths, labels

def load_and_preprocess_image(file_path):
    print(f"Attempting to process file: {file_path}")
    
    try:
        image = tf.io.read_file(file_path)
        image = tf.image.decode_image(image, channels=3)
        print(f"Loaded image shape: {image.shape}")
        
        if image.shape is None or image.shape.ndims != 3:
            raise ValueError(f"Invalid image shape for file: {file_path}")

        image = tf.image.resize(image, [256, 256])
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return tf.zeros([256, 256, 3])

def predict_spam_image(img_path):
    img = cv2.imread(img_path)
    resize = tf.image.resize(img, (256, 256))
    yhat_cnn = model.predict(np.expand_dims(resize / 255, 0))[0][0]
    extracted_text = pytesseract.image_to_string(Image.open(img_path))
    cleaned_text = clean_text(extracted_text)
    text_features = tfidf.transform([cleaned_text]).toarray()
    yhat_nb = spam_detect_model.predict(text_features)[0]
    yhat_nb_proba = spam_detect_model.predict_proba(text_features)[0][1]
    final_score = (0.7 * yhat_nb_proba) + (0.3 * yhat_cnn)

    if final_score > 0.5:
        return 'Predicted class is Spam'
    else:
        return 'Predicted class is not Spam'

def test_predict_spam_image(image_path):
    if os.path.exists(image_path):
        prediction = predict_spam_image(image_path)
        print(f"Prediction for {image_path}: {prediction}")
    else:
        print(f"Error: The image file {image_path} does not exist.")

def calculate_overall_accuracy(test_dataset, cnn_model, text_model, tfidf_vectorizer, weight_cnn=0.3, weight_text=0.7):
    correct_predictions = 0
    total_samples = 0

    for batch_images, batch_labels in test_dataset:
        for i in range(len(batch_images)):
            image = batch_images[i]
            yhat_cnn = cnn_model.predict(np.expand_dims(image, axis=0))[0][0]
            img_path = image_paths[total_samples]

            try:
                extracted_text = pytesseract.image_to_string(Image.open(img_path))
            except Exception as e:
                print(f"Error extracting text from {img_path}: {e}")
                extracted_text = ""
            
            cleaned_text = clean_text(extracted_text)
            text_features = tfidf_vectorizer.transform([cleaned_text]).toarray()
            yhat_text_proba = text_model.predict_proba(text_features)[0][1]
            final_score = (weight_text * yhat_text_proba) + (weight_cnn * yhat_cnn)
            final_prediction = 1 if final_score > 0.5 else 0

            if final_prediction == batch_labels[i]:
                correct_predictions += 1

            total_samples += 1

    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return overall_accuracy

def custom_training_loop(dataset, model):
    for step, (batch_images, batch_labels) in enumerate(dataset):
        for i, img in enumerate(batch_images):
            try:
                file_path = image_paths[step * 8 + i]
                print(f"Processing file: {file_path}")
                print(f"Processing image {i + 1} in batch {step + 1} with shape: {img.shape}")

                prediction = model(np.expand_dims(img, axis=0), training=False)
            except Exception as e:
                print(f"Error processing image in batch {step + 1}, image {i + 1}: {e}")
                print(f"Problematic file: {file_path}")
                continue

process_images('data/spam')
process_images('data/not_spam')

image_paths, labels = load_images_with_filenames('data')
file_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths)
labels_ds = tf.data.Dataset.from_tensor_slices(labels)
images_ds = file_paths_ds.map(load_and_preprocess_image)
dataset = tf.data.Dataset.zip((images_ds, labels_ds))
dataset = dataset.batch(16)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'data'
data = tf.keras.utils.image_dataset_from_directory(data_dir, batch_size=16)
data = data.map(lambda x, y: (x / 255, y))

model = Sequential([
    Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3, 3), 1, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

train_size = int(len(data) * .7)
val_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=25, validation_data=val, callbacks=[tensorboard_callback])

pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

dataset_dir = '/Users/parigill/Documents/project/data'
data = []

for label in ['spam', 'not_spam']:
    folder_path = os.path.join(dataset_dir, label)
    for image_name in os.listdir(folder_path):
        if image_name.endswith(('.jpg', '.png')):
            try:
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Unable to read {image_path}, skipping.")
                    continue
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=1)
            mask_inv = cv2.bitwise_not(mask)
            blurred_gray = cv2.GaussianBlur(gray, (51, 51), 0)
            text_preserved = cv2.bitwise_and(gray, gray, mask=mask)
            blurred_background = cv2.bitwise_and(blurred_gray, blurred_gray, mask=mask_inv)
            final_image = cv2.add(text_preserved, blurred_background)
            filename = "{}.jpg".format(os.getpid())
            cv2.imwrite(filename, final_image)
            text = pytesseract.image_to_string(Image.open(filename))
            os.remove(filename)
            cleaned_text = clean_text(text)
            data.append((cleaned_text, label))

corpus = []
labels = []
ps = PorterStemmer()
for text, label in data:
    review = text.lower().split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    labels.append(1 if label == 'spam' else 0)

tfidf = TfidfVectorizer(max_features=4000)
X = tfidf.fit_transform(corpus).toarray()
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred))

test_image_path = '/Users/parigill/Documents/project/data/spam/download-4.jpg'

if os.path.exists(test_image_path):
    prediction = predict_spam_image(test_image_path)
    print(f"Prediction for {test_image_path}: {prediction}")
else:
    print(f"Error: The image file {test_image_path} does not exist.")

test_predict_spam_image(test_image_path)

overall_accuracy = calculate_overall_accuracy(test, model, spam_detect_model, tfidf)
print(f"Overall accuracy: {overall_accuracy}")


