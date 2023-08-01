from flask import Flask, render_template, request, jsonify
import numpy as np  # For numerical operations
import base64       # To handle image encoding
import matplotlib.pyplot as plt # To plot the probability bar chart

import tensorflow as tf
from PIL import Image
import io

MODEL = None

app = Flask(__name__)

def module_output_to_numpy(tensor):
    return tensor.numpy()

# Preprocess the drawn image (convert it to grayscale and normalize)
def preprocess_image(image_data):

    offset = image_data.index(',') + 1
    img_bytes = base64.b64decode(image_data[offset:])
    img = Image.open(io.BytesIO(img_bytes))

    # DEBUG: save image to file
    # img.save('image.png')

    img = img.convert('L')

    # DEBUG: save image to file
    # img.save('g-image.png')

    img = img.resize((28, 28))

    # DEBUG: save image to file
    # img.save('re-image.png')

    img = np.array(img)
    img = img.reshape((1, 28, 28))
    img = img.astype('float32') / 255.0

    # Convert the numpy array to a TensorFlow tensor
    tensor = tf.convert_to_tensor(img)

    # DEBUG: check tensor shape
    print(tensor.shape)

    return tensor

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def prob_img(probs):
    # Transform the probs from 0-1 to 0-100
    probs = [prob * 100 for prob in probs]

    fig, ax = plt.subplots()
    rects = ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    ax.set_ylim(0, 110)
    ax.set_title('Probability % of Digit by Model')
    autolabel(rects, ax)
    probimg = io.BytesIO()
    fig.savefig(probimg, format='png')
    probencoded = base64.b64encode(probimg.getvalue()).decode('utf-8')
    return probencoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/', methods=['POST'])
def predict():
    image_data = request.form['imageBase64']

    processed_image = preprocess_image(image_data)

    # Perform prediction
    prediction = MODEL.predict(processed_image)

    # Find the predicted digit (the index with the highest probability)
    predicted_digit = np.argmax(prediction)

    probability_list = prediction.tolist()[0]

    probencoded = prob_img(probability_list)

    return jsonify({'prediction': int(predicted_digit), 'probabilities': probencoded})


if __name__ == '__main__':
    # Load the model
    MODEL = tf.keras.models.load_model('models/mnist_cnn.h5')
    app.run(debug=True)
