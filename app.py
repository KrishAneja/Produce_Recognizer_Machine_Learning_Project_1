import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import time
from PIL import Image


# Tensorflow Model Prediction
def model_prediction(image_p1):
    model = load_model('Trained_Model.h5')
    image1 = image.load_img(image_p1, target_size=(64, 64))
    input_arr = image.img_to_array(image1)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element


# Title
img007 = Image.open("Krish Pfp.png")
st.set_page_config(page_title='Produce Recognizer using Machine Learning', page_icon=img007)

# Sidebar
st.sidebar.title('DASHBOARD')
app_mode = st.sidebar.selectbox("SELECT A PAGE", ["Home", "About Project", "Prediction", "Feedback"])

# Main Page
if app_mode == "Home":
    original_title = ('<p style="font-family:Courier; color:black; font-size: 40px;'
                      ' font-weight:bold">PRODUCE RECOGNIZER</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    original_text = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                     'text-align: justify">Welcome to this Fruit & Vegetable Recognizer App,'
                     'through this app you can identify the images of fruits & Vegetables. '
                     'This Model is made using CNN (Convolutional Neural Network)...</p>')
    st.markdown(original_text, unsafe_allow_html=True)
    st.image("Home_image.jpg")

# About Project
elif app_mode == "About Project":
    original_title = ('<p style="font-family:Courier; color:black; font-size: 40px;'
                      ' font-weight:bold">ABOUT PROJECT</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    original_title = ('<p style="font-family:Courier; color:black; font-size: 25px;'
                      ' font-weight:bold">ABOUT THE MODEL</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                      'text-align: justify">The Model which we used in our Project is CNN '
                      '(Convolutional Neural Network) Model A Convolutional Neural Network (CNN) is a type of '
                      'deep learning algorithm that is particularly'
                      ' well-suited for image recognition and processing tasks. It is made up of multiple layers, '
                      'including convolutional layers, pooling layers, and fully connected layers.'
                      'The convolutional layers are the key component of a CNN, where filters are applied '
                      'to the input image to extract features such as edges, textures, and shapes. The output of the '
                      'convolutional layers is then passed through pooling layers, which are used to down-sample the '
                      'feature maps, reducing the spatial dimensions while retaining the most important information. '
                      'The output of the pooling layers is then passed through one or more fully connected layers, '
                      'which are used to make a prediction or classify the image. '
                      'CNNs are trained using a large dataset'
                      'of labeled images, where the network learns to recognize patterns and features'
                      ' that are associated with specific objects or classes.</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    st.image("About_Image.png")
    st.text("")
    original_text = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                     'text-align: justify; font-weight: bold">The Image below shows the layers of the '
                     'CNN Model, based on which '
                     'CNN Architecture is formed:</p>')
    st.markdown(original_text, unsafe_allow_html=True)
    st.image("Cnn_image.png")
    st.text("")
    original_title = ('<p style="font-family:Courier; color:black; font-size: 25px;'
                      ' font-weight:bold">ABOUT DATASET</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                      '">The dataset which we chose for our project contains'
                      ' images of the following food items:</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, "
            "raddish, beetroot, cabbage, lettuce, spinach, soya bean, cauliflower, bell pepper, "
            "chilli pepper, turnip, corn, sweet corn, sweet potato, paprika, "
            "jalapeno, ginger, garlic, peas, eggplant.")
    st.text("")
    original_title = ('<p style="font-family:Courier; color:black; font-size: 25px;'
                      ' font-weight:bold">CONTENT</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                      '">This dataset contains three folders:</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    st.code("train (Almost 100 images each)")
    st.code("test (Almost 10 images each)")
    st.code("validation (Almost 10 images each)")
    st.text("")

    original_title = ('<p style="font-family:Courier; color:black; font-size: 25px;'
                      ' font-weight:bold">MADE BY</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    st.code("KRISH ANEJA")
    st.code("PRANAYJEET SINGH")
    st.code("SHIVAM MAURYA")
    st.text("")

    original_title = ('<p style="font-family:Courier; color:black; font-size: 25px;'
                      ' font-weight:bold">PROJECT GUIDE</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    st.code("MR. RAHUL V. ANAND")

# Prediction Page
elif app_mode == "Prediction":
    original_title = ('<p style="font-family:Courier; color:black; font-size: 40px;'
                      ' font-weight:bold">MODEL PREDICTION</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    img123 = ('<p style="font-family:Courier; color:black; font-size: 17px;'
              '">Upload Image: </p>')
    st.markdown(img123, unsafe_allow_html=True)

    image_path = st.file_uploader("")
    if st.button("Display Image"):
        st.image(image_path, width=4, use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner('Model is predicting the image...Just wait for 5 seconds'):
            time.sleep(5)
        original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                          '">Model Prediction:</p>')
        st.markdown(original_title, unsafe_allow_html=True)

        result_index = model_prediction(image_path)
        # Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.balloons()
        st.success("The image which you have uploaded seems to be of......''{}''".format(label[result_index].upper()))

# Feedback Page
elif app_mode == "Feedback":
    original_title = ('<p style="font-family:Courier; color:black; font-size: 40px;'
                      ' font-weight:bold">FEEDBACK</p>')
    st.markdown(original_title, unsafe_allow_html=True)

    original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                      '">You can provide us your experience through this bar. Moreover, feel free to provide any '
                      'suggestion for the same:</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    feedback = st.select_slider("", ['🤮WORST', '😠VERY BAD', '🙁BAD', '😐AVERAGE', '🙂GOOD', '😊VERY GOOD', '🤯EXCELLENT'])
    st.text("")
    if st.button("Submit"):
        st.text("")
        original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                          '">Your thought about your project:</p>')
        st.markdown(original_title, unsafe_allow_html=True)
        st.write(feedback)
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    original_title = ('<p style="font-family:Courier; color:black; font-size: 17px;'
                      '">Your suggestion:</p>')
    st.markdown(original_title, unsafe_allow_html=True)
    st.text_area('')
    if st.button("Submit Suggestion"):
        st.snow()
        st.success("Thank you for your suggestion, We will surely work on our development...")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)
