#reasons why use tflite because h5 is heavy  and much slower to run
import streamlit as st #for ui | for over all widgets check here: https://docs.streamlit.io/develop/api-reference/widgets 
import tensorflow as tf # the brain
import numpy as np # process
from PIL import Image, ImageOps #for loading and processing the image
import base64 as player #for playing the audio
#set a title for the page

def auto_play_audio(file_path):
    with open(file_path, "rb") as A:
        data = A.read()
        b64 = player.b64encode(data).decode()
        md=f"""
            <audio autolay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

st.set_page_config("Ai art vs Human art Detector")

#title naman para sa loob ng page
st.title(f"**Ai art vs Human art Detector**")
st.write("Upload an art to see if it was made by Human or AI.")

#Setting up the ai brain
def analyzer():
    #loading the tflite model and allocate tensors
    #source: https://ai.google.dev/edge/litert/microcontrollers/python 
    model = tf.lite.Interpreter(model_path="model_unquant.tflite")
    model.allocate_tensors()#mandatory so that thefunction pre-allocates the necessary memory for the tensors to ensure efficient execution.
    return model

#getting the input and output details
model = analyzer()
input_details = model.get_input_details()
output_details = model.get_output_details()

#lets the user upload files
uploaded_image = st.file_uploader("Choose an art...", type=["jpg","png","jpeg","webp"])

#kung may na upload na file mag di display
if uploaded_image:
    #tell to pil to open this image and convert it to rgb
    image = Image.open(uploaded_image).convert("RGB")
    #without this portrait images will be landscape
    image = ImageOps.exif_transpose(image)
    #displaying the image
    st.image(image, caption="Uploaded art", use_container_width= True)

    #pre-processing of the image
    with st.spinner("Analyzing art.."):
        #set size of the image
        size = (224, 224)
        #we use lancos because it is best for downscaling big images while maintaining quality
        cropted_image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(cropted_image).astype(np.float32) #converts colors to number because normaly computers don't see colors they only see numbers
        normalized_image = (image_array/127.5)-1
        input_data = np.expand_dims(normalized_image, axis=0)

        #set input data
        model.set_tensor(input_details[0]['index'],input_data)
        model.invoke()
        #get the ouptut data
        output_data = model.get_tensor(output_details[0]['index'])

        #for reading the labels.txt
        with open("labels.txt","r") as f:
            #normally without this, the label will be read as "0 Ai art"
            #line.strip()[2:] we are exactly excludint 0(space) so it will be read as "Ai art"
            #for line in f.readlines() we tell python to look each lines individually (yes spaces are included)
            labels = [line.strip()[2:] for line in f.readlines()]
        scores = output_data[0]
        st.divider()
        st.subheader("Classification Breakdown")
        
        char_data = {}#dictionary to hold the values of 0(Ai image) and 1(Real image).
        for i in range(len(labels)):
            char_data[labels[i]] = float(scores[i])

        st.bar_chart(char_data)
        
        index = np.argmax(scores)
        confidence = scores[index]
        results = labels[index]


        st.divider()
        if results == "Ai art":
            st.error(f"**Results:** This looks like an **AI Generated art**")
            #st.audio("Fahhhh - Sound effect (HD).mp3", format="audio/mpeg")
            auto_play_audio("Fahhhh - Sound effect (HD).mp3")
        else:
            st.success(f"**Results:** This looks like a **Human-Made art**")
            st.balloons()
        st.info(f"**Confidence Score:** {confidence:.4%}")

