#Author : ADISH007

import numpy as np
from doctr.models import ocr_predictor
from PIL import Image
import math

import streamlit as st
from streamlit_float import *
import base64
import json
import os

st.set_page_config(page_title='DONUT Annotation Tool', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)


st.header(':bagel: DONUT ANNOTATION TOOL ',divider='rainbow')
#st.caption('Upload Image to get started :sunglasses:',icon="✅")
st.info('✅ Upload Image to get started ')
with st.sidebar.expander("Tool info: "):
     st.write("""
         Read OCR and map into [KEY,VALUE] pair and :arrow_down: Download as JSON
     """,width=10,use_column_width=20)
     st.text("OCR MODEL : DocTR")
     st.text("Model Parameters :")
     st.text("det_arch='db_resnet50'")
     st.text("reco_arch='crnn_vgg16_bn'")
     st.text('Creator : ADISH007')
with st.sidebar:
    # Upload image
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if st.sidebar.button("Clear Cache"):
    for key in st.session_state.keys():
        if key!='model':
            del st.session_state[key]
    st.info("Cache cleared. You can now upload a new image.")

st.sidebar.image("https://www.livemint.com/lm-img/img/2023/06/01/1600x900/donuts_1685654257861_1685654258152.jpg")

json_dict={}

if 'json_dict' not in st.session_state:
    st.session_state.json_dict = {}


def convert_coordinates(geometry, page_dim):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]

def get_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:                
                converted_coordinates = convert_coordinates(
                                           obj3["geometry"], page_dim
                                          )
                text_coordinates.append([converted_coordinates, obj3["value"]])
    
    # Sort the text_coordinates list based on the y_min coordinate (index 2 in each sub-list)
    sorted_coordinates = sorted(text_coordinates, key=lambda x: x[0][2])
    
    return sorted_coordinates

def get_csv(data):
    graphical_coordinates = get_coordinates(data)
    # Extract only the values from graphical_coordinates
    values = [value for coordinates, value in graphical_coordinates]
    # Join the values into a CSV format
    csv_data = ','.join(values)
    return csv_data

@st.cache_resource
def load_ocr():
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    return model


def create_float_box(uploaded_image):
    # Float feature initialization
    float_init()

    # Initialize session variable that will show/hide Float Box
    if "show" not in st.session_state:
        st.session_state.show = False  # Change this to False

    # Container with expand/collapse button
    button_container = st.container()
    with button_container:
        if st.session_state.show:
            if st.button("⭳", type="primary"):
                st.session_state.show = False
                st.experimental_rerun()
        else:
            if st.button("⭱", type="secondary"):
                st.session_state.show = True
                st.experimental_rerun()

    # Alter CSS based on expand/collapse state
    if st.session_state.show:
        vid_y_pos = "2rem"
        button_b_pos = "21rem"
    else:
        vid_y_pos = "-19.5rem"
        button_b_pos = "1rem"

    button_css = float_css_helper(width="2.2rem", right="2rem", bottom=button_b_pos, transition=0)

    # Float button container
    button_container.float(button_css)

    # Convert uploaded image to data URL
    if uploaded_image and st.session_state.show:  # Add condition here
        # Read the binary data of the uploaded image
        image_data = uploaded_image.getvalue()
        # Encode the binary data to base64
        uploaded_image_data_url = f"data:{uploaded_image.type};base64,{base64.b64encode(image_data).decode()}"
    else:
        uploaded_image_data_url = ""

    # Add Float Box with embedded image
    float_content = f'''
    <img src="{uploaded_image_data_url}" alt="Your Image" width="100%" height="auto">
    '''
    float_box(float_content, width="29rem", right="2rem", bottom=vid_y_pos, css="padding: 0;transition-property: all;transition-duration: .5s;transition-timing-function: cubic-bezier(0, 1, 0.5, 1);", shadow=12)


@st.cache_data
def create_json(json_dict, column_header,column_values):
    if len(column_header)>1:
        column_header = ''.join(column_header)
    elif len(column_header)==1:
        column_header=column_header[0]
    json_dict[column_header]=column_values
    return json_dict

@st.cache_data
def load_image(uploaded_image):
    return Image.open(uploaded_image)

if uploaded_image is not None:
    # Get the image name
    image_name = uploaded_image.name
    # Remove the extension
    base_name = os.path.splitext(image_name)[0]
    image = load_image(uploaded_image)
    image_array = np.array(image)  # Convert the PIL Image to a NumPy array
    create_float_box(uploaded_image)
    
    tab1, tab2 = st.tabs(["📈 Uploaded Image", "🗃 Create Annotation"])
    with tab1:
        st.header("VISUALIZATION :")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    with tab2:
        #st.header("ANNOTATION :")
        # Display the OCR results
        # Load the OCR model
        if 'model' not in st.session_state or 'results' not in st.session_state:
            with st.spinner('Loading DocTR Model ...'):
                if 'model' not in st.session_state:
                    st.session_state.model = load_ocr()
                model = st.session_state.model
            with st.spinner('Loading OCR Results for Uploaded Image'):
                st.session_state.results = model([image_array])  # Wrap the NumPy array in a list        
        data=st.session_state.results.export()
        rows = []
        columns = []
        # Iterate through pages, blocks, and lines to organize words into rows and columns
        for page in data['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    row = []
                    column = []
                    for word in line['words']:
                        # Extract word value and add it to the current row and column
                        word_value = word['value']
                        row.append(word_value)
                        column.append(word_value)
                    #rows.append(row)
                    columns.append(' '.join(column))
        show_csv = st.toggle("Show_extracted_csv")
        if show_csv:
            csv_data=st.text_area('Extracted CSV [ROW_WISE]:',get_csv(data) )
        with st.form("json",clear_on_submit=True):
            column_header = st.multiselect(
            'KEY (Eg : Column header )',
            columns)

            column_values = st.multiselect(
            'VALUE (Eg : Column values )',
            columns,
            [])
            # Every form must have a submit button.
            submitted = st.form_submit_button("ADD KEY_VALUE_PAIR")
            if submitted:
                st.session_state.json_dict = create_json(st.session_state.json_dict, column_header, column_values)
                st.json(st.session_state.json_dict)
            
        txt = st.text_area('Edit JSON :',st.session_state.json_dict )
        st.download_button(
        label="Create Download Json Link 	:arrow_down: ",
        data=txt,
        file_name=base_name+'.csv',
        mime='text/csv',
    )
