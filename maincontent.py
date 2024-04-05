import streamlit as st
from HTML_functions import render_example
from db_comm import fetch_model_info, fetch_model_names


def main_content():
    #set_background('images\save23.jpg')
    # Model page content
    with st.container():

        st.title('Machine Learning Cook Book')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            model_type = st.selectbox("Learning Type", ['All Models', 'Supervised Learning', 'Unsupervised Learning'])
        if model_type == 'Unsupervised Learning':
            with col2:
                model_subtype = st.selectbox("Task Type", ['Clustering', 'Dimensionality Reduction'])
        elif model_type == 'All Models':
            model_subtype = 'All Models'
        else:
            with col2:
                model_subtype = st.selectbox("Task Type", ['All Models', 'Classification', 'Regression', 'Ensemble Learning'])

        # Retrieve model names based on selected type and subtype
        model_names = fetch_model_names(model_type, model_subtype)
        if model_type == "All Models":
            with col2:
                selected_model = st.selectbox('Select a model:', model_names)
        else:
            with col3:
                selected_model = st.selectbox('Select a model:', model_names)

        st.divider()


        # Display model information
        model_info = fetch_model_info(selected_model)
        if model_info:
            st.header(model_info[1])  # Display model name
            st.caption(f':green[{model_info[2]}] *|* :green[{model_info[3]}] :computer:')  # display model types
            st.write(':blue[Description:]', model_info[4])  # Display model description
            st.write(':blue[Usage Scenarios:]', model_info[5])  # Display model usage scenarios
            st.write(':blue[Advantages:]', model_info[6])  # Display model advantages
            st.write(':blue[Limitations:]', model_info[7])  # Display model limitations
            st.write(':blue[Data Requirements:]', model_info[8])  # Display model limitations
            st.write(':blue[Implementation Code:]')  # Display model implementation code
            st.code(model_info[12])  # Display model implementation code

            st.write(":blue[Additional Resources]")
            st.write('''For a deeper dive and additional insights, we recommend exploring the official [Documentation]({}).
                            It offers comprehensive information on **{}**, including methods, implementation details,
                            and practical examples to enhance your understanding and application of the model.'''.format(
                model_info[11], model_info[1]))

            st.write(''':blue[Practical Example:]
                        Below is a practical example demonstrating the application of **{}**.
                        You can also go to Github and download this notebook. [Download Notebook From Github]({})'''.format(
                model_info[1], model_info[9]))

            render_example(model_info[10])  # display jupyter notebook case study

        else:
            st.error('Model information not found.')
