import streamlit as st
import os

model_image_path = "streamlit_app/images/model.png"  # Path to the image of the model used

cpu_training_img = "streamlit_app/images/cpu_parallel.png"


def introduction():
    st.title('Leveraging Parallel Processing for AI Image Detection')
    st.write("""
    ## Introduction
    This Streamlit app demonstrates the use of an AutoRegressive Neural Network (ARNN) for forecasting residuals in a time series dataset. 
    The ARNN model is trained on exogenous variables and predicts the residuals obtained from a Vector Autoregression (VAR) model.
    """)
    st.write("""
    ## Aim
    Enhance image classification model training through deep learning and parallelization strategies.
    Utilize Multiprocessing, Data Parallelism (DDP), and Automatic Mixed Precision (AMP) to optimize performance throughput.
    Harness the computational power of Discovery clusters' GPUs and CPUs for efficient scaling.
    """)

def display_model_image_and_directory():
    # Display the image of the model used
    st.image(model_image_path, caption='Model Used', use_column_width=True)

def dataset_details():
    st.write("""
    ## Dataset Details
    - **Dataset Name:** CASIA Dataset
    - **Total Images:** 12,610
    - **Size:** Approximately 3.5 GB
    - **Image Size Range:** 30KB - 500KB
    """)

def model_details():
    st.write("""
    ## Model Details
    Combining the **EfficientNetB3** backbone with a custom classifier, our model effectively captures hierarchical features and adapts flexibly to diverse classification tasks.
    """)


def parallel_processing_cpu():
    st.write("""
    ## Parallel Processing on CPU
    - Implemented multiprocessing on the CPU to expedite training on our extensive dataset, totaling nearly 3.5 GB.
    - Utilized multiprocessing to parallelize data loading and leverage the computational power of multiple CPU cores.
    - Trained the model using 'ThreadPoolExecutor' for comparison.
    - Efficiently distributed tasks across multiple CPU cores by dividing the dataset into training and validation sets and employing the `apply_async` method.

    ### Training Time Comparison
    A comparison of training times across varying CPU configurations, from 4 to 28 CPUs, showcased a notable reduction in training duration from 8 hours to 4 hours effectively.
    """)

    # Display the image of the training time comparison
    st.image(cpu_training_img, caption='Training Time Comparison', use_column_width=True)


def main():
    introduction()
    dataset_details()
    model_details()
    display_model_image_and_directory()
    parallel_processing_cpu()



if __name__ == "__main__":
    main()
