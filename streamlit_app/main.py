import streamlit as st
import os

model_image_path = "/Users/bhargavi/PycharmProjects/High_Performance_Parallel_Deep_Learning/streamlit_app/images/model.png"  # Path to the image of the model used

cpu_training_img = "/Users/bhargavi/PycharmProjects/High_Performance_Parallel_Deep_Learning/streamlit_app/images/cpu_parallel.png"
single_gpu_comparison_imp = "/Users/bhargavi/PycharmProjects/High_Performance_Parallel_Deep_Learning/streamlit_app/images/cpu_1_gpu.png"

ddp_image_path = "/Users/bhargavi/PycharmProjects/High_Performance_Parallel_Deep_Learning/streamlit_app/images/ddp.png"


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


def parallel_processing_gpu():
    st.write("""
    ## Parallel Processing on GPU
    - Transitioned from utilizing 28 CPUs to a single GPU over 20 epochs to enhance performance and efficiency in the training process.
    - Leveraged GPU architecture for accelerated computations, marking a significant shift from CPU-based multiprocessing.
    - Unlike the CPU approach, GPU training employed serial code without multiprocessing, streamlining the training process.
    - The training process involved iterating through training and validation datasets, with periodic logging of key metrics like loss and accuracy to monitor training progress across epochs.
    """)

    # Display additional details or images related to GPU performance comparison
    st.image(single_gpu_comparison_imp, caption='Transitioning to Single GPU-based training drastically reduced training time from 4.5 hours to just 30 minutes (~94% decrease in time)', use_column_width=True)

    # Include DDP content
    st.write("""
    ## Distributed Data Parallel (DDP)
    - Further enhanced performance by utilizing multiple GPUs in parallel for training deep learning models.
    - DDP enables synchronous distributed training across multiple GPUs, allowing for efficient scaling of model training.
    - Leveraged the PyTorch framework's DDP module to seamlessly distribute model parameters and gradients across GPUs.
    """)

    # Display image related to DDP
    st.image(ddp_image_path, caption='Distributed Data Parallel (DDP) in action', use_column_width=True)

def main():
    introduction()
    dataset_details()
    model_details()
    display_model_image_and_directory()
    parallel_processing_cpu()
    parallel_processing_gpu()

if __name__ == "__main__":
    main()

