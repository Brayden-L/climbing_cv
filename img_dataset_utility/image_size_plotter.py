# %%
import os
from PIL import Image
import pandas as pd
import plotly.express as px
from statistics import mode

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def process_folder(folder_path):
    dimensions_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(root, file)
                width, height = get_image_dimensions(file_path)
                scalar_value = width * height
                dimensions_list.append({'Scalar_Value': scalar_value})
    return dimensions_list

def count_samples_within_percentage(df, mode_scalar, percentage):
    lower_bound = mode_scalar - (mode_scalar * percentage / 100)
    upper_bound = mode_scalar + (mode_scalar * percentage / 100)
    count_within_percentage = df[
        (df['Scalar_Value'] >= lower_bound) & (df['Scalar_Value'] <= upper_bound)
    ].shape[0]
    return count_within_percentage

def create_histogram(dimensions_list, nbins=10000):  # Specify the number of bins
    df = pd.DataFrame(dimensions_list)
    
    # Find and print the mode
    mode_scalar = mode(df['Scalar_Value'])
    print(f"Mode of Scalar Values: {mode_scalar}")
    
    # Count samples with the mode as their value
    count_mode_samples = count_samples_within_percentage(df, mode_scalar, 0)
    percent_mode_samples = (count_mode_samples / len(df)) * 100
    print(f"Percentage of Samples with Mode as their Value: {percent_mode_samples:.2f}%")
    
    # Count samples within ±10% of the mode
    count_within_10_percent = count_samples_within_percentage(df, mode_scalar, 10)
    percent_within_10_percent = (count_within_10_percent / len(df)) * 100
    print(f"Percentage of Samples within ±10% of the Mode: {percent_within_10_percent:.2f}%")
    
    # Create histogram
    fig = px.histogram(df, x='Scalar_Value', marginal='box', title='Scalar Value Histogram', nbins=nbins)
    fig.update_layout(title='Image Width*Height Histogram', title_x=0.5, xaxis_title='width*height')
    fig.show()


folder_path = r"C:\Users\Brayden\Desktop\climbing_photo_rear_glory_topo_classifier\climbing_classifier_data"
dimensions_list = process_folder(folder_path)
create_histogram(dimensions_list)

# %%
