from setuptools import setup, find_packages

setup(
    name='YourAppName',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit==1.36.0',
        'pandas==1.3.5',
        'matplotlib==3.8.4',
        'seaborn==0.13.2',
        'plotly==5.22.0',
        'numpy==1.26.4',
        'wordcloud==1.9.3',
        'ydata-profiling==4.8.3',
        'scikit-learn==1.5.0',
        'folium==0.17.0',
        'streamlit-folium==0.16.0'
    ],
    python_requires='>=3.7,<3.11',  # Ensure this matches your deployment environment
)
