from setuptools import setup, find_packages

setup(
    name='RemoteSensingTools',  # This will be the name used in pip install
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "planetary-computer",
        "pystac-client",
        "rioxarray",
        "geopandas",
        "matplotlib",
        "xarray",
        "shapely",
        "numpy",
        "dask",
        "rasterio",
        "ipython",
        "rich",
        "odc-stac",
        "pyproj",
        "folium",
        "stackstac"
    ],  # Add dependencies here
    author='Charlie A Ikosi',
    author_email='charlieikosi@gmail.com',
    description='Functions for working with imagery on Planetary Computer',
    url='https://github.com/charlieikosi/RemoteSensingTools',
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
