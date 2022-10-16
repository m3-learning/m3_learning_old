# building the wheel
python setup.py sdist

# install from local build
pip install .

# Build Jupyter Notebook
conda run -n m3_learning jupyter-book build "C:\Users\Joshua Agar\Documents\codes\m3_learning\m3_learning"
