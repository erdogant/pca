echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf pca.egg-info

echo "Making new build.."
echo ""
python setup.py bdist_wheel
echo ""
echo 
python setup.py sdist
echo ""
pip install -U dist/pca-0.1.3-py3-none-any.whl
echo ""
read -p "[PYPI] [python -m twine upload dist/*]"
echo ""
read -p "Press [Enter] key to close window..."
