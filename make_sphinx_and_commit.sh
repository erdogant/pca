echo "$PWD"
cd docs
echo "$PWD"

echo "Cleaning previous builds first.."
make.bat clean

echo "Building new html.."
make.bat html

read -p "Press [Enter] key to: git add -> commit -> push."

cd ..

git add .
git commit -m "update sphinx pages"
git push

read -p "All done! Press [Enter] to close this window."
