
target_folder=${1-~/test}
test_name=${2-test}

echo
echo
echo test_name: $test_name
echo target_folder: $target_folder
echo
echo

mkdir $target_folder
scp ./* $target_folder
python main.py

echo
echo
echo test_name: $test_name
echo target_folder: $target_folder
echo
echo