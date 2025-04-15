pip install https://download.pytorch.org/whl/cu126/torch-2.6.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=c55280b4da58e565d8a25e0e844dc27d0c96aaada7b90b4de70a45397faf604e
pip install torch_geometric
pip install sklearn==0.0
pip install pykeen==1.6.0
pip install .
pip install googledrivedownloader==0.2
# comment out "showsize=True" in files using `GoogleDriveDownloader` packages
if [ -d "~/miniconda3/envs/hyper_relational_ilp/lib/python3.10/site-packages/ilp" ]; then
    rm -r ~/miniconda3/envs/hyper_relational_ilp/lib/python3.10/site-packages/ilp
fi 