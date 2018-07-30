sudo pip3.6 install numpy pandas sklearn scipy
sudo pip3.6 install git+https://github.com/drivendataorg/drivendata-submission-validator.git

## Install TPOT
sudo pip3.6 install deap update_checker tqdm stopit
sudo pip3.6 install scikit-mdr skrebate

git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd ..

sudo pip3.6 install --upgrade --force-reinstall tpot==0.8
#sudo pip3.6 install --upgrade --force-reinstall git+https://github.com/EpistasisLab/tpot/tree/v0.8

