preprocess_data=true
install_gdal=false

options=$(getopt -o : --long skip_preprocessing,skip_gdal_install -- "$@")
eval set -- "$options"
while true; do
    case "$1" in
	--skip_preprocessing)
	    echo "Skipping preprocessing stage!"
	    preprocess_data=false
	    shift
	    ;;
	--install_gdal)
	    echo "Will attempt installing GDAL!"
	    install_gdal=true
	    shift
	    ;;
	--)
	    shift
	    break
	    ;;
	*)
	    echo "Invalid argument"
	    exit 3
	    ;;
    esac
done

if [ "$install_gdal" = true ]
then
    echo "Installing GDAL"
    sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
    sudo apt-get update
    sudo apt-get install gdal-bin
    sudo apt-get install libgdal-dev
    export CPLUS_INCLUDE_PATH=/usr/include/gdal
    export C_INCLUDE_PATH=/usr/include/gdal
    pip install --user pygdal==2.1.2.3
fi

if [ "$preprocess_data" = true ]
then
    echo "Running preprocessing stage, could take a while..."
    if [ ! -d "processed_data" ]
    then
	echo "Creating processed_data folder"
	mkdir processed_data
    fi
    
    if [ ! -d "es262-airquality/GHCND_weather/ghcnd_hcn" ]
    then
	echo "Untarring ghcnd_hcn.tar.gz into ghcnd_hcn..."
	cd es262-airquality/GHCND_weather/ && tar -xf ghcnd_hcn.tar.gz && cd
    fi

    if [ ! -d "es262-airquality/GHCND_weather/relevant_ghc" ]
    then
	echo "Creating relevant_ghc folder"
	mkdir es262-airquality/GHCND_weather/relevant_ghc
    fi

    python code/data_processing.py --tif_to_npy
fi

if [ ! -d "predictions" ]
then
echo "Creating predictions folder"
mkdir predictions
fi

if [ ! -d "plots" ]
then
echo "Creating plots folder"
mkdir plots
fi

if [ ! -d "checkpoints" ]
then
echo "Creating checkpoints folder"
mkdir checkpoints
fi

echo "Begin training phase..."
# TRAINING CODE HERE   

echo "Training Nearest Neighbors baseline"
python code/models/knn_baseline.py

echo "Training Non-Sentinel Net"
python code/models/overfitting_nonsentinel_net.py

echo "Training Sentinel-2 Net"
python code/models/cnn.py

echo "Training Frozen Multi-Modal Net"


echo "Training Finetuned Multi-Modal Net"


echo "Training End-to-end Multi-Modal Net"
python code/models/cnn_combined.py


