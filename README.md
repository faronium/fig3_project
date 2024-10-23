## Environment set up

The code for answering question 3 in the written assignment was written in python. It uses a virtual environment to control packages and package versions to make sure that running is consistent across platforms. This readme walks through the steps to set up the environment and run the code.

### Make a virtual environment

First, a virtual environment has to be made. This is done in the project root directory with

```bash
python -m venv env
```

This will create a directory called env in the project folder

Enter the environment with

```bash
source env/bin/activate
```

Upgrade pip

```bash
pip install --upgrade pip
```

Install the python packages using the package list in the file requirements.txt

```bash
pip install -r requirements.txt
```

Make a couple of directories that are needed for data that will be downloaded and for the figure output

```bash
mkdir -p figures
mkdir -p data/EC_autodownload/Hourly_CSVs/2023
```

Finally, change directory to the code directory. The script should run, with:

```bash
python quinone.py
```

but will need the shapefile with the North Vancouver waterways and the environmental sampling data. These are at

```bash
data/EMSWR_Rus_050802.csv
gis/FWA_STREAM_NETWORKS_SP/FWSTRMNTWR_line.shp
```


