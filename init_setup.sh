echo [$(date)]:"Start"
echo [$(date)]:"Creating environment with Python 3.8 version"
conda create --prefix ./env python=3.8 -y
echo [$(date)]:"activating the env"
source activate ./env
echo [$(date)]:"installing the dev requirements"
pip install -r requirements_dev.txt
echo [$(date)]:"End"