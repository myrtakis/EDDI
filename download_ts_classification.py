from pathlib import Path

from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate as ucr_data_names

import numpy as np
from utils.data import normalize_labels
import json

def save_data(X, y, savedir):
	savedir.mkdir(exist_ok=True, parents=True)
	with open(Path(savedir, 'X.npy'), "wb") as f:
		np.save(f, X)
	with open(Path(savedir, 'y.npy'), "wb") as f:
		np.save(f, y)


if __name__ == '__main__':

	with open('dataset_list.json', 'r') as f:
		data_names = json.load(f)

	print(f'Downloading the {len(data_names)} time series classification datasets')
	for dname in ucr_data_names:
		if dname not in data_names:
			continue
		savedir = Path('datasets', dname)
		if Path(savedir, 'X.npy').exists() and Path(savedir, 'y.npy').exists():
			continue
		print(f'Loading/Saving {dname} in {savedir}')
		X, y = load_classification(dname)
		y = normalize_labels(y)
		save_data(X, y, savedir=savedir)
