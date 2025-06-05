import json
import time

import pandas as pd

import numpy as np
from models.train_loop import train
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import TensorDataset
import torch
from collections import OrderedDict
from pathlib import Path
import shutil
import click
import warnings

from utils.data import data_splitting

from models.dispatcher import model_dispatcher, forward_fn_transform

def train_model(sub_dfolders, seed, model_conf_dict, train_size=0.7, device='cpu'):
	
	for dfolder in sub_dfolders:

		dname = dfolder.name

		local_savedir = Path('results', dname, model_conf_dict['model_name'], str(seed), 'clean', 'ckpts')

		with open(Path(dfolder, 'X.npy'), 'rb') as f:
			X = np.load(f)

		with open(Path(dfolder, 'y.npy'), 'rb') as f:
			y = np.load(f)

		X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(X, y, train_size=train_size, seed=seed)

		y_train = y_train.astype(int)
		y_val = y_val.astype(int)
		y_test = y_test.astype(int)
		
		print(X_train.shape)

		unq_labels = np.unique(y_train)
		num_labels = len(unq_labels)
		num_samples = len(y_train) + len(y_test)
		num_features = X_train.shape[1]
		time_steps_per_sample = X_train.shape[-1]

		print(dname,
			  f'Shape {X_train.shape} #Samples {num_samples}, #Features {num_features}, #Classes {num_labels}, #Timesteps {time_steps_per_sample}')
		label_dict = OrderedDict(zip(unq_labels, np.arange(num_labels)))

		y_train = pd.Series(y_train).replace(label_dict).values
		y_test = pd.Series(y_test).replace(label_dict).values

		if num_labels != len(np.unique(y_test)):
			print(f'Skipping {dname} because number of train labels is not equal to number of test labels')
			continue

		model = model_dispatcher[model_conf_dict['model_name']](
			timesteps=time_steps_per_sample,
			dims=num_features,
			num_classes=num_labels
		)

		batch_size = model_conf_dict['batch_size']
		lr = model_conf_dict['learning_rate']
		epochs = model_conf_dict['epochs']

		train_data = TensorDataset(torch.Tensor(X_train), torch.tensor(y_train))
		test_data = TensorDataset(torch.Tensor(X_test), torch.tensor(y_test))

		if num_samples < batch_size:
			local_batch_size = int(num_samples / 3)
		else:
			local_batch_size = batch_size

		optimizer = torch.optim.Adam(model.parameters(), lr=lr)

		forward_fn = forward_fn_transform[model_conf_dict['model_name']](model)

		model, info_dict = train(
			model=model,
			epochs=epochs,
			learning_rate=lr,
			batch_size=local_batch_size,
			train_data=train_data,
			test_data=test_data,
			optimizer=optimizer,
			device=device,
			save_ckpts=True,
			seed=seed,
			save_every=epochs,
			save_dir=local_savedir,
			forward_fn=forward_fn
		)

		if np.isnan(info_dict['train_loss']):
			warnings.warn(f'{dname} has NaN loss values during training and has been skipped.')
			shutil.rmtree(Path(local_savedir))
			continue

		bal_acc = balanced_accuracy_score(y_test, info_dict['test_preds_labels'])

		forward_fn = model if forward_fn is None else forward_fn

		if num_labels == 2:
			auc = roc_auc_score(y_test, info_dict['test_preds'])
		else:
			test_preds_unreduced = torch.softmax(forward_fn(test_data.tensors[0]), dim=1).detach().numpy()
			auc = roc_auc_score(y_test, test_preds_unreduced, multi_class='ovr')

		col_names = ['data_ops', '#samples', '#features', '#classes', '#timesteps', 'model', 'auc', 'bal_acc']
		vals = [dname, num_samples, num_features, num_labels, time_steps_per_sample, model_conf_dict['model_name'], auc, bal_acc]
		pd.DataFrame([vals], columns=col_names).to_csv(Path(local_savedir, 'info.csv'), index=False)

@click.command()
@click.option("--model_config", type=click.Path(exists=True), default=Path('configs','models','tst.json'))
@click.option("--seed", type=click.INT, default=0)
@click.option("--train_size", type=click.FloatRange(0.1,0.9), default=0.7)
@click.option("--run_only_on", type=click.Path(exists=True), default=None)
@click.option("--device", type=click.STRING, default='cpu')
def run_training(
	model_config,
	seed,
	train_size,
	run_only_on,
	device
):
	start = time.time()
	datasets = [Path(run_only_on)] if run_only_on is not None else list(Path('datasets').iterdir())

	with open(model_config) as f:
		model_conf_dict = json.load(f)

	train_model(datasets, seed=seed, model_conf_dict=model_conf_dict, train_size=train_size, device=device)
 
	print('TOTAL TRAINING TIME', time.time() - start)


if __name__ == '__main__':
	print('Starting training')
	run_training()
