import os
from pathlib import Path

import numpy as np
import torch
from pkbar import pkbar
from torch import nn, optim
from torch.utils.data import DataLoader

from .model_utils import predict


def train(
	model,
	epochs,
	learning_rate,
	batch_size,
	train_data,
	test_data,
	optimizer,
	num_workers=None,
	seed=None,
	device=None,
	save_dir=None,
	ckpt_name=None,
	save_ckpts=True,
	start_ckpt_number=None,
	measure_time = False,
	save_every=1,
	forward_fn=None,
):

	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"

	print(f"Using device: {device}")

	if save_ckpts:
		if save_dir is None:
			raise ValueError("save_dir can not be None if save_ckpts is True")

	ckpt_number = 0 if start_ckpt_number is None else start_ckpt_number

	if num_workers is None:
		num_workers = os.cpu_count() - 1

	if seed is not None:
		torch.manual_seed(seed)

	train_loader = DataLoader(
		train_data,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
	)

	criterion = nn.CrossEntropyLoss()

	model = model.to(device)

	for epoch in range(0, epochs):
		kbar = pkbar.Kbar(
			target=max(len(train_loader) - 1, 1),
			epoch=epoch,
			num_epochs=epochs,
			always_stateful=True,
		)
		model.train()
		correct = 0
		total = 0
		running_loss = 0.0
		
		forward_fn = model if forward_fn is None else forward_fn
		
		# iterates over a batch of training data_ops
		for batch_idx, (inputs, targets) in enumerate(train_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			optimizer.zero_grad()
			outputs = forward_fn(inputs)
			loss = criterion(outputs, targets)
			loss.backward()

			optimizer.step()
			_, predicted = outputs.max(1)

			# calculate the current running loss as well as the total accuracy
			# and update the progressbar accordingly
			running_loss += loss.item()
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

			kbar.update(
				batch_idx,
				values=[
					("loss", running_loss / (batch_idx + 1)),
					("acc", 100.0 * correct / total),
				],
			)

		# save the model in each epoch
		if save_ckpts:

			Path(save_dir).mkdir(exist_ok=True, parents=True)

			checkpoint_name = "-".join(
				["checkpoint", str(epoch + 1 + ckpt_number) + ".pt"]
			)

			if ckpt_name != None:
				checkpoint_name = ckpt_name

			if epoch % save_every == 0 or epoch == epochs - 1:
				torch.save(
					{
						"epoch": epoch,
						"model_state_dict": model.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"loss": running_loss,
						"learning_rate": learning_rate,
					},
					os.path.join(save_dir, checkpoint_name),
				)

	# calculate the train accuracy of the network at the end of the training
	train_preds_labels, train_preds, train_acc = np.array([]), np.array([]), -1
	if not measure_time:
		train_preds_labels, train_preds, train_acc = predict(
			model, train_data, batch_size=batch_size, num_workers=num_workers, device=device, forward_fn=forward_fn
		)

	# calculate the test accuracy of the network at the end of the training
	test_preds_labels, test_preds, test_acc = np.array([]), np.array([]), -1
	if not measure_time:
		test_preds_labels, test_preds, test_acc = predict(
			model,
			test_data,
			batch_size=batch_size,
			num_workers=num_workers,
			device=device,
			forward_fn=forward_fn
		)

	print(
		"Final accuracy: Train: {} | Test: {}".format(
			100.0 * train_acc, 100.0 * test_acc
		)
	)

	info_dict = {
		"train_acc": float(train_acc),
		"test_acc": float(test_acc),
		"train_loss": float(running_loss),
		"test_preds": test_preds.tolist(),
		"test_preds_labels": test_preds_labels.tolist(),
		"train_preds": train_preds.tolist(),
		"train_preds_labels": train_preds_labels.tolist(),
	}

	return model, info_dict
