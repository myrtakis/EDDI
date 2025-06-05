from tsai.models.TST import TST

import torch

def setup_tst(dims: int, timesteps: int, num_classes: int, **kwargs):
	return TST(
		c_in=dims,
		seq_len=timesteps,
		c_out=num_classes,
		max_seq_len=256,
		d_model=128,
	)

def setup_moment(dims: int, timesteps: int, num_classes: int, **kwargs):
	from momentfm import MOMENTPipeline
	model = MOMENTPipeline.from_pretrained(
		"AutonLab/MOMENT-1-base",
		model_kwargs={
			"task_name": "classification",
			"n_channels": dims,
			"num_class": num_classes
		},
	)
	model.init()
	return model

model_dispatcher = {
	'TST': setup_tst,
	'MOMENT': setup_moment
}

def forward_tst(model):
	pass

def forward_moment(model):
	model.forward = lambda x: model.classify(x, None).logits


forward_fn_transform = {
	'TST': forward_tst,
	'MOMENT': forward_moment,
}

def emb_tst(model, X):
	embeddings = []
	def hook_fn(module, input, output):
		embeddings.append(output.detach())
	hook = model.encoder.register_forward_hook(hook_fn)
	model(torch.Tensor(X))
	hook.remove()
	return torch.cat(embeddings, dim=0).view(X.shape[0], -1)

def emb_moment(model, X):
	return model.classify(torch.Tensor(X), None).embeddings.view(X.shape[0], -1)


model_emb_dispatcher = {
	'TST': emb_tst,
	'MOMENT': emb_moment
}

def tst_final_layer(model):
	return model.head[-1]

def moment_final_layer(model):
	return model.head.linear


model_final_layer = {
	'TST': tst_final_layer,
	'MOMENT': moment_final_layer
}
