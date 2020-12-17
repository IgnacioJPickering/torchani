import torch
import torchani
from torchani.training import save_histogram_energies, save_histogram_foscs
import yaml
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yaml_file = '/media/samsung1TBssd/hpg_training_outputs_final/anihv_fosc_mtl_warm/anihv_fosc_mtl_warm.yaml'
model_file = '/media/samsung1TBssd/hpg_training_outputs_final/anihv_fosc_mtl_warm/best.pt'

with open(yaml_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
model = torchani.modules.TemplateModel.from_yaml(config).to(device).shift_before_output_(False)
model.load_state_dict(torch.load(model_file)['model'])
print(sum(p.numel() for p in torchani.models.ANI1x(model_index=0).parameters()))
exit()
with open('/home/ignacio/Datasets/all_excited_2.pkl', 'rb') as f:
    ds = pickle.load(f)
    validation = ds['validation']



#save_histogram_energies(model, validation, ex=0)
for j in range(1, 11):
    save_histogram_foscs(model, validation, ex=j)
    save_histogram_energies(model, validation, ex=j)

