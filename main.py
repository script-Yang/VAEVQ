import importlib
from torch.utils.data import Dataset

import lightning as L
from lightning.pytorch.cli import LightningCLI

import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.deterministic = True #True
torch.backends.cudnn.benchmark = False #False
from taming.data.ours import load_data


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(L.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def _train_dataloader(self):
        mypath = '/data02/imagenet/train'
        # mypath ='/data02/imagenet/val'
        dataloader = load_data(dataset_path=mypath,image_size=256,batch_size=64,state=True,num_workers=self.num_workers, pin_memory=True)
        return dataloader
    def _val_dataloader(self):
        mypath ='/data02/imagenet/val'
        dataloader = load_data(dataset_path=mypath,image_size=256,batch_size=64,state=False,num_workers=self.num_workers, pin_memory=True)
        return dataloader
    def _test_dataloader(self):
        mypath = '/data02/imagenet/val'
        dataloader = load_data(dataset_path=mypath,image_size=256,batch_size=64,state=False,num_workers=self.num_workers, pin_memory=True)
        return dataloader
   
def main():
    cli = LightningCLI(
        save_config_kwargs={"overwrite": True},
    )
    
if __name__ == "__main__":
    main()
