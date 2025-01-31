import torch
from pytorch_lightning.callbacks import Callback

ERA5_GLOBAL_RANGES = {
    # Surface Variables
    't2m': {'min': 170, 'max': 350},  # 2m Temperature [K]
    'msl': {'min': 85000, 'max': 110000},  # Mean Sea Level Pressure [Pa]
    'u10': {'min': -110, 'max': 110},  # 10m U-wind [m/s]
    'v10': {'min': -80, 'max': 80},  # 10m V-wind [m/s]
    
    # Atmospheric Variables (at all pressure levels)
    't': {'min': 160, 'max': 350},  # Temperature [K]
    'q': {'min': 0, 'max': 0.04},  # Specific Humidity [kg/kg]
    'u': {'min': -110, 'max': 110},  # U-wind [m/s]
    'v': {'min': -100, 'max': 100},  # V-wind [m/s]
    'z': {'min': -5000, 'max': 225000},  # Geopotential Height [m]
}

GFS_GLOBAL_RANGES = {
    # Atmospheric Variables (at all pressure levels)
    't': {'min': 160, 'max': 350},  # Temperature [K]
    'q': {'min': 0, 'max': 0.04},  # Specific Humidity [kg/kg]
    'u': {'min': -110, 'max': 110},  # U-wind [m/s]
    'v': {'min': -100, 'max': 100},  # V-wind [m/s]
    'z': {'min': -5000, 'max': 225000},  # Geopotential Height [m]

    # Surface Variables
    'lsm': {'min': 0, 'max': 1},  # Land Sea Mask
    'mslet': {'min': 85000, 'max': 110000},  # Mean Sea Level Pressure,
    'slt': {'min': 0, 'max': 16}, # Soil Type
    'orog': {'min': -500, 'max': 6000} # Orography (Elevation)
}


class UpdateMonitorCallback(Callback):
    def __init__(self, log_every_n_steps=25):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
        self.param_history = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        if self.step_count % self.log_every_n_steps == 0:
            total_update_magnitude = 0.0
            num_params = len(list(pl_module.parameters()))
            for param_name, param in pl_module.named_parameters():
                    if param.requires_grad:
                        if param_name not in self.param_history:
                            self.param_history[param_name] = param.data.clone()
                        else:
                            update = param.data - self.param_history[param_name]
                            update_magnitude  = update.data.norm(2).item()

                            total_update_magnitude += update_magnitude

                            self.param_history[param_name] = param.data.clone()

            total_update_magnitude /= num_params
            trainer.logger.log_metrics({
                f"update_magnitude": total_update_magnitude,
            }, step=trainer.global_step)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_every_n_steps == 0:
            total_norm = 0.0
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2 # Sum of Sqaures
            total_norm = total_norm ** 0.5 # L2 Norm
            
            trainer.logger.log_metrics({"gradient_norm": total_norm}, step=trainer.global_step)


    def on_train_epoch_end(self, trainer, pl_module):
        self.step_count = 0
        self.param_history.clear()

def print_debug(debug=True, *args):
    if debug:
        msg = " ".join(map(str, args))
        print(msg)

def check_gpu_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = mem - allocated
        gb = free * 1e-9
        return gb
    else:
        return None