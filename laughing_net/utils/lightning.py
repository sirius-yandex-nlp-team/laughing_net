from typing import Dict, Any, Optional

import dvclive
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

class DVCLiveCompatibleModelCheckpoint(ModelCheckpoint):
   def _get_metric_interpolated_filepath_name(
        self,
        monitor_candidates: Dict[str, Any],
        epoch: int,
        step: int,
        trainer,
        del_filepath: Optional[str] = None,
    ) -> str:
        filepath = self.format_checkpoint_name(epoch, step, monitor_candidates)
        return filepath

class DVCLiveNextStepCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        dvclive.next_step()
