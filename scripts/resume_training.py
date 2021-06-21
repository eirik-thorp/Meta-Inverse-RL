from garage.trainer import Trainer
from garage.experiment.deterministic import set_seed
from garage import wrap_experiment

@wrap_experiment
def resume_push_training(ctxt=None, 
                         snapshot_dir='data/local/experiment/pearl_metaworld_ml1_push_3/',
                         seed=1):
          
    trainer = Trainer(snapshot_config=ctxt)
    trainer.restore(snapshot_dir)
    trainer.resume()

resume_push_training()
