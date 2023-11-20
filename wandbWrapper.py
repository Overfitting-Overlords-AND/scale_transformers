import constants
import wandb

def init():
  # start a new wandb run to track this script
  if constants.WANDB_ON:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Scaling Transformer WPP",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": constants.LEARNING_RATE,
        "dimensions": constants.DIMENSIONS,
        "dataset": constants.DATASET,
        "vocab_size": constants.VOCAB_SIZE,
        "epochs": constants.NUM_OF_EPOCHS,
        "num_heads" : constants.NUM_HEADS,
        "num_layers" : constants.NUM_LAYERS,
        "d_ff" : constants.D_FF,
        "max_seq_length" : constants.MAX_SEQ_LENGTH,
        "dropout" : constants.DROPOUT
        }
    )

def log(metrics):
  if constants.WANDB_ON:
    wandb.log(metrics)

def finish():
  if constants.WANDB_ON:
    wandb.finish()