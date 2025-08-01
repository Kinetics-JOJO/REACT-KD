from config import config
from trainer import train_and_evaluate, validate_only
from topo_gradcam_visualization import run_topo_gradcam

if __name__ == "__main__":
    print(f"Stage: {config.stage}")

    if config.stage == "student" or config.stage == "teacher":
        train_and_evaluate(config)
    elif config.stage == "val_only":
        validate_only(config)
    elif config.stage == "topo_cam":
        run_topo_gradcam(config)
    else:
        raise ValueError(f"Unsupported stage: {config.stage}")
