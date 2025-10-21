from habitat.config.default import get_config, patch_config
from habitat.config import read_write
from habitat import Env


if __name__ == "__main__":

    cfg = get_config("habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
    cfg = patch_config(cfg)

    with read_write(cfg):

        cfg.habitat.task.type = "Personalized_ObjectNav-v1"
        cfg.habitat.dataset.type = "Personalized-ObjectNav-v1"
        cfg.habitat.dataset.data_path = "habitat-lab/data/datasets/PersONAL/active/val/easy/easy.json.gz"
        cfg.habitat.dataset.scenes_dir = "habitat-lab/data/scene_datasets"

        cfg.habitat.simulator.scene_dataset = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

    env = Env(config=cfg)
    obs = env.reset()
    print("\nPersONAL setup successful!\n")
    env.close()
