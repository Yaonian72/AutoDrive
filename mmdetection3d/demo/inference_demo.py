from mmdet3d.apis import init_detector, inference_detector, show_result_meshlab

#%%

config_file = '../configs/3dssd/3dssd_kitti-3d-car.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../checkpoints/3dssd_kitti-3d-car/latest.pth'

#%%

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

#%%

# test a single sample
pcd = 'kitti_000008.bin'
result, data = inference_detector(model, pcd)

print(model)