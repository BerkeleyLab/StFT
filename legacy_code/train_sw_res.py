import sys
import torch
from torch.nn.utils import clip_grad_norm_
import os
import pickle
# from harrm_spectrum2 import TemporalDataset, get_grid, LpLoss, HierARRM
from sw_2d import TemporalDataset, get_grid, LpLoss, HierARRM
from torch.utils.data import DataLoader
from einops import rearrange
import tempfile
from ray.train import Checkpoint
from ray import train, tune
from ray.train import RunConfig
import ray
# TUNE_MAX_PENDING_TRIALS_PG=50
def train_model(config):
    def unnorm_data(data,mean,std,B,C,H,W):
        data_copy = data.detach().clone()
        return (data_copy.reshape(B,C, H,W)[:,None,:,:,:])*std+mean
    dataset = config["dataset"]
    patch_sizes = config["patch_sizes"]
    overlaps = config["overlaps"]
    dim = config["dim"]
    vit_depth = config["vit_depth"]
    modes = config["modes"]
    mlp_dim = dim
    num_heads = config["num_heads"]
    snapshots = config["snapshots"]
    lr = config["lr"]
    max_epochs = config["max_epochs"]
    batchsize = config["batchsize"]
    cond_time = config["cond_time"]
    myloss = LpLoss(size_average=False)
    num_levels = len(patch_sizes)
    with open(dataset, "rb") as file:
        dataset = pickle.load(file)
    train_data = torch.tensor(dataset["train"],dtype=torch.float32,device="cuda")
    test = torch.tensor(dataset["test"],dtype=torch.float32,device="cuda")
    val = torch.tensor(dataset["val"],dtype=torch.float32,device="cuda")
    train_mean = train_data.mean(dim=(0, 1, 3, 4), keepdim=True)
    train_std = train_data.std(dim=(0, 1, 3, 4), keepdim=True)
    train_data = (train_data-train_mean)/train_std
    test = (test-train_mean)/train_std
    val = (val-train_mean)/train_std
    num_in_states = dataset["channels"]
    img_size = dataset["img_size"]
        
    train_loader = DataLoader(TemporalDataset(train_data,snapshot_length=snapshots),batch_size=batchsize,shuffle=True)

    in_channels = num_in_states*cond_time+2
    
    grid = get_grid(img_size[0],img_size[1]).cuda()
    out_channesl = num_in_states
    model = HierARRM(patch_sizes,overlaps,in_channels,out_channesl,img_size=img_size,dim=dim,vit_depth=vit_depth,modes=modes,num_heads=num_heads,mlp_dim=mlp_dim).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = torch.tensor(1e10,dtype=torch.float32,device="cuda")
    best_test = torch.tensor(1e10,dtype=torch.float32,device="cuda")
    best_test_under_val = torch.tensor(1e10,dtype=torch.float32,device="cuda")
    
    
    
    for ep in range(max_epochs):
        model.train()
        train_l2 = 0
        train_num_examples = 0
        for _, example in enumerate(train_loader): 
            B, L, C, H, W = example.shape
            for i in range(L - cond_time):
                train_num_examples += B*C
                x = example[:, i : (i + cond_time)].cuda()
                x = rearrange(x,"B L C H W -> B (L C) H W")
                y = example[:, i + cond_time].cuda()
                x = torch.cat((x, grid[None, :, :,:].repeat(B, 1, 1, 1)), axis=1)
                preds = model(x)
                sum_residues = torch.zeros_like(preds[0].reshape(B*num_in_states, -1),device="cuda",dtype=torch.float32)
                if True:
                    for level in range(num_levels):
                        cur_preds = preds[level]
                        sum_residues+=cur_preds.reshape(B*num_in_states, -1)
                    loss = myloss(sum_residues.reshape(B*num_in_states,-1), y.reshape(B*num_in_states,-1)) 
                    optimizer.zero_grad()
                    loss.backward(retain_graph=False)
                    clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step() 
                loss = myloss(sum_residues.reshape(B*num_in_states,-1), y.reshape(B*num_in_states,-1))  
                train_l2 += loss
        train_l2=train_l2/train_num_examples
        model.eval()
        if ep%50==0:
            num_examples = 0
            l2_val = 0.0
            with torch.no_grad():
                B, L, C, H, W = val.shape
                x_old = None
                preds_or = val[:, :cond_time]
                for i in range(L - cond_time):
                    num_examples+=B*num_in_states
                    if i == 0:
                        x = preds_or
                    else:
                        x = torch.cat(
                            (x_old[:, 1:, :, :, :], preds_or[:, None, :, :, :]), axis=1
                        )
                    x_old = x.detach().clone()
                    x = rearrange(x,"B L C H W -> B (L C) H W")
                    y = val[:, i + cond_time].cuda()
                    x = torch.cat((x, grid[None, :, :,:].repeat(B, 1, 1, 1)), axis=1)
                    preds = model(x)
                    sum_residues = torch.zeros_like(preds[0].reshape(B*num_in_states, -1),device="cuda",dtype=torch.float32)
                    for level in range(num_levels):
                        cur_preds = preds[level]
                        sum_residues+=cur_preds.reshape(B*num_in_states, -1).detach().clone()
                    l2_val += myloss(unnorm_data(sum_residues,train_mean,train_std,B,C,H,W).reshape(B*num_in_states, -1), unnorm_data(y,train_mean,train_std,B,C,H,W).reshape(B*num_in_states,-1))
                    preds_or = sum_residues.reshape(B,C,H,W)
            error_val=l2_val/num_examples
            num_examples = 0
            l2_test = 0.0
            with torch.no_grad():
                B, L, C, H, W = test.shape
                x_old = None
                preds_or = test[:, :cond_time]
                for i in range(L - cond_time):
                    num_examples+=B*num_in_states
                    if i == 0:
                        x = preds_or
                    else:
                        x = torch.cat(
                            (x_old[:, 1:, :, :, :], preds_or[:, None, :, :, :]), axis=1
                        )
                    x_old = x.detach().clone()
                    x = rearrange(x,"B L C H W -> B (L C) H W")
                    y = test[:, i + cond_time].cuda()
                    x = torch.cat((x, grid[None, :, :,:].repeat(B, 1, 1, 1)), axis=1)
                    preds = model(x)
                    sum_residues = torch.zeros_like(preds[0].reshape(B*num_in_states, -1),device="cuda",dtype=torch.float32)
                    for level in range(num_levels):
                        cur_preds = preds[level]
                        sum_residues+=cur_preds.reshape(B*num_in_states, -1).detach().clone()
                    l2_test += myloss(unnorm_data(sum_residues,train_mean,train_std,B,C,H,W).reshape(B*num_in_states, -1), unnorm_data(y,train_mean,train_std,B,C,H,W).reshape(B*num_in_states,-1))
                    preds_or = sum_residues.reshape(B,C,H,W)
            error_test=(l2_test/num_examples).clone()
            if error_val<best_val:
                best_val = error_val
                best_test_under_val = error_test       
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"model_state": model.state_dict()}, os.path.join(tempdir, "checkpoint_harrm.pt"))
                    metrics = {"epoch": ep, "train_l2": train_l2.item(),"best_val":best_val.item(),"best_test_under_val": best_test_under_val.item(),"best_test":best_test.item(),"test_error":error_test.item(),"val_error":error_val.item()}
                    train.report(metrics=metrics,checkpoint=Checkpoint.from_directory(tempdir))
            else:  
                metrics = {"epoch": ep, "train_l2": train_l2.item(),"best_val":best_val.item(),"best_test_under_val": best_test_under_val.item(),"best_test":best_test.item(),"test_error":error_test.item(),"val_error":error_val.item()}
                train.report(metrics=metrics)

            if error_test<best_test:
                best_test = error_test

            
            
                    
if __name__ == "__main__":

    patch_sizes =  ((128,128),(64,64),(32,32))

    overlaps =    ((1,1),(1,1),(1,1))

    dim = 512

    vit_depth = 6
    config={
        "dataset":  "/global/cfs/cdirs/m2956/zbai/StFT/Reruns_sw/sw_res.pkl",
        "patch_sizes": patch_sizes,
        "overlaps":  overlaps,
        "dim": dim,
        "vit_depth": vit_depth,
        "modes": tune.grid_search([8]),
        "num_heads": 1,
        "snapshots": 20,
        "lr": 1e-4,
        "max_epochs": 1000000,
        "batchsize": 20,
        "cond_time": 5,
    }  

    save_path = "/global/cfs/cdirs/m2956/zbai/StFT/Reruns_sw_res"
    trainable_with_cpu_gpu = tune.with_resources(train_model, {"cpu": 32, "gpu": 1})
    tuner = tune.Tuner(trainable_with_cpu_gpu,
            param_space=config,
            run_config=RunConfig(storage_path=save_path, name="re-sw-res"),
        )
    tuner.fit()
    
