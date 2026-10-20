import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import time
from pandas.io.sas.sas_constants import subheader_count_length
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from utils import calculate_model_flops
from EDDA import noise_injection_augmentation, channel_wise_scaling, scale_one_channel_bands_energy, scale_all_channels_bands_energy, random_global_scaling
import EDDA
from model import RuLCNN
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

torch.cuda.empty_cache()
torch.manual_seed(0)

def datanorm(x):
    for i in range(np.shape(x)[0]):
        x[i] = (x[i] - np.min(x[i])) / (np.max(x[i]) - np.min(x[i]))
    return x


FS = 128  # sample frequency

def augment_one_eeg_1C30T(x_3d, fs=128):
    x_np = x_3d.squeeze(0).detach().cpu().numpy().astype(np.float32)

    # 1. noise injection
    # x_np = noise_injection_augmentation(x_np, noise_level=0.02)
    # x_np = random_global_scaling(x_np, scale_range=(0.8, 1.2))
    # x_np = EDDA.bandwise_channel_scaling(x_np, fs=fs, scale_range=(0.8,1.2))

    # x_np = x_np.astype(np.float32) 

    # 2.1 channel-wise scaling
    # x_np = channel_wise_scaling(x_np, scale_range=(0.8, 1.2))
    
    # x_np = x_np.astype(np.float32) 


    # 2.2 band-wise scaling (only c1, beta/alpha/theta)
    # x_np = scale_one_channel_bands_energy(
    #     x_np,
    #     fs=fs,
    #     target_channel=28,     # OZ 通道
    #     scale_range=(0.9,1.1),
    #     k=0.3
    # )

    # 2.2 band-wise scaling (c30, beta/alpha/theta)
    # x_np = scale_all_channels_bands_energy(
    # x_np,
    # fs=fs,
    # scale_range=(0.8,1.2),
    # k=0.3
    # )

    return torch.from_numpy(x_np).unsqueeze(0)



def augment_batch_B1C30T(inputs, fs=128, device="cuda"):
    # print("EDDA....")
    """
    inputs: torch.Tensor (B,1,30,384)
    返回: 增强后的 Tensor (B,1,30,384) 放回 device
    """
    B = inputs.shape[0]
    inputs_cpu = inputs.detach().cpu()

    out = []
    for i in range(B):
        x_3d = inputs_cpu[i]      # (1,30,384)
        x_aug = augment_one_eeg_1C30T(x_3d, fs=fs)  # (1,30,384)
        # x_aug = x_3d
        out.append(x_aug)

    batch_aug = torch.stack(out, dim=0)  # (B,1,30,384)
    return batch_aug.to(device)




def run():
    # ========= 基础准备 =========
    predict_total = []
    label_total = []

    subject = 4
    data_path = './EEGdataset/dataset.mat'  # './EEGdataset/dataset.mat' or './EEGdataset/subject_0{}.mat'.format(subject) 
    print("Loading dataset from {}...".format(data_path))

    data_T = sio.loadmat(data_path)

    X_raw = data_T['EEGsample']              # (2022, 30, 384)
    Y_raw = data_T['substate'].squeeze()     # (2022,)

    print("Raw Lable Distribution:", dict(zip(*np.unique(Y_raw, return_counts=True))))

    # 数据添加 channel 维度 → (N, 1, 30, 384)
    X_raw = np.expand_dims(X_raw, axis=1)

    # 归一化
    X_raw = datanorm(X_raw).astype(np.float32)
    Y_raw = Y_raw.astype(np.int64)

    # ========= KFold =========
    skf = KFold(n_splits=10, shuffle=True)

    acclist, prelist, spelist, recalllsit, f1scorelist = [], [], [], [], []
    inference_times = []
    iter = 0

    for train_index, test_index in skf.split(X_raw, Y_raw):
        iter += 1
        print("\n============================")
        print("Iteration:", iter)

        X_train = X_raw[train_index]
        X_test  = X_raw[test_index]
        y_train = Y_raw[train_index]
        y_test  = Y_raw[test_index]

        print("Train Label Distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
        print("Test Label Distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

        # 转 tensor
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)

        X_test  = torch.from_numpy(X_test)
        y_test  = torch.from_numpy(y_test)

        train_loader = DataLoader(
            Data.TensorDataset(X_train, y_train),
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        test_loader = DataLoader(
            Data.TensorDataset(X_test, y_test),
            batch_size=1,
            shuffle=False
        )

        # ========= 构建模型 =========
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        my_net = RuLCNN().float().to(device)

        # ========= Step 1: 训练 =========
        print("Training model on {}...".format(device))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-3)
        
        total_epoch = 20
        my_net.train()
        for epoch in range(total_epoch):  # 建议 50 epoch
            for inputs, labels in train_loader:
                labels = labels.to(device)
                # inputs = inputs.to(device)
                inputs = augment_batch_B1C30T(inputs, fs=FS, device=device)
                
                optimizer.zero_grad()
                logits = my_net(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{total_epoch} | loss={loss.item():.4f}")

        # ========= Step 2: 剪枝 =========
        print("Pruning model (20%) ...")
        prune.l1_unstructured(my_net.pointwise, name='weight', amount=0.2)
        prune.remove(my_net.pointwise, 'weight')

        prune.l1_unstructured(my_net.depthwise, name='weight', amount=0.1)
        prune.remove(my_net.depthwise, 'weight')

        prune.l1_unstructured(my_net.fc, name='weight', amount=0.2)
        prune.remove(my_net.fc, 'weight')

        # ========= Step 3: 剪枝后 Finetune =========
        print("Finetuning pruned model...")
        optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4)
        my_net.train()
        my_net.get_model_sparsity()
        Tepoch=5
        for epoch in range(5):  # 
            for inputs, labels in train_loader:
                labels = labels.cuda()
                # inputs = inputs.to(device)
                inputs = augment_batch_B1C30T(inputs, fs=FS, device=device)
                optimizer.zero_grad()
                logits = my_net(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            print(f"Finetune Epoch {epoch+1}/{Tepoch} | loss={loss.item():.4f}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if iter == 10:
            torch.save(my_net.state_dict(), f"./models-260728/sub{subject}_fold{iter}_pruned_{timestamp}.pth")

        # ========= Step 4: 测试 =========
        print("Evaluating...")
        my_net.eval()

        preds = []
        labels = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.cuda()
                start = time.time()
                logits = my_net(x)
                end = time.time()

                inference_times.append(end - start)

                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).cpu().item()

                preds.append(pred)
                labels.append(y.item())

        preds = np.array(preds)
        labels = np.array(labels)

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        rec = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        spe = recall_score(labels, preds, pos_label=0)

        print(f"Acc={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Spec={spe:.4f}")

        acclist.append(acc)
        prelist.append(prec)
        recalllsit.append(rec)
        f1scorelist.append(f1)
        spelist.append(spe)

    # ========= 最终统计 =========
    print("\n===================Performances=====================")
    print("Average Accuracy:", np.mean(acclist))
    print("Average Precision:", np.mean(prelist))
    print("Average Recall:", np.mean(recalllsit))
    print("Average F1 Score:", np.mean(f1scorelist))
    print("Average Specificity:", np.mean(spelist))
    print(f"Average Running Time: {np.mean(inference_times):.6f}s")



if __name__ == '__main__':
    run()