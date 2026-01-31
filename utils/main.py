from generate_channels import get_channels
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
from sklearn.metrics import mutual_info_score
import sys
sys.path.append('../')
sys.path.append('../model')

from AnoFusion import Net
from MyDataset import MyTorchDataset
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import argparse
import logging
import os
import pickle as pkl
from posixpath import join
import json

np.set_printoptions(suppress=True)
# Prefer CUDA, then MPS (Apple), otherwise CPU.
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


cuda_device = select_device()
print(f'device: {cuda_device}')
store_nmiMatrix_path = '../nmiMatrix'


def build_nmi_matrix(channels, metric_shape, log_shape, trace_shape):
    """Compute 6-layer NMI adjacency tensors for metrics/logs/traces."""
    n = channels.shape[0]
    nmiMatrix1 = np.zeros((n, n))
    nmiMatrix2 = np.zeros((n, n))
    nmiMatrix3 = np.zeros((n, n))
    nmiMatrix4 = np.zeros((n, n))
    nmiMatrix5 = np.zeros((n, n))
    nmiMatrix6 = np.zeros((n, n))

    for i1 in range(metric_shape):
        for j1 in range(metric_shape):
            nmiMatrix1[i1][j1] = mutual_info_score(channels[i1], channels[j1])
    for i2 in range(metric_shape, metric_shape + log_shape):
        for j2 in range(metric_shape, metric_shape + log_shape):
            nmiMatrix2[i2][j2] = mutual_info_score(channels[i2], channels[j2])

    for i3 in range(metric_shape + log_shape, metric_shape + log_shape + trace_shape):
        for j3 in range(metric_shape + log_shape, metric_shape + log_shape + trace_shape):
            nmiMatrix3[i3][j3] = mutual_info_score(channels[i3], channels[j3])

    # metric & log
    for i4 in range(metric_shape):
        for j4 in range(metric_shape, metric_shape + log_shape):
            nmiMatrix4[i4][j4] = mutual_info_score(channels[i4], channels[j4])
    for i4 in range(metric_shape, metric_shape + log_shape):
        for j4 in range(metric_shape):
            nmiMatrix4[i4][j4] = mutual_info_score(channels[i4], channels[j4])

    # metric & trace
    for i5 in range(metric_shape):
        for j5 in range(metric_shape + log_shape, metric_shape + log_shape + trace_shape):
            nmiMatrix5[i5][j5] = mutual_info_score(channels[i5], channels[j5])
    for i5 in range(metric_shape + log_shape, metric_shape + log_shape + trace_shape):
        for j5 in range(metric_shape):
            nmiMatrix5[i5][j5] = mutual_info_score(channels[i5], channels[j5])

    # log & trace
    for i6 in range(metric_shape, metric_shape + log_shape):
        for j6 in range(metric_shape + log_shape, metric_shape + log_shape + trace_shape):
            nmiMatrix6[i6][j6] = mutual_info_score(channels[i6], channels[j6])
    for i6 in range(metric_shape + log_shape, metric_shape + log_shape + trace_shape):
        for j6 in range(metric_shape, metric_shape + log_shape):
            nmiMatrix6[i6][j6] = mutual_info_score(channels[i6], channels[j6])

    return np.array([nmiMatrix1, nmiMatrix2, nmiMatrix3, nmiMatrix4, nmiMatrix5, nmiMatrix6])


def get_data(mode_name, matrix_store_path, proportion):
    label_with_timestamp, metric_dataset, log_dataset, trace_dataset = get_channels(args.service_s, mode_name, proportion)
    metric_shape = metric_dataset.shape[1]
    log_shape = log_dataset.shape[1]
    trace_shape = trace_dataset.shape[1]
    channels = pd.concat([metric_dataset, log_dataset, trace_dataset], axis=1)
    channels_columns = channels.columns
    channels = channels.reset_index(drop=True)
    channels = channels.values.T
    print("function get_data")
    print(channels)

    # check for nans
    if not np.isfinite(channels).all():
        bad = np.where(~np.isfinite(channels))
        raise ValueError(f"Found {len(bad[0])} NaNs in channels before normalization: {bad[:5]}")
    
    channels = normalize(channels, axis=1, norm='max')

    # check for nans
    if not np.isfinite(channels).all():
        bad = np.where(~np.isfinite(channels))
        raise ValueError(f"Found {len(bad[0])} NaNs in channels after normalization: {bad[:5]}")
    
    matrix_file = join(store_nmiMatrix_path, matrix_store_path)
    need_recompute = True
    # TO DO: раскомментить
    if os.path.exists(matrix_file):
        with open(matrix_file, 'rb') as f:
            nmiMatrix = pkl.load(f)
        if isinstance(nmiMatrix, np.ndarray) and nmiMatrix.shape[1] == channels.shape[0]:
            need_recompute = False
        else:
            print(f"Cached nmiMatrix shape {getattr(nmiMatrix,'shape',None)} "
                  f"!= current nodes {channels.shape[0]}, recomputing.")
    if need_recompute:
        nmiMatrix = build_nmi_matrix(channels, metric_shape, log_shape, trace_shape)
        print("nmiMatrix.shape:", nmiMatrix.shape)
        if not os.path.exists(store_nmiMatrix_path):
            os.mkdir(store_nmiMatrix_path)
        with open(matrix_file, 'wb') as f:
            pkl.dump(nmiMatrix, f)
    return channels_columns, label_with_timestamp, channels, nmiMatrix


def compute_distance_scores(pred_np, label_np):
    """Calculate max normalized error per sample as in the paper."""
    scores = []
    for t in range(pred_np.shape[0]):
        err = np.abs(pred_np[t] - label_np[t])
        med = np.median(err)
        q25 = np.percentile(err, 25)
        denom = q25 if q25 != 0 else 1e-8  # avoid division by zero
        ed_dis = (err - med) / denom
        scores.append(float(np.max(ed_dis)))
    return scores


def calibrate_threshold(train_loader, net, incident_label_df, quantile=0.99, pred_out='train_predictions.csv'):
    """Collect distances on train/val and pick a quantile as threshold."""
    net.eval()
    scores = []
    preds = []
    labels = []
    timestamps = []
    with torch.no_grad():
        for batch_label, batch_aj, batch_channel, batch_timestamp in train_loader:
            X = batch_channel.float().to(cuda_device)
            A = batch_aj.float().to(cuda_device)
            y_true = batch_label.to(device=cuda_device, dtype=torch.float32)
            output = net(X, A)
            pred_np = output.cpu().numpy()
            label_np = y_true.cpu().numpy()
            scores.extend(compute_distance_scores(pred_np, label_np))
            if pred_out:
                preds.append(pred_np)
                labels.append(label_np)
                timestamps.append(batch_timestamp.cpu().numpy())
    if len(scores) == 0:
        return None
    print('Saving model predictions on test')
    ts = np.concatenate(timestamps, axis=0).astype("int64").reshape(-1)
    # label_arr = np.concatenate(labels, axis=0)
    # label_json = [json.dumps(row.tolist()) for row in label_arr]
    incident_label = None
    if incident_label_df is not None and "timestamp" in incident_label_df.columns and "label" in incident_label_df.columns:
        label_series = pd.Series(
            incident_label_df["label"].values,
            index=pd.to_numeric(incident_label_df["timestamp"], errors="coerce").fillna(0).astype("int64"),
        )
        incident_label = pd.Series(ts).map(label_series).fillna(0).astype("int64").values
    pd.DataFrame(
        {
            "timestamp": ts,
            "distance": np.array(scores, dtype="float32"),
            # "label": label_json,
            "incident_label": incident_label
        }
    ).to_csv(pred_out, index=False)
    return float(np.quantile(scores, quantile))


def eval(label_with_timestamp, model_path, test_loader, threshold_path=None, threshold_value=None, max_samples=None):
    with torch.no_grad():
        if model_path:
            # net = torch.load(model_path).to(cuda_device).eval()
            net = torch.load(model_path, weights_only=False).to(cuda_device).eval()
            distance_frame = pd.DataFrame(
                columns=['timetamp', 'distance', 'pred_bin', 'label', 'threshold_score'])
            debug_prints = 0
            debug_print_limit = 5  # keep stdout readable
            processed = 0

            if threshold_value is None and threshold_path and os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    data = json.load(f)
                    threshold_value = data.get('threshold')
            if threshold_value is not None:
                print(f"[info] Using threshold: {threshold_value:.6f}")
            else:
                print("[info] Threshold not provided; pred_bin will be NaN.")

            for _, (batch_label, batch_aj, batch_channel, batch_timestamp) in enumerate(tqdm(test_loader)):
                X = batch_channel
                A = batch_aj
                X = X.float().to(cuda_device)
                A = A.float().to(cuda_device)

                # check for nans
                if not torch.isfinite(X).all():
                    raise ValueError(f"Found NaNs in X: {torch.where(~torch.isfinite(X))}")
                if not torch.isfinite(A).all():
                    raise ValueError(f"Found NaNs in A: {torch.where(~torch.isfinite(A))}")
                    
                batch_label = batch_label.to(device=cuda_device, dtype=torch.float32)
                label = np.array(
                    batch_label.squeeze().cpu().numpy(), dtype=np.double)
                output = net(X, A)
                if not torch.isfinite(output).all():
                    raise ValueError(f"Found NaNs in output: {torch.where(~torch.isfinite(output))}")
                pred = np.array(output.cpu().numpy(), dtype=np.double)
                batch_timestamp = np.array(batch_timestamp.cpu().numpy())
                for t in range(batch_timestamp.shape[0]):
                    ground_truth = label_with_timestamp[label_with_timestamp['timestamp'] == batch_timestamp[t][0]]
                    ground_truth = ground_truth['label'].values
                    err = []
                    for m in range(len(pred[t])):
                        err.append(abs(pred[t][m] - label[t][m]))
                    ed_dis = []
                    for m in range(len(pred[t])):
                        ed_dis_m = (abs(pred[t][m] - label[t][m]) - np.median(np.array(err))) / np.percentile(np.array(err), 25)
                        ed_dis.append(ed_dis_m)
                    distance = max(ed_dis)
                    if debug_prints < debug_print_limit:
                        print(
                            f"[debug] ts={batch_timestamp[t][0]} "
                            f"err_mean={np.mean(err):.6f} "
                            f"err_med={np.median(err):.6f} "
                            f"err_q25={np.percentile(err,25):.6f} "
                            f"max_ed={distance:.6f} "
                            f"pred_head={pred[t][:5]} "
                            f"label_head={label[t][:5]}"
                        )
                        debug_prints += 1
                    pred_bin = float(distance > threshold_value) if threshold_value is not None else np.nan
                    new_row = pd.DataFrame(
                        {
                            'timetamp': [batch_timestamp[t][0]],
                            'distance': [distance],
                            'pred_bin': [pred_bin],
                            'label': [ground_truth],
                            'threshold_score': [distance],
                        }
                    )
                    distance_frame = pd.concat([distance_frame, new_row], ignore_index=True)
                    processed += 1
                    if max_samples is not None and processed >= max_samples:
                        break
                if max_samples is not None and processed >= max_samples:
                    break

    distance_frame = distance_frame.sort_values('timetamp')
    distance_frame.to_csv(args.service_s+'_ed.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the feed-forward extractor (ff-ext, ML)'
    )
    parser.add_argument('--mode', required=True, help='run in running mode')
    parser.add_argument('--service_s', required=True, help='the service name')
    parser.add_argument('--version', type=int, action='store', help='the evaluate version')
    parser.add_argument('--epoch_num', type=int, default=1, help='the epoch of training')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch_size of training')
    parser.add_argument('--window_size', type=int, default=20, help='the windowsize of data')
    parser.add_argument('--eval_limit', type=int, default=None, help='Limit number of eval samples for quick debug')
    parser.add_argument('--early_stop_patience', type=int, default=5, help='epochs to wait for loss improvement before early stopping')
    parser.add_argument('--early_stop_delta', type=float, default=0.0001, help='min loss improvement to reset patience')
    
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        filename='./train.log',
                        filemode='a',
                        format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )

    checkpoint_path = '../checkpoint'
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
        
    if args.mode == 'train':
        print('====================train===========================')
        print(f'window_size={args.window_size}')
        channels_columns, label_with_timestamp, channels, nmiMatrix = get_data(
            'train', args.service_s+'train_nmiMatrix.pk', 0.6)
        print("Now channels shape:", channels.shape)

        train_data = MyTorchDataset(label_with_timestamp=label_with_timestamp,
                                            channels=channels, aj_matrix=nmiMatrix, window_size=args.window_size)

        
        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataset_length = int(len(train_data)/args.batch_size)

        node_num = channels.shape[0]
        edge_types = 6
        net = Net(node_num=node_num, edge_types=edge_types, window_samples_num=args.window_size, dropout=0.1).to(cuda_device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)

        model_version = 0
        for i in range(100):
            if not os.path.exists(join(checkpoint_path, f'model_save_v{i}')):
                model_version = i
                os.mkdir(join(checkpoint_path, f'model_save_v{i}'))
                model_path = join(
                    join(checkpoint_path, f'model_save_v{i}'))
                break
                
        best_loss = float('inf')
        no_improve = 0
        for epoch in range(args.epoch_num):
            distance_frame = pd.DataFrame(
                columns=['timetamp', 'distance', 'label'])
            running_loss = 0.0
            count = 0
            with tqdm(total=dataset_length) as pbar:
                for step, (batch_label, batch_aj, batch_channel, batch_timestamp) in enumerate(train_loader):
                    count = step
                    X = batch_channel
                    A = batch_aj
                    X = X.float().to(cuda_device)
                    A = A.float().to(cuda_device)

                    # prepare for output show
                    t = batch_timestamp.squeeze().cpu().numpy()
                    output = net(X, A)
                    batch_label = batch_label.to(device=cuda_device, dtype=torch.float32)
                    label = np.array(
                        batch_label.squeeze().cpu().numpy(), dtype=np.double)

                    loss = criterion(
                        output, batch_label.float().squeeze())  # MSE Loss
                    loss = loss.sum()
                    loss.backward()
                    optimizer.step()
                    net.zero_grad()

                    running_loss += loss.item()
                    pred = np.array(
                        output.cpu().detach().numpy(), dtype=np.double)

                    pbar.set_postfix(loss=loss.item(),
                                        epoch=epoch, v_num=model_version)
                    pbar.update(1)

            count += 1
            # logging.info(('epoch:', str(epoch), 'loss:', str(running_loss/count),))
            avg_loss = running_loss/count
            logging.info(('epoch:', str(epoch), 'loss:', str(avg_loss),))
            scheduler.step(running_loss/count)
            torch.save(net, model_path + '/checkpoint_'+args.service_s+'_' +
                        str(epoch) + '_' + str(avg_loss)[:6]+'_model.pkl')
            
            # early stopping
            if avg_loss < best_loss - args.early_stop_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch} with best loss = {best_loss}")
                    break

        # Calibrate anomaly threshold on train data (quantile)
        calibr_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
        threshold = calibrate_threshold(calibr_loader, net, incident_label_df=label_with_timestamp, quantile=0.99)
        if threshold is not None:
            threshold_path = join(model_path, 'threshold.json')
            with open(threshold_path, 'w') as f:
                json.dump({'threshold': threshold, 'quantile': 0.99}, f)
            print(f"Saved threshold {threshold:.4f} to {threshold_path}")
        else:
            print("Threshold calibration skipped (no scores collected).")

    if args.mode == 'eval':
        channels_columns, label_with_timestamp, test_channels, test_nmiMatrix = get_data(
            'test', args.service_s+'test_nmiMatrix.pk', 0.6)
        test_data = MyTorchDataset(label_with_timestamp=label_with_timestamp,
                                        channels=test_channels, aj_matrix=test_nmiMatrix, window_size=args.window_size)
        if len(test_data) == 0:
            raise ValueError("Evaluation dataset is empty. Try reducing window_size or widening the time window.")
        test_loader = DataLoader(
            dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

        threshold_path = '../checkpoint/model_save_v0/threshold.json'
        eval(
            label_with_timestamp,
            '../checkpoint/model_save_v0/checkpoint_mobservice2_0_0.0055_model.pkl',
            test_loader,
            threshold_path=threshold_path,
            max_samples=args.eval_limit,
        )
