import torch
import numpy as np 

def get_batch_generator(
    dataset: np.array,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        assert device == 'cpu' or ('cuda' in device and torch.cuda.is_available())
        if 'cuda' in device:
            num_devices = torch.cuda.device_count()
            target_device = int(device.split(':')) if ':' in device else 0
            assert target_device < num_devices
        device = torch.device(device) 
    except:
        raise AssertionError(f'({device}) is an invalid device name!')

    inputs = torch.from_numpy(
        np.stack(
            [dataset[x:x+context_length] for x in range(0, len(dataset)-context_length)]
        ),
    ).to(device)
    
    labels = torch.from_numpy(
        np.stack(
            [dataset[x:x+context_length] for x in range(1, 1+len(dataset)-context_length)]
        ),
    ).to(device)

    batches_used = 0
    while True:
        if batches_used + batch_size > inputs.shape[0]:
            idx = torch.randperm(inputs.shape[0])
            labels = labels[idx]
            inputs = inputs[idx]
            batches_used = 0
        inputs_sampled = inputs[batches_used:batches_used+batch_size, :]
        labels_sampled = labels[batches_used:batches_used+batch_size, :]
        batches_used += batch_size
        yield inputs_sampled, labels_sampled

def get_batch(
    dataset: np.array,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        assert device == 'cpu' or ('cuda' in device and torch.cuda.is_available())
        if 'cuda' in device:
            num_devices = torch.cuda.device_count()
            target_device = int(device.split(':')) if ':' in device else 0
            assert target_device < num_devices
        device = torch.device(device)
    except:
        raise AssertionError(f'({device}) is an invalid device name!')

    inputs = torch.from_numpy(
        np.stack(
            [dataset[x:x+context_length] for x in range(0, len(dataset)-context_length)]
        ),
    ).to(device)
    
    labels = torch.from_numpy(
        np.stack(
            [dataset[x:x+context_length] for x in range(1, 1+len(dataset)-context_length)]
        ),
    ).to(device)

    idx = torch.randperm(inputs.shape[0])
    labels = labels[idx]
    inputs = inputs[idx]
    inputs_sampled = inputs[:batch_size, :]
    labels_sampled = labels[:batch_size, :]

    return inputs_sampled, labels_sampled