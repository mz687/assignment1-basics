from args import parse_args, print_args
from module.model import TransformerLM
from module.optimizer import AdamW, grad_clip, learning_rate_schedule, LRScheduler
from module.loss_fn import cross_entropy_loss
from module.checkpointing import save_checkpoint, load_checkpoint
from module.data_loader import get_batch, iterate_eval_dataset

import os
import numpy as np
from datetime import datetime
import time

import torch

@torch.no_grad()
def eval(eval_data_np, model, args):
    losses = 0.
    num_losses = 0
    model.eval()
    for inputs, labels in iterate_eval_dataset(
        dataset=eval_data_np,
        batch_size=args.eval_batch_size,
        context_length=args.context_length,
        device=args.device_name
    ):
        pred = model(inputs)
        losses += cross_entropy_loss(
            pred.view(-1, args.vocab_size), 
            labels.view(-1)
        ).detach().item()
        num_losses += 1
    return losses / num_losses

def main():
    args = parse_args()
    print_args(args)

    args.device_name = "cuda" if torch.cuda.is_available() else 'cpu'
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_data_np = np.load(
        args.train_data_path,
        mmap_mode='r'
    )
    eval_data_np = np.load(
        args.valid_data_path,
        mmap_mode='r'
    )

    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay
    )

    t = 0
    # load ckpt if args.load exists
    if os.path.exists(args.load):
        t = load_checkpoint(
            src=args.load,
            model=model,
            optimizer=optimizer
        )
        print(f'Found ckpt stored in {args.load}\nWill resume training from step ({t})')

    lr_scheduler = LRScheduler(
        t=t,
        optimizer=optimizer,
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.train_iters - args.warmup_iters # hardcode 
    )

    for step in range(t, args.train_iters):
        optimizer.zero_grad()
        model.train()
        start = time.time()
        inputs, labels = get_batch(
            dataset=train_data_np,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device_name
        )
        pred = model(inputs)
        loss = cross_entropy_loss(
            pred.view(-1, args.vocab_size), 
            labels.view(-1)
        )
        loss.backward()

        grad_norm = grad_clip(
            params=model.parameters(),
            max_l2_norm=args.clip_grad
        )
        optimizer.step()
        lr_scheduler.step()
        latest_lr = lr_scheduler.get_last_lr()[0]

        elapsed = (time.time() - start) * 1000

        if (step+1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=args.save
            )

        time_now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f'{time_now} iteration {step+1} / {args.train_iters} | elapsed time: {elapsed:.4f} | loss: {loss.item():.4f} | grad_norm: {grad_norm:.4e} | learning rate: {latest_lr:.4e}')

        if args.eval_interval is not None and (step + 1) % args.eval_interval == 0:
            avg_eval_loss = eval(
                model=model,
                eval_data_np=eval_data_np,
                args=args
            )
            print('-'*40)
            print(f'validation loss at step {step + 1}: {avg_eval_loss:.4f}')
            print('-'*40)

if __name__ == '__main__':
    main()