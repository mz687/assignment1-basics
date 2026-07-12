from .args import parse_args
from module.model import TransformerLM
from module.optimizer import AdamW, grad_clip, learning_rate_schedule, LRScheduler
from module.loss_fn import cross_entropy_loss
from module.checkpointing import save_checkpoint, load_checkpoint
from module.data_loader import get_batch

import os
from datetime import datetime

def main():
    args = parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

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

    lr_scheduler = LRScheduler(
        t=t,
        optimizer=optimizer,
        max_learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.train_iters - args.warmup_iters # hardcode 
    )

    for step in range(t, args.train_iters):
        inputs, logits = 
        pred = model(inputs)
        loss = cross_entropy_loss(logits, pred)

        optimizer.step()
        lr_scheduler.step()

        latest_lr = lr_scheduler.get_last_lr()

        if step % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=args.save
            )

        time_now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f'{time_now} iteration {step+1} / {args.train_iters} | loss: {loss:.2f} | learning rate: {latest_lr:.2f}')

if __name__ == '__main__':
    main()