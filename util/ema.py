def adjust_ema_momentum(epoch, args):
    if epoch < args.ema_warmup_epochs:
        temp_ema_decay = args.model_ema_decay + epoch/args.ema_warmup_epochs * (0.9999 - args.model_ema_decay)
    else:
        if args.model_ema_dynamic:
            temp_ema_decay = 0.9999 + min(args.epochs, epoch-args.ema_warmup_epochs)/args.epochs * (0.99999 - 0.9999)
        else:
            temp_ema_decay = 0.9999
    return temp_ema_decay