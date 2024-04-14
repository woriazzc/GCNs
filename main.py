import os
import time
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from parse import *
from dataset import load_data, implicit_CF_dataset, implicit_CF_dataset_test
from evaluation import Evaluator
import modeling
from utils import seed_all, avg_dict, to_np, Logger

def main(args):
    # Dataset
    num_users, num_items, train_pairs, valid_pairs, test_pairs, train_dict, valid_dict, test_dict, train_matrix, user_pop, item_pop = load_data(args.dataset)
    trainset = implicit_CF_dataset(args.dataset, num_users, num_items, train_pairs, train_matrix, train_dict, user_pop, item_pop, args.num_ns)
    testset = implicit_CF_dataset_test(num_users, num_items, valid_dict, test_dict)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # Model
    all_models = [e.lower() for e in dir(modeling)]
    if args.model.lower() in all_models:
        model = getattr(modeling, dir(modeling)[all_models.index(args.model.lower())])(trainset, args)
    else:
        logger.log('Invalid model name.')
        raise(NotImplementedError, 'Invalid model name.')
    model = model.cuda()

    # Evaluator
    evaluator = Evaluator(args)

    if args.epochs == 0:
        # Evaluation only
        is_improved, early_stop, eval_results, elapsed = evaluator.evaluate_while_training(model, 0, train_loader, testset)
    else:
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        logger.log(f'Epoch [{epoch + 1}/{args.epochs}]')
        tic1 = time.time()
        logger.log('Negative sampling...')
        train_loader.dataset.negative_sampling()

        if hasattr(model, 'do_something_in_each_epoch'):
            logger.log("Model's personal time...")
            model.do_something_in_each_epoch(epoch)

        epoch_loss = []
        logger.log('Training...')
        
        for idx, (batch_user, batch_pos_item, batch_neg_item) in enumerate(train_loader):
            batch_user = batch_user.cuda()
            batch_pos_item = batch_pos_item.cuda()
            batch_neg_item = batch_neg_item.cuda()
            
            # Forward Pass
            model.train()
            output = model(batch_user, batch_pos_item, batch_neg_item)
            loss = model.get_loss(output)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss)

        epoch_loss = torch.mean(torch.stack(epoch_loss)).item()

        toc1 = time.time()
        
        # evaluation
        if epoch % args.eval_period == 0:
            logger.log("Evaluating...")
            is_improved, early_stop, eval_results, elapsed = evaluator.evaluate_while_training(model, epoch, train_loader, testset)
            evaluator.print_result_while_training(logger, epoch_loss, eval_results, is_improved=is_improved, train_time=toc1-tic1, test_time=elapsed)
            if early_stop:
                break
            if is_improved:
                users, items = model.get_all_pre_embedding()
                users, items = to_np(users), to_np(items)
                os.makedirs("crafts", exist_ok=True)
                np.save(f'crafts/{args.dataset}_{args.model}{"" if args.suffix == "" else "_"}{args.suffix}_users.npy', users)
                np.save(f'crafts/{args.dataset}_{args.model}{"" if args.suffix == "" else "_"}{args.suffix}_items.npy', items)
                try:
                    users, items = model.get_all_post_embedding()
                    users, items = to_np(users), to_np(items)
                    np.save(f'crafts/{args.dataset}_{args.model}{"" if args.suffix == "" else "_"}{args.suffix}_post_users.npy', users)
                    np.save(f'crafts/{args.dataset}_{args.model}{"" if args.suffix == "" else "_"}{args.suffix}_post_items.npy', items)
                except:
                    pass
    
    eval_dict = evaluator.eval_dict
    Evaluator.print_final_result(logger, eval_dict)

    return eval_dict


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logger = Logger(LOG_DIR, args.dataset, args.model, args.suffix, args.no_log)

    if args.run_all:
        args_copy = deepcopy(args)
        eval_dicts = []
        for seed in range(5):
            args = deepcopy(args_copy)
            args.seed = seed
            seed_all(args.seed)
            logger.log_args(args)
            eval_dicts.append(main(args))
        
        avg_eval_dict = avg_dict(eval_dicts)

        logger.log('=' * 60)
        Evaluator.print_final_result(logger, avg_eval_dict, prefix="avg ")
    else:
        logger.log_args(args)
        seed_all(args.seed)
        main(args)
