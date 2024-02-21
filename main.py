import os
import torch
import argparse
import numpy as np

from models import PacificModel
from trainers import PacificTrainer
from utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloder, get_rating_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)

    # model args
    parser.add_argument("--model_name", default="Pacific", type=str)
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--num_heads", default=1, type=int)

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--full_sort", default=False, action="store_true")
    parser.add_argument("--patience", default=10, type=int,
                        help="how long to wait after last time validation loss improved")
    parser.add_argument("--eta", default=0.8, type=float)
    parser.add_argument("--lambda_", default=0.05, type=float)
    parser.add_argument("--mu", default=0.3, type=float)
    parser.add_argument("--phi_max", default=0.9, type=float)
    parser.add_argument("--phi_min", default=0.6, type=float)
    parser.add_argument("--tau", default=0.05, type=float)
    parser.add_argument("--window_size", default=3, type=int)
    parser.add_argument("--last_N", default=2, type=int)


    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item = get_seq_dic(args)

    args.item_size = max_item + 1
    args.user_size = seq_dic['num_users']

    # save model args
    cur_time = get_local_time()
    args_str = f'{args.model_name}-{args.data_name}-{cur_time}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    model = PacificModel(args=args)
    trainer = PacificTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.full_sort:
        args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0, full_sort=args.full_sort)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=args.full_sort)
            # evaluate on MRR
            early_stopping(np.array(scores[-2:-1]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("---------------Sample 99 results---------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=args.full_sort)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

main()
