import os
import time
import torch
import argparse
from loguru import logger
from datetime import datetime
from transformers import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from src.dataset import AgNewsDataset, collate_fn
from src.model import PetModel, PetCriterion
from src.utils import stats_time
from src.eval import evaluation



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization.")
    parser.add_argument('--batch_size', default=4, type=int, help="Total batch size for training.")
    parser.add_argument('--epochs', default=50, type=int, help='The epoch of train')
    parser.add_argument('--model_path', required=True, type=str, help='The pretrained model')
    parser.add_argument('--max_seq_length', default=256, type=int, help="The maximum length of squence")
    parser.add_argument('--train_data_path', type=str, required=True, help="The path of train dataset")
    parser.add_argument('--eval_data_path', type=str, required=True, help="The path of eval dataset")
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpts', help="The directory of checkpoints")
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard', help="The directory of tensorboard")
    parser.add_argument('--log_dir', type=str, default='./log', help="The directory of log")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help="The step of gradient accumulation")
    parser.add_argument('--learning_rate', default=1.5e-3, type=float, help="The initial learning rate for optimizer")
    parser.add_argument('--eval_freq', default=1000, type=int, help='The freq of eval test set')
    parser.add_argument('--log_freq', default=200, type=int, help='The freq of print log')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="The adam epsilon")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help="The maximum gradient normalization")
    parser.add_argument('--warmup_steps', default=0, type=float, help="The steps of  warm up")

    parser.add_argument('--pet_repetitions', default=3, type=int, help="The pet repetitions")
    parser.add_argument('--pattern_ids', default=4, type=int, help="The ids of pet pattern")
    parser.add_argument('--pet_batch_size', default=4, type=int, help="The batch size of pet")
    parser.add_argument('--max_num_verbalizers', default=1, type=int, help="The maximum numbers of verbalizer")

    args = parser.parse_args()

    return args


def setup_training(config):
    set_seed(config.seed)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.device = device

    # tensorboard set up
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    comment = f'bath_size={config.batch_size} lr={config.learning_rate}'
    writer = SummaryWriter(log_dir=config.tensorboard_dir + "/" + time_stamp, comment=comment)

    # cuda setup
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # hidden tokenizer warning
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # logger setup
    logger.add(os.path.join(config.log_dir) + "/" +"{time}.log")

    # checkpoint dir setup
    checkpoint_dir = os.path.join(config.checkpoint_dir, time_stamp)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    config.checkpoint_dir = checkpoint_dir

    return config, writer

def prepare_data_loader(config, num_workers=1):
    train_dataset = AgNewsDataset(config.train_data_path, config.model_path, config.pet_repetitions, config.max_seq_length)
    train_data_iter = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)

    eval_dataset = AgNewsDataset(config.eval_data_path, config.model_path, config.pet_repetitions, config.max_seq_length)
    eval_data_iter = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=config.batch_size, num_workers=num_workers, shuffle=True)


    return train_data_iter, eval_data_iter, train_dataset.m2c_tensor, train_dataset.filler_len

def prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step):
    model = PetModel(config)
    model.to(config.device)
    criterion = PetCriterion(config, m2c_tensor, filler_len)
    criterion.to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon, no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_step)
    return model, criterion, optimizer, scheduler


def trainer():
    config = parse_arguments()
    config, writer = setup_training(config)
    train_iter, eval_iter,  m2c_tensor, filler_len = prepare_data_loader(config)
    total_step = config.epochs * len(train_iter)

    model, criterion, optimizer, scheduler = prepare_model_and_optimizer(config, m2c_tensor, filler_len, total_step)

    logger.info(f"{'#' * 41} Config {'#' * 41}")
    for k in list(vars(config).keys()):
        logger.info('{0}: {1}'.format(k, vars(config)[k]))
    logger.info(f'total step: {total_step}')
    logger.info(f'the number of train step: {len(train_iter)}')
    logger.info(f'the size of train set: {len(train_iter) * config.batch_size}')
    logger.info(f'the size of eval set: {len(eval_iter) * config.batch_size}')
    logger.info(f"{'#' * 41} Training {'#' * 41}")

    start = int(time.time())
    step = 0
    avg_loss = 0.0
    global_acc = 0.0
    most_recent_ckpts_paths = []
    for epoch in range(1, config.epochs+1):
        # train
        for i, batch in enumerate(train_iter):
            model.train()
            step += 1
            input_ids, token_type_ids, attention_mask, mlm_labels, labels = [w.to(config.device) for w in batch]

            logit = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            loss, predictions = criterion(logit, mlm_labels, labels)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            avg_loss += loss.item()
            if step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


            # tensorboard
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('avg_loss', avg_loss / step, step)

            # log
            if step % config.log_freq == 0:
                end = int(time.time())
                logger.info(f"epochs:{str(epoch) + '/' + str(config.epochs)}, batch:{str(i + 1) + '/' + str(len(train_iter))}, step:{str(step) + '/' + str(total_step)}, cur_loss:{'{:.6f}'.format(loss)}, avg_loss:{'{:.6f}'.format(avg_loss/step)}, eta:{stats_time(start, end, step, total_step)}h")

            # evaluation
            if step % config.eval_freq == 0:
                acc = evaluation(eval_iter, model, criterion, config.device)
                writer.add_scalar('acc', acc, step)
                logger.info(f"epochs:{str(epoch) + '/' + str(config.epochs)}, step:{str(step) + '/' + str(total_step)}, avg_acc:{'{:.6f}'.format(acc)}")

                # checkpoint path
                if acc > global_acc:
                    cur_path = os.path.join(config.checkpoint_dir, f"epoch{epoch}_step{step}_acc{'{:.6f}'.format(acc)}.pt")
                    torch.save(model.state_dict(), cur_path)
                    global_acc = acc

                    most_recent_ckpts_paths.append(cur_path)
                    if len(most_recent_ckpts_paths) > 3: # 仅仅保存最新的三个checkpoint
                        ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                        os.remove(ckpt_to_be_removed)


if __name__ == '__main__':
    trainer()
