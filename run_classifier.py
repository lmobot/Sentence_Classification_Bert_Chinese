""" Finetuning the library models for sequence classification ."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from prepare_finetune_data import get_train_examples, get_dev_examples, get_test_examples, load_train_dev_data

# from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification, \
    SiameseAlbertForSequenceClassification  # chinese version
from model import tokenization_albert

from model.modeling_bert import BertConfig, BertForSequenceClassification, \
    SiameseBertForSequenceClassification  # chinese version
from model import tokenization_bert

from model.modeling_electra import ElectraConfig, ElectraForSequenceClassification

from callback.optimization.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from processors import ccf_convert_examples_to_features, ccf_convert_examples_to_Siamese_features
from processors import collate_fn, collate_fn_predict, collate_fn_Siamese, collate_fn_predict_Siamese
from tools.common import seed_everything
from tools.common import init_logger, logger
from callback.progressbar import ProgressBar

id2label = {0: '财经', 1: '房产', 2: '家居', 3: '教育', 4: '科技', 5: '时尚', 6: '时政', 7: '游戏', 8: '娱乐', 9: '体育'}
id2risk = {0: '高风险', 1: '中风险', 2: '可公开', 3: '低风险', 4: '中风险', 5: '低风险', 6: '高风险', 7: '低风险', 8: '可公开', 9: '可公开'}
label2id = {'财经': 0, '房产': 1, '家居': 2, '教育': 3, '科技': 4, '时尚': 5, '时政': 6, '游戏': 7, '娱乐': 8, '体育': 9}


def train(args, train_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_training_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Instantaneous batch size all GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(args.num_train_epochs):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training epoch ' + str(epoch + 1))
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }
            # inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            pbar(step, {'batch loss': loss.item()})

        print(" ")

        # save model every epoch
        if args.local_rank in [-1, 0]:
            # Save model checkpoint
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            epoch_average_loss = epoch_loss / len(train_dataloader)
            finetune_model_name = '{}_epoch={}_batch={}_loss={:.4f}_torch.bin'.format(args.model_name, epoch + 1,
                                                                                      args.train_batch_size,
                                                                                      epoch_average_loss)
            finetune_model_save_path = os.path.join(args.output_dir, finetune_model_name)
            model_to_save.save_pretrained(finetune_model_save_path)
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Epoch = %d, Epoch average loss = %.4f", epoch + 1, epoch_average_loss)
            logger.info("Saving model %s to %s", finetune_model_name, args.output_dir)
        tr_loss += epoch_average_loss
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / args.num_train_epochs


def train_and_dev(args, train_dataset, valid_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(valid_dataset) if args.local_rank == -1 else DistributedSampler(valid_dataset)
    eval_dataloader = DataLoader(valid_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_training_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Instantaneous batch size all GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_eval_acc = 0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(args.num_train_epochs):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training epoch ' + str(epoch + 1))
        epoch_loss = 0.0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }
            # inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            pbar(step, {'batch loss': loss.item()})

        print(" ")

        # save model every epoch
        if args.local_rank in [-1, 0]:
            # eval model
            epoch_acc = 0
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(valid_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluation epoch ' + str(epoch + 1))
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                with torch.no_grad():
                    inputs = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]
                    }
                    # inputs['token_type_ids'] = batch[2]
                    outputs = model(**inputs)
                    logits = outputs[0]
                preds = torch.argmax(logits, dim=1)
                labels = batch[3]
                epoch_acc += (preds == labels).sum().item()
                pbar(step)
            print(' ')

            # Model description
            epoch_average_loss = epoch_loss / len(train_dataloader)
            epoch_acc = epoch_acc / len(valid_dataset)
            max_eval_acc = max(max_eval_acc, epoch_acc)
            finetune_model_description = 'model={}  epoch={}  batch={}  loss={:.4f}  acc={:.4f}'.format(args.model_name,
                                                                                                        epoch + 1,
                                                                                                        args.train_batch_size,
                                                                                                        epoch_average_loss,
                                                                                                        epoch_acc)
            logger.info("Model description: %s", finetune_model_description)

            # Save model checkpoint
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training

            finetune_model_name = '{}_epoch={}_batch={}_loss={:.4f}_acc={:.4f}_torch.bin'.format(args.model_name,
                                                                                                 epoch + 1,
                                                                                                 args.train_batch_size,
                                                                                                 epoch_average_loss,
                                                                                                 epoch_acc)
            finetune_model_save_path = os.path.join(args.output_dir, finetune_model_name)
            model_to_save.save_pretrained(finetune_model_save_path)
            logger.info("Epoch = %d, Epoch average loss = %.4f, Eval acc = %.4f, Best eval acc = %.4f", epoch + 1,
                        epoch_average_loss,
                        epoch_acc, max_eval_acc)
            logger.info("Saving model %s to %s", finetune_model_name, args.output_dir)
        tr_loss += epoch_average_loss
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / args.num_train_epochs


def train_and_dev_Siamese(args, train_dataset, valid_dataset, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn_Siamese)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(valid_dataset) if args.local_rank == -1 else DistributedSampler(valid_dataset)
    eval_dataloader = DataLoader(valid_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn_Siamese)

    num_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(num_training_steps * args.warmup_proportion)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer = Lamb(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_training_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Instantaneous batch size all GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    max_eval_acc = 0
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(args.num_train_epochs):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training epoch ' + str(epoch + 1))
        epoch_loss = 0.0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_temp_len = batch[0].size()[1]
            if input_temp_len > 512:
                input_ids_list = torch.split(batch[0], [512, input_temp_len - 512], dim=1)
                attention_mask_list = torch.split(batch[1], [512, input_temp_len - 512], dim=1)
                token_type_ids_list = torch.split(batch[2], [512, input_temp_len - 512], dim=1)
                inputs = {
                    'input_ids_a': input_ids_list[0],
                    'input_ids_b': input_ids_list[1],
                    'attention_mask_a': attention_mask_list[0],
                    'attention_mask_b': attention_mask_list[1],
                    'token_type_ids_a': token_type_ids_list[0],
                    'token_type_ids_b': token_type_ids_list[1],
                    'labels': batch[3]
                }
            else:
                inputs = {
                    'input_ids_a': batch[0],
                    'attention_mask_a': batch[1],
                    'token_type_ids_a': batch[2],
                    'labels': batch[3]
                }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            pbar(step, {'batch loss': loss.item()})

        print(" ")

        # save model every epoch
        if args.local_rank in [-1, 0]:
            # Eval model
            epoch_acc = 0
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(valid_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)
            pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluation epoch ' + str(epoch + 1))
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                with torch.no_grad():
                    input_temp_len = batch[0].size()[1]
                    if input_temp_len > 512:
                        input_ids_list = torch.split(batch[0], [512, input_temp_len - 512], dim=1)
                        attention_mask_list = torch.split(batch[1], [512, input_temp_len - 512], dim=1)
                        token_type_ids_list = torch.split(batch[2], [512, input_temp_len - 512], dim=1)
                        inputs = {
                            'input_ids_a': input_ids_list[0],
                            'input_ids_b': input_ids_list[1],
                            'attention_mask_a': attention_mask_list[0],
                            'attention_mask_b': attention_mask_list[1],
                            'token_type_ids_a': token_type_ids_list[0],
                            'token_type_ids_b': token_type_ids_list[1],
                        }
                    else:
                        inputs = {
                            'input_ids_a': batch[0],
                            'attention_mask_a': batch[1],
                            'token_type_ids_a': batch[2],
                        }
                    outputs = model(**inputs)
                    logits = outputs[0]

                preds = torch.argmax(logits, dim=1)
                labels = batch[3]
                epoch_acc += (preds == labels).sum().item()
                pbar(step)
            print(' ')

            # Model description
            epoch_average_loss = epoch_loss / len(train_dataloader)
            epoch_acc = epoch_acc / len(valid_dataset)
            max_eval_acc = max(max_eval_acc, epoch_acc)
            finetune_model_description = 'model={}  epoch={}  batch={}  loss={:.4f}  acc={:.4f}'.format(args.model_name,
                                                                                                        epoch + 1,
                                                                                                        args.train_batch_size,
                                                                                                        epoch_average_loss,
                                                                                                        epoch_acc)
            logger.info("Model description: %s", finetune_model_description)

            # Save model checkpoint
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training

            finetune_model_name = '{}_epoch={}_batch={}_loss={:.4f}_acc={:.4f}_torch.bin'.format(args.model_name,
                                                                                                 epoch + 1,
                                                                                                 args.train_batch_size,
                                                                                                 epoch_average_loss,
                                                                                                 epoch_acc)
            finetune_model_save_path = os.path.join(args.output_dir, finetune_model_name)
            model_to_save.save_pretrained(finetune_model_save_path)
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Epoch = %d, Epoch average loss = %.4f, Eval acc = %.4f, Best eval acc = %.4f", epoch + 1,
                        epoch_average_loss,
                        epoch_acc, max_eval_acc)
            logger.info("Saving model %s to %s", finetune_model_name, args.output_dir)
        tr_loss += epoch_average_loss
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / args.num_train_epochs


def predict_Siamese(args, predict_dataset, model):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(predict_dataset) if args.local_rank == -1 else DistributedSampler(predict_dataset)
    eval_dataloader = DataLoader(predict_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn_predict_Siamese)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("***** Model path : {} *****".format(args.model_path))
    logger.info("  Num examples = %d", len(predict_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    result_list = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Predict")
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            input_temp_len = batch[0].size()[1]
            if input_temp_len > 512:
                input_ids_list = torch.split(batch[0], [512, input_temp_len - 512], dim=1)
                attention_mask_list = torch.split(batch[1], [512, input_temp_len - 512], dim=1)
                token_type_ids_list = torch.split(batch[2], [512, input_temp_len - 512], dim=1)
                inputs = {
                    'input_ids_a': input_ids_list[0],
                    'input_ids_b': input_ids_list[1],
                    'attention_mask_a': attention_mask_list[0],
                    'attention_mask_b': attention_mask_list[1],
                    'token_type_ids_a': token_type_ids_list[0],
                    'token_type_ids_b': token_type_ids_list[1],
                }
            else:
                inputs = {
                    'input_ids_a': batch[0],
                    'attention_mask_a': batch[1],
                    'token_type_ids_a': batch[2],
                }
            outputs = model(**inputs)
            logits = outputs[0]
        preds = torch.argmax(logits, dim=1)
        preds = preds.detach().cpu().tolist()
        result_list.extend(preds)
        pbar(step)
    print(' ')
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return result_list


def predict(args, predict_dataset, model):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(predict_dataset) if args.local_rank == -1 else DistributedSampler(predict_dataset)
    eval_dataloader = DataLoader(predict_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn_predict)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("***** Model path : {} *****".format(args.model_path))
    logger.info("  Num examples = %d", len(predict_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    result_list = []
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Predict")
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }
            # inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]
        preds = torch.argmax(logits, dim=1)
        preds = preds.detach().cpu().tolist()
        result_list.extend(preds)
        pbar(step)
    print(' ')
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()
    return result_list


def convert_features_to_tensors_dataset(features, data_type='train'):
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if data_type == 'test':
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens)
    else:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    return dataset


def convert_features_to_tensors_Siamese_dataset(features, data_type='train'):
    # Convert to Tensors and build Siamese dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens_a = torch.tensor([f.input_len_a for f in features], dtype=torch.long)
    all_lens_b = torch.tensor([f.input_len_b for f in features], dtype=torch.long)
    if data_type == 'test':
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens_a, all_lens_b)
    else:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens_a, all_lens_b,
                                all_labels)
    return dataset


def ccf_load_train_and_dev_examples(args, tokenizer):
    # ccf_classification
    # 面向数据安全治理的数据内容智能发现与分级分类
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    train_examples, valid_examples = load_train_dev_data()

    label_list = label2id.keys()
    train_features = ccf_convert_examples_to_features(train_examples,
                                                      tokenizer,
                                                      label_list=label_list,
                                                      max_seq_length=args.max_seq_length)

    valid_features = ccf_convert_examples_to_features(valid_examples,
                                                      tokenizer,
                                                      label_list=label_list,
                                                      max_seq_length=args.max_seq_length)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return convert_features_to_tensors_dataset(train_features), convert_features_to_tensors_dataset(valid_features)


def ccf_load_train_and_dev_examples_Siamese(args, tokenizer):
    # ccf_classification
    # 面向数据安全治理的数据内容智能发现与分级分类
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    train_examples, valid_examples = load_train_dev_data()

    label_list = label2id.keys()
    train_features = ccf_convert_examples_to_Siamese_features(train_examples, tokenizer,
                                                              max_seq_length=args.max_seq_length, label_list=label_list)

    valid_features = ccf_convert_examples_to_Siamese_features(valid_examples, tokenizer,
                                                              max_seq_length=args.max_seq_length, label_list=label_list)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return convert_features_to_tensors_Siamese_dataset(train_features), convert_features_to_tensors_Siamese_dataset(
        valid_features)


def ccf_load_and_cache_examples_Siamese(args, tokenizer, data_type='train'):
    # ccf_classification
    # 面向数据安全治理的数据内容智能发现与分级分类
    if args.local_rank not in [-1, 0] and data_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if data_type == 'train':
        examples = get_train_examples()
    elif data_type == 'dev':
        examples = get_dev_examples()
    else:
        examples = get_test_examples()

    if data_type == 'test':
        features = ccf_convert_examples_to_Siamese_features(examples,
                                                            tokenizer,
                                                            max_seq_length=args.max_seq_length)
    else:
        label_list = label2id.keys()
        features = ccf_convert_examples_to_Siamese_features(examples,
                                                            tokenizer,
                                                            label_list=label_list,
                                                            max_seq_length=args.max_seq_length)
    if args.local_rank == 0 and data_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    return convert_features_to_tensors_Siamese_dataset(features, data_type)


def ccf_load_and_cache_examples(args, tokenizer, data_type='train'):
    # ccf_classification
    # 面向数据安全治理的数据内容智能发现与分级分类
    if args.local_rank not in [-1, 0] and data_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if data_type == 'train':
        examples = get_train_examples()
    elif data_type == 'dev':
        examples = get_dev_examples()
    else:
        examples = get_test_examples()

    if data_type == 'test':
        features = ccf_convert_examples_to_features(examples,
                                                    tokenizer,
                                                    max_seq_length=args.max_seq_length)
    else:
        label_list = label2id.keys()
        features = ccf_convert_examples_to_features(examples,
                                                    tokenizer,
                                                    label_list=label_list,
                                                    max_seq_length=args.max_seq_length)

    if args.local_rank == 0 and data_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    return convert_features_to_tensors_dataset(features, data_type)


def confirm_model(args, use_model='albert'):
    if use_model == 'albert':
        config = AlbertConfig.from_pretrained(args.config_name if args.config_name else args.model_path,
                                              num_labels=args.num_labels)

        tokenizer = tokenization_albert.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case,
                                                      spm_model_file=args.spm_model_file)

        if not args.siamese_flag:
            model = AlbertForSequenceClassification.from_pretrained(args.model_path,
                                                                    from_tf=bool('.ckpt' in args.model_path),
                                                                    config=config)
        if args.siamese_flag:
            # albert孪生网络
            model = SiameseAlbertForSequenceClassification.from_pretrained(args.model_path,
                                                                           from_tf=bool('.ckpt' in args.model_path),
                                                                           config=config)
    elif use_model == 'bert':
        config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_path,
                                            num_labels=args.num_labels)

        tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

        if not args.siamese_flag:
            model = BertForSequenceClassification.from_pretrained(args.model_path,
                                                                  from_tf=bool('.ckpt' in args.model_path),
                                                                  config=config)
        if args.siamese_flag:
            # albert孪生网络
            model = SiameseBertForSequenceClassification.from_pretrained(args.model_path,
                                                                         from_tf=bool('.ckpt' in args.model_path),
                                                                         config=config)
    elif use_model == 'electra':
        config = ElectraConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                               num_labels=args.num_labels)

        tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

        model = ElectraForSequenceClassification.from_pretrained(args.model_path,
                                                                 from_tf=bool('.ckpt' in args.model_path),
                                                                 config=config)
    return tokenizer, model


def main(model_config_dict, do_operation='train', use_model='albert', siamese_flag=True):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,  # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_path", default=None, type=str,  # required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--output_dir", default=None, type=str,  # required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--vocab_file", default='', type=str)
    parser.add_argument("--num_labels", default=10, type=int)
    parser.add_argument("--siamese_flag", default=True, type=bool, help="Use siamese network")
    parser.add_argument("--spm_model_file", default=None, type=str)  # 外部sentencePiece词表

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--do_train", default=False, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", default=False, type=bool,
                        help="Whether to run the model in inference mode on the test set.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # 可调
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    # 可调 bert 0.01
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    # 可调 0.05 1e-6
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    # local_rank代表当前程序进程使用的GPU标号
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    # 参数
    args.data_dir = './finetunedata'
    args.vocab_file = './prev_trained_model/vocab.txt'
    args.config_name = model_config_dict['config']
    args.model_path = model_config_dict['model']
    args.siamese_flag = siamese_flag

    if 'train' in do_operation:
        args.do_train = True
        args.output_dir = 'outputs/ccf_model'
        args.learning_rate = model_config_dict['learning_rate']
        suffix = 'finetune'

    else:
        if 'dev' in do_operation:
            args.output_dir = 'outputs/ccf_unlabeled_result'
            suffix = 'labeled'
            args.do_eval = True
        elif 'test' in do_operation:
            args.output_dir = 'outputs/ccf_test_result'
            suffix = 'predict'
            args.do_predict = True


    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    _model_name = args.model_path.split('/')[-1].split('_')[:-1]
    args.model_name = '_'.join(_model_name)
    _model_name.append(suffix)
    args.output_dir = args.output_dir + '/{}'.format('_'.join(_model_name))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    init_logger(log_file=args.output_dir + '/process.log')

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        '''
        在运行的过程中产生了很多个进程，具体多少个取决你GPU的数量，
        需要torch.cuda.set_device(args.local_rank)
        设定默认的GPU（默认gpu的作用）
        因为torch.distributed.launch为我们触发了n个YOUR_TRAINING_SCRIPT.py进程
        n就是我们将要使用的GPU数量。
        '''
        # 这里不太明白 为什么只有一块gpu?
        # local_rank 到底干嘛用的？
        torch.cuda.set_device(args.local_rank)  # 模型和数据加载到对应GPU上
        device = torch.device("cuda", args.local_rank)  # 指定gpu
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    args.device = device
    # Setup logging
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    seed_everything(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # model
    tokenizer, model = confirm_model(args, use_model=use_model)
    model.to(args.device)

    # summary(model, input_size=[(10, 10)], batch_size=1, device="cuda")

    # Training
    if args.do_train:
        logger.info("Training/evaluation parameters %s", args)

        # train
        # train_dataset = ccf_load_and_cache_examples(args, tokenizer, data_type='train')
        # global_step, tr_loss = train(args, train_dataset, model)

        if not args.siamese_flag:
            # train and dev
            train_dataset, valid_dataset = ccf_load_train_and_dev_examples(args, tokenizer)
            global_step, tr_loss = train_and_dev(args, train_dataset, valid_dataset, model)

        if args.siamese_flag:
            # train and dev Siamese network
            train_dataset, valid_dataset = ccf_load_train_and_dev_examples_Siamese(args, tokenizer)
            global_step, tr_loss = train_and_dev_Siamese(args, train_dataset, valid_dataset, model)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    elif args.do_eval:
        # predict unlabeled data
        logger.info("Predicting parameters %s", args)

        if not args.siamese_flag:
            # predict general
            predict_dataset = ccf_load_and_cache_examples(args, tokenizer, data_type='test')
            predict_result = predict(args, predict_dataset, model)

        if args.siamese_flag:
            # predict Siamese network
            predict_dataset = ccf_load_and_cache_examples_Siamese(args, tokenizer, data_type='test')
            predict_result = predict_Siamese(args, predict_dataset, model)

        result_path = os.path.join(args.output_dir, 'unlabeled_result.txt')
        logger.info("Save predict result to %s", result_path)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write('id,class_label,rank_label\n')
            for _id, _label in enumerate(predict_result):
                f.write(str(_id) + ',' + id2label[_label] + ',' + id2risk[_label] + '\n')
        logger.info("Save done")

    elif args.do_predict:
        logger.info("Predicting parameters %s", args)

        if not args.siamese_flag:
            # predict general
            predict_dataset = ccf_load_and_cache_examples(args, tokenizer, data_type='test')
            predict_result = predict(args, predict_dataset, model)

        if args.siamese_flag:
            # predict Siamese network
            predict_dataset = ccf_load_and_cache_examples_Siamese(args, tokenizer, data_type='test')
            predict_result = predict_Siamese(args, predict_dataset, model)

        result_path = os.path.join(args.output_dir, 'test_result.txt')
        logger.info("Save predict result to %s", result_path)
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write('id,class_label,rank_label\n')
            for _id, _label in enumerate(predict_result):
                f.write(str(_id) + ',' + id2label[_label] + ',' + id2risk[_label] + '\n')
        logger.info("Save done")


if __name__ == "__main__":
    
    # albert tiny pre
    albert_tiny_config = './prev_trained_model/torch_albert_tiny/albert_config_tiny.json'
    albert_tiny_pre_model = './prev_trained_model/torch_albert_tiny/albert_tiny_zh_pre.bin'
    albert_tiny_pre_config = {'config': albert_tiny_config, 'model': albert_tiny_pre_model, 'learning_rate': 5e-5}

    # macbert base pre
    macbert_config = './prev_trained_model/torch_macbert_base/macbert_config.json'
    macbert_pre_model = './prev_trained_model/torch_macbert_base/macbert_base_pre.bin'
    macbert_pre_config = {'config': macbert_config, 'model': macbert_pre_model, 'learning_rate': 2e-5}
    
    # albert base
    albert_base_config = './prev_trained_model/torch_albert_base/albert_config_base.json'
    albert_base_pre_model = './prev_trained_model/torch_albert_base/albert_base_zh_pre.bin'
    albert_base_pre_config = {'config': albert_base_config, 'model': albert_base_pre_model, 'learning_rate': 5e-5}

    # albert large
    albert_large_config = './prev_trained_model/torch_albert_large/albert_config_large.json'
    albert_large_pre_model = './prev_trained_model/torch_albert_large/albert_large_zh_pre.bin'
    albert_large_pre_config = {'config': albert_large_config, 'model': albert_large_pre_model, 'learning_rate': 5e-5}
    
    # roberta large
    roberta_large_config = './prev_trained_model/torch_roberta_wwm_large_ext/roberta_wwm_large_ext_zh_config.json'
    roberta_large_pre_model = './prev_trained_model/torch_roberta_wwm_large_ext/roberta_wwm_large_ext_zh_pre.bin'
    roberta_large_pre_config = {'config': roberta_large_config, 'model': roberta_large_pre_model, 'learning_rate': 2e-5}

    # roberta base pre
    roberta_config = './prev_trained_model/torch_roberta_wwm_ext/roberta_wwm_ext_zh_config.json'
    roberta_pre_model = './prev_trained_model/torch_roberta_wwm_ext/roberta_wwm_ext_zh_pre.bin'
    roberta_pre_config = {'config': roberta_config, 'model': roberta_pre_model, 'learning_rate': 2e-5}
    
    # electra base pre
    electra_config = './prev_trained_model/torch_electra_base/electra_base_discriminator_config.json'
    electra_pre_model = './prev_trained_model/torch_electra_base/electra_base_discriminator_pre.bin'
    electra_pre_config = {'config': electra_config, 'model': electra_pre_model, 'learning_rate': 5e-5}


    # train dev test

    # albert predict
    # main(model_config_dict=albert_tiny_pre_config, do_operation='train', use_model='albert', siamese_flag=False)

    # roberta finetune
    main(model_config_dict=roberta_pre_config, do_operation='train', use_model='bert', siamese_flag=False)
    
    # macbert
    # main(model_config_dict=macbert_pre_config, do_operation='train', use_model='bert', siamese_flag=False)
    
    # electra
    # main(model_config_dict=electra_pre_config, do_operation='train', use_model='electra', siamese_flag=False)

