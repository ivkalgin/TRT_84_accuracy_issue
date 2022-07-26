import argparse

import tensorrt as trt
import torch
from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class TrtModel:
    def __init__(self, path_to_engine):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path_to_engine, 'rb') as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        self.context = self.engine.create_execution_context()

        self.bindings = [None]*2
        self.input_ids_index = self.engine.get_binding_index('input_ids')
        self.logits_index = self.engine.get_binding_index('logits')

        self.bs, self.seq_len = self.engine.get_binding_shape(self.input_ids_index)

    def run(self, input_ids, logits):
        self.bindings[self.input_ids_index] = int(input_ids.data_ptr())
        self.bindings[self.logits_index] = int(logits.data_ptr())

        if not self.context.execute_v2(bindings=self.bindings):
            raise RuntimeError('execute_v2 FAILED')


def get_validation_dataloader(batch_size, seq_length, seed=42):
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = seq_length

    validation_dataset = load_dataset('imdb')
    validation_dataset = validation_dataset['test'].map(
        lambda data: tokenizer(data['text'], padding='max_length', truncation=True),
        batched=True,
    )
    validation_dataset = validation_dataset.remove_columns(["text"])
    validation_dataset.set_format('torch')
    validation_dataset = validation_dataset.shuffle(seed=seed)

    return DataLoader(validation_dataset, batch_size=batch_size, drop_last=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run and test engine.')
    parser.add_argument('--engine', type=str, required=True, help='path to engine')
    parser.add_argument('--max_batches', type=int, required=False, help='maximum number of batches in dataloader')
    args = parser.parse_args()

    print('LOAD ENGINE')
    model = TrtModel(args.engine)

    print('PREPARE DATALOADER')
    validation_dataloader = get_validation_dataloader(batch_size=model.bs, seq_length=model.seq_len)
    metric = load_metric('accuracy')

    print('VALIDATION')
    logits = torch.empty([model.bs, 2], dtype=torch.float32).cuda().contiguous()
    for i, batch in enumerate(tqdm(validation_dataloader, total=len(validation_dataloader))):
        if i > args.max_batches:
            break

        label = batch['label'].cuda()
        input_ids = batch['input_ids'].cuda()

        model.run(input_ids=input_ids, logits=logits)
        predictions = torch.argmax(logits, dim=-1)

        metric.add_batch(predictions=predictions, references=label)

    metric_final = metric.compute()
    print('ACCURACY:', metric_final)
